import torch
import logging
import os, sys
from pathlib import Path
import folder_paths
import numpy as np
from .streamdiffusionwrapper import StreamDiffusionWrapper
import inspect
from PIL import Image
import time
import folder_paths

# Define constants for model paths
MODELS_ROOT = folder_paths.models_dir

ENGINE_DIR = os.path.join(folder_paths.models_dir, "StreamDiffusion--engines")

def get_wrapper_defaults(param_names):
    """Helper function to get default values from StreamDiffusionWrapper parameters
    Args:
        param_names (list): List of parameter names to extract
    Returns:
        dict: Dictionary of parameter names and their default values
    """
    wrapper_params = inspect.signature(StreamDiffusionWrapper).parameters
    defaults = {}
    for name in param_names:
        if name not in wrapper_params:
            continue
        #override engine_dir to be the models/StreamDiffusion--engines directory as opposed to the wrapper's default
        if name == 'engine_dir': 
            defaults[name] = os.path.join(folder_paths.models_dir, 'StreamDiffusion--engines')
        else:
            defaults[name] = wrapper_params[name].default
    return defaults

def get_engine_configs():
    """Scan StreamDiffusion--engines directory for available engine configurations"""
    if not os.path.exists(ENGINE_DIR):
        return []
    
    configs = []
    # Look for parent directories that contain complete engine sets
    for parent_dir in os.listdir(ENGINE_DIR):
        parent_path = os.path.join(ENGINE_DIR, parent_dir)
        if not os.path.isdir(parent_path):
            continue
            
        # Look for subdirectories containing the engines
        has_unet = False
        has_vae = False
        
        for subdir in os.listdir(parent_path):
            subdir_path = os.path.join(parent_path, subdir)
            if not os.path.isdir(subdir_path):
                continue
                
            # Check for engines
            if os.path.exists(os.path.join(subdir_path, "unet.engine")):
                has_unet = True
            if os.path.exists(os.path.join(subdir_path, "vae_encoder.engine")) and \
               os.path.exists(os.path.join(subdir_path, "vae_decoder.engine")):
                has_vae = True
        
        # Only add if we found both UNet and VAE engines
        if has_unet and has_vae:
            configs.append(parent_dir)
    
    return configs

class StreamDiffusionTensorRTEngineLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "engine_name": (get_engine_configs(), ),
            }
        }

    RETURN_TYPES = ("SDMODEL",)
    FUNCTION = "load_engine"
    CATEGORY = "StreamDiffusion"

    def load_engine(self, engine_name):
        return (os.path.join(ENGINE_DIR, engine_name),)

class StreamDiffusionLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA model to load."}),
                "strength": ("FLOAT", {"default": 0.5, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "The strength scale for the LoRA model. Higher values result in greater influence of the LoRA on the output."}),
            },
            "optional": {
                "previous_loras": ("LORA_DICT", {"tooltip": "Optional dictionary of previously loaded LoRAs to which the new LoRA will be added. Use this to combine multiple LoRAs."}),
            }
        }

    RETURN_TYPES = ("LORA_DICT",)
    OUTPUT_TOOLTIPS = ("Dictionary of loaded LoRA models.",)
    FUNCTION = "load_lora"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Loads a LoRA (Low-Rank Adaptation) model and adds it to the existing LoRA dictionary for application to the pipeline."

    def load_lora(self, lora_name, strength, previous_loras=None):
        # Initialize with previous loras if provided
        lora_dict = {} if previous_loras is None else previous_loras.copy()
        
        # Add new lora to dictionary
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_dict[lora_path] = strength
        
        return (lora_dict,)

class StreamDiffusionVaeLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"), ),
            }
        }

    RETURN_TYPES = ("VAE_PATH",)
    FUNCTION = "load_vae"
    CATEGORY = "StreamDiffusion"

    def load_vae(self, vae_name):
        vae_path = folder_paths.get_full_path("vae", vae_name)
        return (vae_path,)

class StreamDiffusionCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "checkpoint": (folder_paths.get_filename_list("checkpoints"), ),
            }
        }

    RETURN_TYPES = ("SDMODEL",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "StreamDiffusion"

    def load_checkpoint(self, checkpoint):
        checkpoint_path = folder_paths.get_full_path("checkpoints", checkpoint)
        return (checkpoint_path,)

class StreamDiffusionAccelerationConfig:
    @classmethod
    def INPUT_TYPES(s):
        defaults = get_wrapper_defaults(["warmup", "do_add_noise", "use_denoising_batch"])
        return {
            "required": {
                "warmup": ("INT", {"default": defaults["warmup"], "min": 0, "max": 100, "tooltip": "The number of warmup steps to perform before actual inference. Increasing this may improve stability at the cost of speed."}),
                "do_add_noise": ("BOOLEAN", {"default": defaults["do_add_noise"], "tooltip": "Whether to add noise during denoising steps. Enable this to allow the model to generate diverse outputs."}),
                "use_denoising_batch": ("BOOLEAN", {"default": defaults["use_denoising_batch"], "tooltip": "Whether to use batch denoising for performance optimization."}),
            }
        }

    RETURN_TYPES = ("ACCELERATION_CONFIG",)
    OUTPUT_TOOLTIPS = ("Configuration settings for acceleration.",)
    FUNCTION = "get_acceleration_config"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Configures acceleration settings for the StreamDiffusion model to optimize performance."

    def get_acceleration_config(self, warmup, do_add_noise, use_denoising_batch):
        return ({
            "warmup": warmup,
            "do_add_noise": do_add_noise,
            "use_denoising_batch": use_denoising_batch
        },)

class StreamDiffusionSimilarityFilterConfig:
    @classmethod
    def INPUT_TYPES(s):
        defaults = get_wrapper_defaults([
            "enable_similar_image_filter",
            "similar_image_filter_threshold",
            "similar_image_filter_max_skip_frame"
        ])
        return {
            "required": {
                "enable_similar_image_filter": ("BOOLEAN", {"default": defaults["enable_similar_image_filter"], "tooltip": "Enable filtering out images that are too similar to previous outputs."}),
                "similar_image_filter_threshold": ("FLOAT", {"default": defaults["similar_image_filter_threshold"], "min": 0.0, "max": 1.0, "tooltip": "Threshold determining how similar an image must be to previous outputs to be filtered out (0.0 to 1.0)."}),
                "similar_image_filter_max_skip_frame": ("INT", {"default": defaults["similar_image_filter_max_skip_frame"], "min": 0, "max": 100, "tooltip": "Maximum number of frames to skip when filtering similar images."}),
            }
        }

    RETURN_TYPES = ("SIMILARITY_FILTER_CONFIG",)
    OUTPUT_TOOLTIPS = ("Configuration settings for similarity filtering.",)
    FUNCTION = "get_similarity_filter_config"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Configures similarity filtering to prevent generating images that are too similar to previous outputs."

    def get_similarity_filter_config(self, enable_similar_image_filter, similar_image_filter_threshold, similar_image_filter_max_skip_frame):
        return ({
            "enable_similar_image_filter": enable_similar_image_filter,
            "similar_image_filter_threshold": similar_image_filter_threshold,
            "similar_image_filter_max_skip_frame": similar_image_filter_max_skip_frame
        },)

class StreamDiffusionDeviceConfig:
    @classmethod
    def INPUT_TYPES(s):
        defaults = get_wrapper_defaults(["device", "device_ids", "dtype", "use_safety_checker", "engine_dir"])
        return {
            "required": {
                "device": (["cuda", "cpu"], {"default": defaults["device"], "tooltip": "Device to run inference on. CPU will be significantly slower."}),
                "dtype": (["float16", "float32"], {"default": "float16" if defaults["dtype"] == torch.float16 else "float32", "tooltip": "Data type for inference. float16 uses less memory but may be less precise."}),
                "device_ids": ("STRING", {"default": str(defaults["device_ids"] or ""), "tooltip": "Comma-separated list of device IDs for multi-GPU support. Leave empty for single GPU."}),
                "use_safety_checker": ("BOOLEAN", {"default": defaults["use_safety_checker"], "tooltip": "Enable safety checker to filter NSFW content. May impact performance."}),
                "engine_dir": ("STRING", {"default": defaults["engine_dir"], "tooltip": "Directory for TensorRT engine files when using tensorrt acceleration."}),
            }
        }

    RETURN_TYPES = ("DEVICE_CONFIG",)
    OUTPUT_TOOLTIPS = ("Configuration settings for device and engine parameters.",)
    FUNCTION = "get_device_config"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Configures device, engine, and safety checker settings for the StreamDiffusion model."

    def get_device_config(self, device, dtype, device_ids, use_safety_checker, engine_dir):
        # Convert device_ids string to list if provided
        device_ids = [int(x.strip()) for x in device_ids.split(",")] if device_ids.strip() else None
        
        return ({
            "device": device,
            "dtype": torch.float32 if dtype == "float32" else torch.float16,
            "device_ids": device_ids,
            "use_safety_checker": use_safety_checker,
            "engine_dir": engine_dir,
        },)

class StreamDiffusionAccelerationSampler:
    # Add class-level storage
    _current_model = None
    _current_config = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stream_model": ("STREAM_MODEL", {"tooltip": "The configured StreamDiffusion model to use for generation."}),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "The text prompt to guide image generation."}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Text prompt specifying undesired aspects to avoid in the generated image."}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100, "tooltip": "The number of denoising steps. More steps often yield better results but take longer."}),
                "guidance_scale": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 20.0, "step": 0.01, "tooltip": "Controls the strength of the guidance. Higher values make the image more closely match the prompt."}),
                "delta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1, "tooltip": "Delta multiplier for virtual residual noise, affecting image diversity."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "The input image for image-to-image generation mode."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("The generated image.",)
    FUNCTION = "generate"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Generates images using the configured StreamDiffusion model and specified prompts and settings."

    def __init__(self):
        self.profile_file = "/home/ryan/comfyRealtime/ComfyUI/custom_nodes/ComfyUI-StreamDiffusion_tweak/profiling.txt"
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.profile_file), exist_ok=True)

    def log_profile(self, message):
        with open(self.profile_file, 'a') as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {message}\n")
    
    def get_model_config(self, stream_model, prompt, negative_prompt, num_inference_steps, 
                        guidance_scale, delta):
        """Create a configuration dictionary for comparing model states"""
        return {
            'model_id': getattr(stream_model, 'model_id_or_path', None),
            'mode': getattr(stream_model, 'mode', None),
            'acceleration': getattr(stream_model, 'acceleration', None),
            'frame_buffer_size': getattr(stream_model, 'frame_buffer_size', None),
            'width': getattr(stream_model, 'width', None),
            'height': getattr(stream_model, 'height', None),
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'delta': delta
        }

    def generate(self, stream_model, prompt, negative_prompt, num_inference_steps, 
                guidance_scale, delta, image=None):
        
        start_time = time.perf_counter()
        self.log_profile(f"\n=== New Generation ===")
        
        # Create current configuration
        current_config = self.get_model_config(stream_model, prompt, negative_prompt, 
                                             num_inference_steps, guidance_scale, delta)
        
        # Check if we need to update the model
        new_model = self.__class__._current_model is None or self.__class__._current_config != current_config
        if (new_model):
            self.log_profile("Model configuration changed or first run - initializing model")
            self.__class__._current_model = stream_model
            self.__class__._current_config = current_config
            
            # Time the preparation step
            prep_start = time.perf_counter()
            stream_model.prepare(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                delta=delta
            )
            prep_time = time.perf_counter() - prep_start
            self.log_profile(f"Model preparation took {prep_time:.3f}s")

        else:
            self.log_profile("Using existing model configuration")
            stream_model = self.__class__._current_model

        if stream_model.mode == "img2img" and image is not None:
            # Handle batch of images
            batch_size = image.shape[0]
            outputs = []
            self.log_profile(f"Processing batch of {batch_size} images in img2img mode")
            
            for i in range(batch_size):
                batch_start = time.perf_counter()
                self.log_profile(f"Processing batch image {i+1}/{batch_size}")
                
                # Preprocess timing
                preprocess_start = time.perf_counter()
                image_tensor = stream_model.preprocess_image(
                    Image.fromarray((image[i].numpy() * 255).astype(np.uint8))
                )
                preprocess_time = time.perf_counter() - preprocess_start
                self.log_profile(f"Image preprocessing took {preprocess_time:.3f}s")

                # Warmup timing
                if new_model:
                    warmup_start = time.perf_counter()
                    for _ in range(stream_model.batch_size - 1):
                        stream_model(image=image_tensor)
                    warmup_time = time.perf_counter() - warmup_start
                    self.log_profile(f"Warmup iterations ({stream_model.batch_size-1} steps) took {warmup_time:.3f}s")

                # Generation timing
                generation_start = time.perf_counter()
                output = stream_model(image=image_tensor)
                generation_time = time.perf_counter() - generation_start
                self.log_profile(f"Image generation took {generation_time:.3f}s")
                
                outputs.append(output)
                batch_total = time.perf_counter() - batch_start
                self.log_profile(f"Batch {i+1} completed in {batch_total:.3f}s")
            
            # Stack outputs
            stack_start = time.perf_counter()
            output_array = np.stack([np.array(img) for img in outputs], axis=0)
            stack_time = time.perf_counter() - stack_start
            self.log_profile(f"Output stacking took {stack_time:.3f}s")
            
        else:
            # Text to image generation
            self.log_profile("Performing txt2img generation")
            generation_start = time.perf_counter()
            output = stream_model.txt2img()
            generation_time = time.perf_counter() - generation_start
            self.log_profile(f"Txt2img generation took {generation_time:.3f}s")
            
            output_array = np.array(output)
            if len(output_array.shape) == 3:  # Single image
                output_array = np.expand_dims(output_array, 0)
        
        # Convert to tensor and normalize
        tensor_start = time.perf_counter()
        output_tensor = torch.from_numpy(output_array).float() / 255.0
        
        # Ensure BHWC format
        if len(output_tensor.shape) == 3:  # If HWC
            output_tensor = output_tensor.unsqueeze(0)  # Add batch dimension -> BHWC
            
        tensor_time = time.perf_counter() - tensor_start
        self.log_profile(f"Tensor conversion took {tensor_time:.3f}s")
        
        total_time = time.perf_counter() - start_time
        self.log_profile(f"Total generation time: {total_time:.3f}s")
        self.log_profile("=== Generation Complete ===\n")
        
        return (output_tensor,)

class StreamDiffusionConfigMixin:
    @staticmethod
    def get_optional_inputs():
        return {
            "opt_lora_dict": ("LORA_DICT", {"tooltip": "Optional dictionary of LoRA models to apply."}),
            "opt_acceleration_config": ("ACCELERATION_CONFIG", {"tooltip": "Optional acceleration configuration to fine-tune performance settings."}),
            "opt_similarity_filter_config": ("SIMILARITY_FILTER_CONFIG", {"tooltip": "Optional similarity filter configuration to filter out similar images."}),
            "opt_device_config": ("DEVICE_CONFIG", {"tooltip": "Optional device and engine configuration settings."}),
            # "opt_lcm_lora_path": ("LCM_LORA_PATH",),
            # "opt_vae_path": ("VAE_PATH",),
        }

    @staticmethod
    def apply_optional_configs(config, opt_lora_dict=None, opt_acceleration_config=None,
                             opt_similarity_filter_config=None, opt_device_config=None):
        if opt_lora_dict:
            config["lora_dict"] = opt_lora_dict

        if opt_acceleration_config:
            config.update(opt_acceleration_config)

        if opt_similarity_filter_config:
            config.update(opt_similarity_filter_config)

        if opt_device_config:
            config.update(opt_device_config)

        # if opt_lcm_lora_path:
        #     config["lcm_lora_id"] = opt_lcm_lora_path

        # if opt_vae_path is not None:
        #     config["vae_id"] = opt_vae_path.strip() if opt_vae_path.strip() else None
        
        return config

    @staticmethod
    def get_base_config():
        return get_wrapper_defaults([
            "mode", "width", "height", "acceleration", "frame_buffer_size",
            "use_tiny_vae", "cfg_type", "seed", "engine_dir"
        ])  

class StreamDiffusionConfig(StreamDiffusionConfigMixin):
    @classmethod
    def INPUT_TYPES(s):
        defaults = s.get_base_config()
        
        return {
            "required": {
                "model": ("SDMODEL", {"tooltip": "The StreamDiffusion model to use for generation."}),
                "t_index_list": ("STRING", {"default": "39,35,30", "tooltip": "Comma-separated list of t_index values determining at which steps to output images."}),
                "mode": (["img2img", "txt2img"], {"default": defaults["mode"], "tooltip": "Generation mode: image-to-image or text-to-image. NoteL txt2img requires cfg_type of 'none'"}),
                "width": ("INT", {"default": defaults["width"], "min": 64, "max": 2048, "tooltip": "The width of the generated images."}),
                "height": ("INT", {"default": defaults["height"], "min": 64, "max": 2048, "tooltip": "The height of the generated images."}),
                "acceleration": (["none", "xformers", "tensorrt"], {"default": defaults["acceleration"], "tooltip": "Acceleration method to optimize performance."}),
                "frame_buffer_size": ("INT", {"default": defaults["frame_buffer_size"], "min": 1, "max": 16, "tooltip": "Size of the frame buffer for batch denoising. Increasing this can improve performance at the cost of higher memory usage."}),
                "use_tiny_vae": ("BOOLEAN", {"default": defaults["use_tiny_vae"], "tooltip": "Use a TinyVAE model for faster decoding with slight quality tradeoff."}),
                "cfg_type": (["none", "full", "self", "initialize"], {"default": defaults["cfg_type"], "tooltip": """Classifier-Free Guidance type to control how guidance is applied:
- none: No guidance, fastest but may reduce quality
- full: Full guidance on every step, highest quality but slowest
- self: Self-guidance using previous frame, good balance of speed/quality
- initialize: Only apply guidance on first frame, fast with decent quality"""}),
                "use_lcm_lora": ("BOOLEAN", {"default": True, "tooltip": "Enable use of LCM-LoRA for latent consistency."}),
                "seed": ("INT", {"default": defaults["seed"], "min": -1, "max": 100000000, "tooltip": "Seed for generation. Use -1 for random seed."}),
            },
            "optional": s.get_optional_inputs()
        }

    RETURN_TYPES = ("STREAM_MODEL",)
    OUTPUT_TOOLTIPS = ("The configured StreamDiffusion model.",)
    FUNCTION = "load_model"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Configures and initializes the StreamDiffusion model with specified settings for generation."

    def load_model(self, model, t_index_list, mode, width, height, acceleration, 
                  frame_buffer_size, use_tiny_vae, cfg_type, use_lcm_lora, seed,
                  **optional_configs):
        
        t_index_list = [int(x.strip()) for x in t_index_list.split(",")]
        
        # Build base configuration
        config = {
            "model_id_or_path": model,
            "t_index_list": t_index_list,
            "mode": mode,
            "width": width,
            "height": height,
            "acceleration": "None",
            "frame_buffer_size": frame_buffer_size,
            "use_tiny_vae": use_tiny_vae,
            "cfg_type": cfg_type,
            "use_lcm_lora": use_lcm_lora,
            "seed": seed,
            "engine_dir": self.get_base_config()["engine_dir"]
        }

        # Apply optional configs using mixin method
        config = self.apply_optional_configs(config, **optional_configs)

        model_name = os.path.splitext(os.path.basename(model))[0] if os.path.isfile(model) else model.split('/')[-1]
        parent_name = f"{model_name}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--mode-{mode}--t_index-{len(t_index_list)}--buffer-{frame_buffer_size}"
        config["engine_dir"] = os.path.join(
            config["engine_dir"],
            parent_name
        )
        wrapper = StreamDiffusionWrapper(**config)

        return (wrapper,)

class StreamDiffusionPrebuiltConfig(StreamDiffusionConfigMixin):
    @classmethod
    def INPUT_TYPES(s):
        defaults = s.get_base_config()
        return {
            "required": {
                "engine_config": (get_engine_configs(), {"tooltip": "Select from available prebuilt engine configurations"}),
                "t_index_list": ("STRING", {"default": "39,35,30", "tooltip": "Comma-separated list of t_index values. Must match the number of steps used to build the engine."}),
                "mode": (["img2img", "txt2img"], {"default": defaults["mode"], "tooltip": "Generation mode: image-to-image or text-to-image"}),
                "frame_buffer_size": ("INT", {"default": defaults["frame_buffer_size"], "min": 1, "max": 16, "tooltip": "Size of the frame buffer. Must match what was used when building engines."}),
                "width": ("INT", {"default": defaults["width"], "min": 64, "max": 2048, "tooltip": "Must match the width used when building engines"}),
                "height": ("INT", {"default": defaults["height"], "min": 64, "max": 2048, "tooltip": "Must match the height used when building engines"}),
            },
            "optional": {
                **s.get_optional_inputs(),
                "model": ("SDMODEL", ),
            }
        }

    RETURN_TYPES = ("STREAM_MODEL",)
    FUNCTION = "configure_model"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Configures StreamDiffusion to use existing TensorRT engines"

    def configure_model(self, engine_config, t_index_list, mode, frame_buffer_size, width, height, model=None, **optional_configs):
        # Convert t_index_list from string to list of ints
        t_index_list = [int(x.strip()) for x in t_index_list.split(",")]
        
        # Get the engine directory path
        engine_dir = os.path.join(ENGINE_DIR, engine_config)
        
        # Build base configuration
        config = {
            "model_id_or_path": model if model is not None else engine_config,
            "mode": mode,
            "acceleration": "tensorrt",
            "frame_buffer_size": frame_buffer_size,
            "t_index_list": t_index_list,
            
            "width": width,
            "height": height,
            "use_denoising_batch": True,
            "use_tiny_vae": True  # Assuming TinyVAE was used in engine building
        }

        # Apply optional configs using mixin method
        config = self.apply_optional_configs(config, **optional_configs)
        config["engine_dir"] = engine_dir
        wrapper = StreamDiffusionWrapper(**config)
        return (wrapper,)

NODE_CLASS_MAPPINGS = {
    "StreamDiffusionConfig": StreamDiffusionConfig,
    "StreamDiffusionAccelerationSampler": StreamDiffusionAccelerationSampler,
    "StreamDiffusionLoraLoader": StreamDiffusionLoraLoader,
    "StreamDiffusionAccelerationConfig": StreamDiffusionAccelerationConfig,
    "StreamDiffusionSimilarityFilterConfig": StreamDiffusionSimilarityFilterConfig,
    "StreamDiffusionDeviceConfig": StreamDiffusionDeviceConfig,
    "StreamDiffusionCheckpointLoader": StreamDiffusionCheckpointLoader,
    "StreamDiffusionPrebuiltConfig": StreamDiffusionPrebuiltConfig,
    "StreamDiffusionTensorRTEngineLoader": StreamDiffusionTensorRTEngineLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StreamDiffusionConfig": "StreamDiffusionConfig",
    "StreamDiffusionAccelerationSampler": "StreamDiffusionAccelerationSampler",
    "StreamDiffusionLoraLoader": "StreamDiffusionLoraLoader",
    "StreamDiffusionAccelerationConfig": "StreamDiffusionAccelerationConfig",
    "StreamDiffusionSimilarityFilterConfig": "StreamDiffusionSimilarityFilterConfig", 
    "StreamDiffusionDeviceConfig": "StreamDiffusionDeviceConfig",
    "StreamDiffusionCheckpointLoader": "StreamDiffusionCheckpointLoader",
    "StreamDiffusionPrebuiltConfig": "StreamDiffusionPrebuiltConfig",
}