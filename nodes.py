import torch
import logging
import os, sys
from pathlib import Path
import folder_paths
import numpy as np
from .streamdiffusionwrapper import StreamDiffusionWrapper
import inspect
from PIL import Image


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


class StreamDiffusionLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA model to load."}),
                "strength": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "The strength scale for the LoRA model. Higher values result in greater influence of the LoRA on the output."}),
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

    RETURN_TYPES = ("STREAMDIFFUSION_MODEL",)
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
                "model": ("STREAMDIFFUSION_MODEL", {"tooltip": "The StreamDiffusion model to use for generation."}),
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
            "acceleration": acceleration,
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

class StreamDiffusionAccelerationSampler:
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

    def generate(self, stream_model, prompt, negative_prompt, num_inference_steps, 
                guidance_scale, delta, image=None):
        
        stream_model.prepare(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta
        )

        # Warmup loop for img2img mode
        if stream_model.mode == "img2img" and image is not None:
            # Handle batch of images
            batch_size = image.shape[0]
            outputs = []
            
            for i in range(batch_size):
                # Process each image in the batch
                image_tensor = stream_model.preprocess_image(
                    Image.fromarray((image[i].numpy() * 255).astype(np.uint8))
                )
                # Perform warmup iterations for each image
                for _ in range(stream_model.batch_size - 1):
                    stream_model(image=image_tensor)
                # Final generation
                output = stream_model(image=image_tensor)
                outputs.append(output)
            
            # Stack outputs into a single batch
            output_array = np.stack([np.array(img) for img in outputs], axis=0)
        else:
            output = stream_model.txt2img()
            output_array = np.array(output)
            if len(output_array.shape) == 3:  # Single image
                output_array = np.expand_dims(output_array, 0)
        
        # Convert to tensor and normalize to 0-1 range
        output_tensor = torch.from_numpy(output_array).float() / 255.0
        
        # Ensure BHWC format
        if len(output_tensor.shape) == 3:  # If HWC
            output_tensor = output_tensor.unsqueeze(0)  # Add batch dimension -> BHWC
        
        return (output_tensor,)   

class StreamDiffusionPrebuiltConfig(StreamDiffusionConfigMixin):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "engine_config": (get_engine_configs(), {"tooltip": "Select from available prebuilt engine configurations"}),
                "t_index_list": ("STRING", {"default": "39,35,30", "tooltip": "Comma-separated list of t_index values. Must match the number of steps used to build the engine."}),
            },
            "optional": s.get_optional_inputs()
        }

    RETURN_TYPES = ("STREAM_MODEL",)
    FUNCTION = "configure_model"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Configures StreamDiffusion to use existing TensorRT engines"

    def configure_model(self, engine_config, t_index_list, **optional_configs):
        # Parse configuration from the parent directory name
        config_parts = engine_config.split('--')
        
        # First part is the model name
        model_name = config_parts[0]
        # Convert model name back to path if it's a local file
        if os.path.exists(os.path.join(folder_paths.models_dir, "checkpoints", f"{model_name}.safetensors")):
            model_path = os.path.join(folder_paths.models_dir, "checkpoints", f"{model_name}.safetensors")
        else:
            model_path = model_name  # Assume it's a model ID if not a local file

        # Parse t_index_list
        t_indices = [int(x.strip()) for x in t_index_list.split(",")]
        
        # Extract engine parameters to validate t_index_list length
        engine_params = {}
        for part in config_parts[1:]:
            key, value = part.split('-', 1)
            if key == 't_index':
                engine_steps = int(value)
                if len(t_indices) != engine_steps:
                    raise ValueError(f"t_index_list length ({len(t_indices)}) must match the number of steps ({engine_steps}) used to build the engine")
            elif key == 'buffer':
                engine_params['frame_buffer_size'] = int(value)
            elif key == 'lcm_lora':
                engine_params['use_lcm_lora'] = value.lower() == 'true'
            elif key == 'tiny_vae':
                engine_params['use_tiny_vae'] = value.lower() == 'true'
            elif key == 'mode':
                engine_params['mode'] = value
                engine_params['cfg_type'] = "none" if value == "txt2img" else "self"
        
        config = {
            "model_id_or_path": model_path,
            "t_index_list": t_indices,
            "acceleration": "tensorrt",
            "width": 512,
            "height": 512,
            "use_denoising_batch": True,
            "engine_dir": os.path.join(folder_paths.models_dir, 'StreamDiffusion--engines', engine_config),
            **engine_params
        }

        # Apply optional configs using mixin method
        config = self.apply_optional_configs(config, **optional_configs)

        wrapper = StreamDiffusionWrapper(**config)
        return (wrapper,)

NODE_CLASS_MAPPINGS = {
    "StreamDiffusionConfig": StreamDiffusionConfig,
    "StreamDiffusionAccelerationSampler": StreamDiffusionAccelerationSampler,
    "StreamDiffusionLoraLoader": StreamDiffusionLoraLoader,
    # "StreamDiffusionVaeLoader": StreamDiffusionVaeLoader,
    "StreamDiffusionAccelerationConfig": StreamDiffusionAccelerationConfig,
    "StreamDiffusionSimilarityFilterConfig": StreamDiffusionSimilarityFilterConfig,
    "StreamDiffusionDeviceConfig": StreamDiffusionDeviceConfig,
    "StreamDiffusionCheckpointLoader": StreamDiffusionCheckpointLoader,
    "StreamDiffusionPrebuiltConfig": StreamDiffusionPrebuiltConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StreamDiffusionConfig": "StreamDiffusionConfig",
    "StreamDiffusionAccelerationSampler": "StreamDiffusionAccelerationSampler",
    "StreamDiffusionLoraLoader": "StreamDiffusionLoraLoader",
    # "StreamDiffusionVaeLoader": "StreamDiffusionVaeLoader",
    "StreamDiffusionAccelerationConfig": "StreamDiffusionAccelerationConfig",
    "StreamDiffusionSimilarityFilterConfig": "StreamDiffusionSimilarityFilterConfig", 
    "StreamDiffusionDeviceConfig": "StreamDiffusionDeviceConfig",
    "StreamDiffusionCheckpointLoader": "StreamDiffusionCheckpointLoader",
    "StreamDiffusionPrebuiltConfig": "StreamDiffusionPrebuiltConfig",
}