import torch
import logging
import os, sys
from pathlib import Path
import folder_paths
import numpy as np
from .streamdiffusionwrapper import StreamDiffusionWrapper
import inspect
from PIL import Image

# Define constants for model paths
MODELS_ROOT = os.path.expanduser("/home/ryan/models")

def get_engine_dir(model_id: str) -> str:
    """Get the engine directory for a specific model"""
    # Handle full snapshot paths
    if "snapshots" in model_id:
        if "KBlueLeaf--kohaku-v2.1" in model_id:
            model_name = "KBlueLeaf"
        elif "stabilityai--sd-turbo" in model_id:
            model_name = "stabilityai"
        else:
            model_name = model_id.split('/')[-1]  # fallback behavior
    # Handle huggingface-style IDs
    elif model_id == "stabilityai/sd-turbo":
        model_name = "stabilityai"
    elif model_id == "KBlueLeaf/kohaku-v2.1":
        model_name = "KBlueLeaf"
    else:
        model_name = model_id.split('/')[-1]  # fallback behavior
        
    engine_dir = os.path.join(MODELS_ROOT, "StreamDiffusion--engines", model_name)
    print(f"Using engine directory: {engine_dir}")  # Debug print
    return engine_dir

def get_model_path(model_id: str) -> str:
    """Helper function to get the correct model path including snapshots"""
    base_path = os.path.join(MODELS_ROOT, f"models--{model_id.replace('/', '--')}")
    if not os.path.exists(base_path):
        return model_id
        
    # Get the snapshot directory
    snapshots_dir = os.path.join(base_path, "snapshots")
    if not os.path.exists(snapshots_dir):
        return model_id
        
    # Get the first (and usually only) snapshot directory
    snapshot_dirs = os.listdir(snapshots_dir)
    if not snapshot_dirs:
        return model_id
        
    return os.path.join(snapshots_dir, snapshot_dirs[0])

def get_wrapper_defaults(param_names):
    """Helper function to get default values from StreamDiffusionWrapper parameters
    Args:
        param_names (list): List of parameter names to extract
    Returns:
        dict: Dictionary of parameter names and their default values
    """
    wrapper_params = inspect.signature(StreamDiffusionWrapper).parameters
    return {name: wrapper_params[name].default for name in param_names if name in wrapper_params}

class StreamDiffusionLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA model to load."}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "The strength scale for the LoRA model. Higher values result in greater influence of the LoRA on the output."}),
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

# "latent-consistency/lcm-lora-sdv1-5"

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
                "device": (["cuda", "cpu"], {"default": "cuda", "tooltip": "Device to run inference on. CPU will be significantly slower."}),
                "dtype": (["float16", "float32"], {"default": "float16", "tooltip": "Data type for inference. float16 uses less memory but may be less precise."}),
                "device_ids": ("STRING", {"default": "", "tooltip": "Comma-separated list of device IDs for multi-GPU support. Leave empty for single GPU."}),
                "use_safety_checker": ("BOOLEAN", {"default": False, "tooltip": "Enable safety checker to filter NSFW content. May impact performance."}),
                "engine_dir": ("STRING", {"default": "engines", "tooltip": "Directory for TensorRT engine files when using tensorrt acceleration."}),
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

class StreamDiffusionModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_id_or_path": (["stabilityai/sd-turbo", "KBlueLeaf/kohaku-v2.1"], {"default": "stabilityai/sd-turbo", "tooltip": "The model identifier or path to the model to load for generation."}),
            }
        }   

    RETURN_TYPES = ("STREAMDIFFUSION_MODEL",)
    OUTPUT_TOOLTIPS = ("The loaded StreamDiffusion model.",)
    FUNCTION = "load_model"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Loads the specified StreamDiffusion model for use in generation."

    def load_model(self, model_id_or_path):
        if model_id_or_path in ["stabilityai/sd-turbo", "KBlueLeaf/kohaku-v2.1"]:
            model_path = get_model_path(model_id_or_path)
            if model_path != model_id_or_path:  # Only use local path if it exists
                model_id_or_path = model_path
                print(f"Using local model path: {model_path}")
        return (model_id_or_path,)

class StreamDiffusionConfig:
    @classmethod
    def INPUT_TYPES(s):
        defaults = get_wrapper_defaults([
            "mode", "width", "height", "acceleration", "frame_buffer_size",
            "use_tiny_vae", "cfg_type", "seed"
        ])
        
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
            "optional": {
                "opt_lora_dict": ("LORA_DICT", {"tooltip": "Optional dictionary of LoRA models to apply."}),
                "opt_acceleration_config": ("ACCELERATION_CONFIG", {"tooltip": "Optional acceleration configuration to fine-tune performance settings."}),
                "opt_similarity_filter_config": ("SIMILARITY_FILTER_CONFIG", {"tooltip": "Optional similarity filter configuration to filter out similar images."}),
                 # "opt_lcm_lora_path": ("LCM_LORA_PATH",),
                # "opt_vae_path": ("VAE_PATH",),
                "opt_device_config": ("DEVICE_CONFIG", {"tooltip": "Optional device and engine configuration settings."}),
            }
        }

    RETURN_TYPES = ("STREAM_MODEL",)
    OUTPUT_TOOLTIPS = ("The configured StreamDiffusion model.",)
    FUNCTION = "load_model"
    CATEGORY = "StreamDiffusion"
    DESCRIPTION = "Configures and initializes the StreamDiffusion model with specified settings for generation."

    def load_model(self, model, t_index_list, mode, width, height, acceleration, 
                  frame_buffer_size, use_tiny_vae, cfg_type, use_lcm_lora, seed,
                  opt_lora_dict=None, opt_acceleration_config=None,
                  #opt_lcm_lora_path=None, opt_vae_path=None,
                  opt_similarity_filter_config=None, opt_device_config=None):
        
        t_index_list = [int(x.strip()) for x in t_index_list.split(",")]
        
        # Build base configuration with all current parameters
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
        }

        if opt_lora_dict:
            config["lora_dict"] = opt_lora_dict

        # if opt_lcm_lora_path:
        #     config["lcm_lora_id"] = opt_lcm_lora_path

        # if opt_vae_path is not None:
        #     config["vae_id"] = opt_vae_path.strip() if opt_vae_path.strip() else None

        # Add acceleration config if provided
        if opt_acceleration_config:
            config.update(opt_acceleration_config)

        # Add similarity filter config if provided
        if opt_similarity_filter_config:
            config.update(opt_similarity_filter_config)

        # Add device config if provided
        if opt_device_config:
            config.update(opt_device_config)

        engine_dir = get_engine_dir(model)
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
            image_tensor = stream_model.preprocess_image(
                Image.fromarray((image[0].numpy() * 255).astype(np.uint8))
            )
            # Perform warmup iterations
            for _ in range(stream_model.batch_size - 1):
                stream_model(image=image_tensor)
            # Final generation
            output = stream_model(image=image_tensor)
        else:
            output = stream_model.txt2img()
        
        output_array = np.array(output)
        
        # Convert to tensor and normalize to 0-1 range
        output_tensor = torch.from_numpy(output_array).float() / 255.0
        
        # Ensure BHWC format
        if len(output_tensor.shape) == 3:  # If HWC
            output_tensor = output_tensor.unsqueeze(0)  # Add batch dimension -> BHWC
        
        return (output_tensor,)

NODE_CLASS_MAPPINGS = {
    "StreamDiffusionConfig": StreamDiffusionConfig,
    "StreamDiffusionAccelerationSampler": StreamDiffusionAccelerationSampler,
    "StreamDiffusionLoraLoader": StreamDiffusionLoraLoader,
    # "StreamDiffusionVaeLoader": StreamDiffusionVaeLoader,
    "StreamDiffusionAccelerationConfig": StreamDiffusionAccelerationConfig,
    "StreamDiffusionSimilarityFilterConfig": StreamDiffusionSimilarityFilterConfig,
    "StreamDiffusionDeviceConfig": StreamDiffusionDeviceConfig,
    "StreamDiffusionModelLoader": StreamDiffusionModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StreamDiffusionConfig": "StreamDiffusionConfig",
    "StreamDiffusionAccelerationSampler": "StreamDiffusionAccelerationSampler",
    "StreamDiffusionLoraLoader": "StreamDiffusionLoraLoader",
    # "StreamDiffusionVaeLoader": "StreamDiffusionVaeLoader",
    "StreamDiffusionAccelerationConfig": "StreamDiffusionAccelerationConfig",
    "StreamDiffusionSimilarityFilterConfig": "StreamDiffusionSimilarityFilterConfig", 
    "StreamDiffusionDeviceConfig": "StreamDiffusionDeviceConfig",
    "StreamDiffusionModelLoader": "StreamDiffusionModelLoader",
}
