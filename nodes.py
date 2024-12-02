import torch
import logging
import os, sys
from pathlib import Path
import folder_paths
import numpy as np
from .streamdiffusionwrapper import StreamDiffusionWrapper

# Define constants for model paths
MODELS_ROOT = "/workspace/models"
ENGINE_DIR = os.path.join(MODELS_ROOT, "StreamDiffusion--engines")

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

def get_lcm_loras():
    #NOTE: this should ultimately point to the regular COMFYUI Lora directory. 
    """Get list of available LCM LoRAs from StreamDiffusion's model directory"""
    lcm_path = os.path.join(MODELS_ROOT, "models--latent-consistency--lcm-lora-sdv1-5")
    if not os.path.exists(lcm_path):
        return ["latent-consistency/lcm-lora-sdv1-5"]  # Default option
    
    snapshots_dir = os.path.join(lcm_path, "snapshots")
    if not os.path.exists(snapshots_dir):
        return ["latent-consistency/lcm-lora-sdv1-5"]
        
    snapshot_dirs = os.listdir(snapshots_dir)
    if not snapshot_dirs:
        return ["latent-consistency/lcm-lora-sdv1-5"]
    
    return [os.path.join(snapshots_dir, snapshot_dirs[0])]

def get_available_vaes():
    """Get list of available VAEs from StreamDiffusion's model directory"""
    # Default tiny VAE
    vaes = ["madebyollin/taesd"]
    
    # Check for local VAEs
    vae_path = os.path.join(MODELS_ROOT, "models--madebyollin--taesd")
    if os.path.exists(vae_path):
        snapshots_dir = os.path.join(vae_path, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshot_dirs = os.listdir(snapshots_dir)
            if snapshot_dirs:
                vaes.append(os.path.join(snapshots_dir, snapshot_dirs[0]))
    
    return vaes

class StreamDiffusionLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "previous_loras": ("LORA_DICT",),  # Allow chaining
            }
        }

    RETURN_TYPES = ("LORA_DICT",)
    FUNCTION = "load_lora"
    CATEGORY = "StreamDiffusion"

    def load_lora(self, lora_name, strength, previous_loras=None):
        # Initialize with previous loras if provided
        lora_dict = {} if previous_loras is None else previous_loras.copy()
        
        # Add new lora to dictionary
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_dict[lora_path] = strength
        
        return (lora_dict,)

class StreamDiffusionLcmLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lcm_lora_path": (get_lcm_loras(), {"default": "latent-consistency/lcm-lora-sdv1-5"}),
            }
        }

    RETURN_TYPES = ("LCM_LORA_PATH",)
    FUNCTION = "load_lcm_lora"
    CATEGORY = "StreamDiffusion"

    def load_lcm_lora(self, lcm_lora_path):
        return (lcm_lora_path,)

class StreamDiffusionVaeLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_path": (get_available_vaes(), {"default": "madebyollin/taesd"}),
            }
        }

    RETURN_TYPES = ("VAE_PATH",)
    FUNCTION = "load_vae"
    CATEGORY = "StreamDiffusion"

    def load_vae(self, vae_path):
        return (vae_path,)

class StreamDiffusionModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_id_or_path": (["stabilityai/sd-turbo", "KBlueLeaf/kohaku-v2.1"], {"default": "stabilityai/sd-turbo"}),
            }
        }   

    RETURN_TYPES = ("STREAMDIFFUSION_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "StreamDiffusion"

    def load_model(self, model_id_or_path):
        if model_id_or_path in ["stabilityai/sd-turbo", "KBlueLeaf/kohaku-v2.1"]:
            model_path = get_model_path(model_id_or_path)
            if model_path != model_id_or_path:  # Only use local path if it exists
                model_id_or_path = model_path
                print(f"Using local model path: {model_path}")
        return (model_id_or_path,)

class StreamDiffusionLoaderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("STREAMDIFFUSION_MODEL",),
                "t_index_list": ("STRING", {"default": "39,35"}),
                "mode": (["img2img", "txt2img"], {"default": "img2img"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "acceleration": (["none", "xformers", "tensorrt"], {"default": "tensorrt"}),
                "frame_buffer_size": ("INT", {"default": 1, "min": 1, "max": 16}),
                "use_lcm_lora": ("BOOLEAN", {"default": True}),
                "use_tiny_vae": ("BOOLEAN", {"default": True}),
                "cfg_type": (["none", "full", "self", "initialize"], {"default": "self"}),
                
            },
            "optional": {
                "lora_dict": ("LORA_DICT",),
                "lcm_lora_path": ("LCM_LORA_PATH",),
                "vae_path": ("VAE_PATH",),
                "acceleration_config": ("ACCELERATION_CONFIG",),
                "similarity_filter_config": ("SIMILARITY_FILTER_CONFIG",),
            }
        }

    RETURN_TYPES = ("STREAMDIFFUSION",)
    FUNCTION = "load_model"
    CATEGORY = "StreamDiffusion"

    def load_model(self, model, t_index_list, mode, width, height, acceleration, 
                  frame_buffer_size, use_lcm_lora, use_tiny_vae, cfg_type, 
                  lora_dict=None, lcm_lora_path=None, vae_path=None, acceleration_config=None,
                  similarity_filter_config=None):
        
        # Parse t_index_list from string to actual list
        t_index_list = [int(x.strip()) for x in t_index_list.split(",")]

        # Convert empty strings to None for optional parameters
        if vae_path is not None:
            vae_path = vae_path.strip() if vae_path.strip() else None


        # Extract additional acceleration config
        warmup = acceleration_config.get("warmup", 10) if acceleration_config else 10
        do_add_noise = acceleration_config.get("do_add_noise", True) if acceleration_config else True
        use_denoising_batch = acceleration_config.get("use_denoising_batch", True) if acceleration_config else True

        # Extract similarity filter config
        enable_similar_image_filter = similarity_filter_config.get("enable_similar_image_filter", False) if similarity_filter_config else False
        similar_image_filter_threshold = similarity_filter_config.get("similar_image_filter_threshold", 0.98) if similarity_filter_config else 0.98
        similar_image_filter_max_skip_frame = similarity_filter_config.get("similar_image_filter_max_skip_frame", 10) if similarity_filter_config else 10

        wrapper = StreamDiffusionWrapper(
            model_id_or_path=model,
            t_index_list=t_index_list,
            lora_dict=lora_dict,
            mode=mode,
            lcm_lora_id=lcm_lora_path,
            vae_id=vae_path,
            width=width,
            height=height,
            frame_buffer_size=frame_buffer_size,
            acceleration=acceleration,
            warmup=warmup,
            do_add_noise=do_add_noise,
            use_lcm_lora=use_lcm_lora,
            use_tiny_vae=use_tiny_vae,
            cfg_type=cfg_type,
            engine_dir=ENGINE_DIR,
            use_denoising_batch=use_denoising_batch,
            enable_similar_image_filter=enable_similar_image_filter,
            similar_image_filter_threshold=similar_image_filter_threshold,
            similar_image_filter_max_skip_frame=similar_image_filter_max_skip_frame
        )
        
        return (wrapper,)

class StreamDiffusionAccelerationConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "warmup": ("INT", {"default": 10, "min": 0, "max": 100}),
                "do_add_noise": ("BOOLEAN", {"default": True}),
                "use_denoising_batch": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("ACCELERATION_CONFIG",)
    FUNCTION = "get_acceleration_config"
    CATEGORY = "StreamDiffusion"

    def get_acceleration_config(self, warmup, do_add_noise, use_denoising_batch):
        return ({
            "warmup": warmup,
            "do_add_noise": do_add_noise,
            "use_denoising_batch": use_denoising_batch
        },)

class StreamDiffusionSimilarityFilterConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enable_similar_image_filter": ("BOOLEAN", {"default": False}),
                "similar_image_filter_threshold": ("FLOAT", {"default": 0.98, "min": 0.0, "max": 1.0}),
                "similar_image_filter_max_skip_frame": ("INT", {"default": 10, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("SIMILARITY_FILTER_CONFIG",)
    FUNCTION = "get_similarity_filter_config"
    CATEGORY = "StreamDiffusion"

    def get_similarity_filter_config(self, enable_similar_image_filter, similar_image_filter_threshold, similar_image_filter_max_skip_frame):
        return ({
            "enable_similar_image_filter": enable_similar_image_filter,
            "similar_image_filter_threshold": similar_image_filter_threshold,
            "similar_image_filter_max_skip_frame": similar_image_filter_max_skip_frame
        },)

class StreamDiffusionGenerateNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stream_model": ("STREAMDIFFUSION",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 20.0}),
                "delta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "StreamDiffusion"

    def generate(self, stream_model, prompt, negative_prompt, num_inference_steps, 
                guidance_scale, delta, image=None):
        
        # Prepare the model with parameters
        stream_model.prepare(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta
        )

        # Generate based on mode
        if stream_model.mode == "img2img" and image is not None:
            # Convert from BHWC to BCHW for the VAE
            image_tensor = image[0].permute(2, 0, 1)  # HWC -> CHW
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension: CHW -> BCHW
            # Ensure values are in [-1, 1] range as expected by the VAE
            image_tensor = (image_tensor * 2.0) - 1.0
            output = stream_model(image=image_tensor, prompt=prompt)
        else:
            output = stream_model(prompt=prompt)

        # Convert PIL Image or list of PIL Images to ComfyUI tensor format (BHWC)
        if isinstance(output, list):
            output = output[0]  # Take first image if list
        
        # Convert PIL Image to tensor in BHWC format
        output_tensor = torch.from_numpy(np.array(output)).float() / 255.0
        output_tensor = output_tensor.unsqueeze(0)  # Add batch dimension: HWC -> BHWC
        
        return (output_tensor,)

NODE_CLASS_MAPPINGS = {
    "StreamDiffusionLoader": StreamDiffusionLoaderNode,
    "StreamDiffusionGenerate": StreamDiffusionGenerateNode,
    "StreamDiffusionLoraLoader": StreamDiffusionLoraLoader,
    "StreamDiffusionLcmLoraLoader": StreamDiffusionLcmLoraLoader,
    "StreamDiffusionVaeLoader": StreamDiffusionVaeLoader,
    "StreamDiffusionAccelerationConfig": StreamDiffusionAccelerationConfig,
    "StreamDiffusionSimilarityFilterConfig": StreamDiffusionSimilarityFilterConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StreamDiffusionLoader": "StreamDiffusionLoader",
    "StreamDiffusionGenerate": "StreamDiffusionGenerate",
    "StreamDiffusionLoraLoader": "StreamDiffusionLoraLoader",
    "StreamDiffusionLcmLoraLoader": "StreamDiffusionLcmLoraLoader",
    "StreamDiffusionVaeLoader": "StreamDiffusionVaeLoader",
    "StreamDiffusionAccelerationConfig": "StreamDiffusionAccelerationConfig",
    "StreamDiffusionSimilarityFilterConfig": "StreamDiffusionSimilarityFilterConfig", 
}