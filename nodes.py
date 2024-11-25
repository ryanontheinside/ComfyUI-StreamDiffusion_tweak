import torch
import os
import numpy as np
import logging
import json


import comfy.model_management as mm
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))

class DownloadAndLoadStreamDiffusionModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ([ 
                    '',
                    ],),
            "device": (['cuda', 'cpu', 'mps'], ),
            },
        }

    RETURN_TYPES = ("STREAMDIFFUSIONMODEL",)
    RETURN_NAMES = ("streamdiffusion_model",)
    FUNCTION = "_loadmodel"
    CATEGORY = "StreamDiffusion"

    def _loadmodel(self, model, segmentor, device, precision):
   
        streamdiffusion_model = {}
        return (streamdiffusion_model,)

class StreamDiffusion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "streamdiffusion_model": ("STREAMDIFFUSIONMODEL",),
            },
        }

    RETURN_NAMES = ("PROCESSED_IMAGES",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "_streamdiffusion"
    CATEGORY = "StreamDiffusion"

    def __init__(self):
        self.predictor = None
        self.if_init = False

    def _streamdiffusion(
        self,
        images,
        streamdiffusion_model,
    ):
        model = streamdiffusion_model["model"]
        device = streamdiffusion_model["device"]
        model.to(device)

        processed_frames = []

        def process_frame(frame, frame_idx):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                frame = frame.to(device).float()

                processed_frames.append(frame)



        for frame_idx, img in enumerate(images):
            process_frame(img, frame_idx)

        stacked_frames = torch.stack(processed_frames, dim=0) 
        return (stacked_frames,)

NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadStreamDiffusionModel": DownloadAndLoadStreamDiffusionModel,
    "StreamDiffusion": StreamDiffusion
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadStreamDiffusionModel": "(Down)Load Streamdiffusion Model",
    "StreamDiffusion": "StreamDiffusion"
}
