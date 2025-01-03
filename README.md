# ComfyUI-StreamDiffusion

##

### Usage in ComfyUI
It is recommended to use the StreamDiffusionCheckpoint loader along with the StreamDiffusionConfig and StreamDiffusionSampler nodes.  The other included nodes are either still in development, or may have specific use cases such as StreamDiffusionLPCheckpointLoader, and StreamDiffusionTensorRTEngineLoader - TensorRT acceleration is currently not supported by these nodes when used in combination with [ComfyUI-SAM2-Realtime](https://github.com/pschroedl/ComfyUI-SAM2-Realtime).

### Example Usage
Example .json files found in `examples/` can be opened in ComfyUI or loaded straight into ComfyStream, with the exception of those in workflow_format/, which will only work in ComfyUI as they contain layout information that ComfyStream doesn't recognize.

Many of the ComfyUI workflows in the examples/ folder require the [ComfyUI-SAM2-Realtime](https://github.com/pschroedl/ComfyUI-SAM2-Realtime) nodes to be installed, as well as various [StableDiffusion checkpoints](https://huggingface.co/pschroedl/comfystream_checkpoints/tree/main) and [LoRAs](https://huggingface.co/pschroedl/comfystream_checkpoints/tree/main).  Checkpoints and LoRAs are expected to be downloaded to your ComfyUI installation under the `models/checkpoints/` folder, and `models/loras/` folders, respectively.

### Implementation
This set of ComfyUI nodes relies upon [a wrapper](https://github.com/pschroedl/StreamDiffusion/blob/main/utils/wrapper.py) around StableDiffusion from https://github.com/cumulo-autumn/StreamDiffusion. We have [forked StreamDiffusion](https://github.com/pschroedl/StreamDiffusion) only to change some of the requirements versions in order to ensure compatibility with [ComfyUI-SAM2-Realtime](https://github.com/pschroedl/ComfyUI-SAM2-Realtime).
Thanks to [@ryanontheinside](https://github.com/ryanontheinside) for the bulk of this ComfyUI custom node implementation.
