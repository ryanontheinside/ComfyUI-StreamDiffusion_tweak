import torch
from PIL import Image
from streamdiffusionwrapper import StreamDiffusionWrapper
import numpy as np

def test_txt2img():
    # Initialize wrapper for txt2img
    wrapper = StreamDiffusionWrapper(
        model_id_or_path="KBlueLeaf/kohaku-v2.1",
        t_index_list=[0],
        mode="txt2img",
        output_type="pil",
        device="cuda",
        dtype=torch.float16,
        frame_buffer_size=1,
        width=512,
        height=512,
        warmup=1,
        acceleration="tensorrt",
        use_lcm_lora=True,  # Enable LCM-LoRA for faster inference
        cfg_type="none",
    )

    # Test txt2img generation
    prompt = "1girl, white hair, golden eyes, detailed face, masterpiece"
    negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    
    wrapper.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=7.0
    )
    image = wrapper.txt2img()
    
    # Save the output
    if isinstance(image, list):
        for i, img in enumerate(image):
            img.save(f"kohaku_txt2img_output_{i}.png")
    else:
        image.save("kohaku_txt2img_output.png")

def test_img2img():
    # Initialize wrapper for img2img with more aggressive settings
    wrapper = StreamDiffusionWrapper(
        model_id_or_path="KBlueLeaf/kohaku-v2.1",
        t_index_list=[1, 10],  # Very strong effect
        mode="img2img",
        output_type="pil",
        device="cuda",
        dtype=torch.float16,
        frame_buffer_size=1,
        width=512,
        height=512,
        warmup=1,
        acceleration="tensorrt",
        use_lcm_lora=True,
        cfg_type="full",  # Using full CFG
        do_add_noise=True,
    )

    # Load and preprocess input image
    try:
        input_image = Image.open("ryan.png").convert("RGB")
        input_image = input_image.resize((512, 512))
        
        print(f"Original image size: {input_image.size}")
        
        # Convert PIL to tensor manually
        input_tensor = torch.from_numpy(np.array(input_image)).float()
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)  # BHWC -> BCHW
        input_tensor = input_tensor / 255.0  # Scale to [0,1]
        input_tensor = input_tensor.to(device=wrapper.device, dtype=wrapper.dtype)
        
        print(f"Preprocessed tensor shape: {input_tensor.shape}")
        print(f"Tensor value range: {input_tensor.min():.2f} to {input_tensor.max():.2f}")
        
    except FileNotFoundError:
        print("Error: ryan.png not found in the current directory!")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Test img2img generation with extreme settings
    prompt = "1girl, white hair, golden eyes, detailed face, masterpiece, highly detailed"
    negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    
    wrapper.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=15.0,  # Very high guidance scale
        num_inference_steps=50
    )
    
    print("Starting img2img generation...")
    
    # Try to directly use the stream's img2img method
    try:
        # First warmup pass
        print("Performing warmup pass...")
        for _ in range(wrapper.batch_size):
            wrapper.stream(input_tensor)
            
        print("Starting main generation...")
        # Main generation
        image_tensor = wrapper.stream(input_tensor)
        output_image = wrapper.postprocess_image(image_tensor, output_type="pil")
    except Exception as e:
        print(f"Error during generation: {e}")
        return
        
    print("Generation completed")
    
    # Save both input and output for comparison
    input_image.save("input_image.png")
    if isinstance(output_image, list):
        for i, img in enumerate(output_image):
            img.save(f"kohaku_img2img_output_{i}.png")
    else:
        output_image.save("kohaku_img2img_output.png")

if __name__ == "__main__":
    #print("Testing txt2img with Kohaku v2.1...")
   # test_txt2img()
    #print("txt2img test completed!")

    print("\nTesting img2img with Kohaku v2.1...")
    test_img2img()
    print("img2img test completed!") 