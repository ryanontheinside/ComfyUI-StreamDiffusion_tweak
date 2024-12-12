import torch
from PIL import Image
from streamdiffusionwrapper import StreamDiffusionWrapper
import numpy as np
import cProfile
import pstats
import time
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from test_utiles import create_profile_visualizations, timer


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

def test_img2img(num_iterations=100):
    # Create a profile output directory
    output_dir = "profile_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_path = f"{output_dir}/profile_stats_{timestamp}"

    profiler = cProfile.Profile()
    profiler.enable()
    
    total_start = time.perf_counter()
    iteration_times = []
    
    with timer("Wrapper initialization"):
        wrapper = StreamDiffusionWrapper(
            model_id_or_path="KBlueLeaf/kohaku-v2.1",
            t_index_list=[22, 32, 45],
            mode="img2img",
            output_type="pil",
            # device="cuda",
            # dtype=torch.float16,
            frame_buffer_size=1,
            width=512,
            height=512,
            # warmup=10,
            acceleration="none",
            use_lcm_lora=True,
            cfg_type="self",
            # do_add_noise=True,
            # use_denoising_batch=True,
        )

    # Load input image once
    with timer("Image loading and preprocessing"):
        try:
            input_image = Image.open("ryan.png").convert("RGB")
            input_image = input_image.resize((512, 512))
            input_tensor = wrapper.preprocess_image(input_image)
        except FileNotFoundError:
            print("Error: ryan.png not found in the current directory!")
            profiler.disable()
            return
        except Exception as e:
            print(f"Error loading image: {e}")
            profiler.disable()
            return

    prompt = "1girl, white hair, golden eyes, detailed face, masterpiece"
    negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    
    with timer("Preparation"):
        wrapper.prepare(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=1.2,
            num_inference_steps=50,
            delta=0.5
        )
    
    print("Starting warmup phase...")
    # Warmup phase
    with timer("Warmup phase"):
        for i in range(5):
            wrapper(image=input_tensor)
    
    print(f"\nStarting main test phase ({num_iterations} iterations)...")
    # Main test loop
    with timer("Main inference loop"):
        for i in range(num_iterations):
            iter_start = time.perf_counter()
            
            with timer(f"Iteration {i}"):  # Individual iteration timing
                wrapper(image=input_tensor)
            
            iter_time = time.perf_counter() - iter_start
            iteration_times.append(iter_time)
        

    total_time = time.perf_counter() - total_start
    
    # Calculate and print statistics
    avg_time = sum(iteration_times) / len(iteration_times)
    max_time = max(iteration_times)
    min_time = min(iteration_times)
    fps = 1.0 / avg_time
    
    print("\nTest Results:")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average iteration time: {avg_time:.4f} seconds")
    print(f"Maximum iteration time: {max_time:.4f} seconds")
    print(f"Minimum iteration time: {min_time:.4f} seconds")
    print(f"Average FPS: {fps:.2f}")
    
    # Save profiling results
    profiler.disable()
    
    # Save stats to file
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.dump_stats(f"{profile_path}.prof")
    
    # Create visualizations
    with timer("Stats calculation and visualization"):
        create_profile_visualizations(iteration_times, output_dir)
    
    # Print instructions for visualization
    print("\nProfile data has been saved.")
    print(f"To view interactive visualization, run:")
    print(f"snakeviz {profile_path}.prof")

if __name__ == "__main__":
    print("\nTesting img2img with Kohaku v2.1...")
    test_img2img(500)  # Default to 100 iterations
    print("\nTest completed!") 