

import time
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

import functools
import time
from contextlib import contextmanager

import functools
import time
from contextlib import contextmanager
import inspect

def profiled_section(description, start_time=None):
    """This function creates detailed entries in the .prof file"""
    # Get caller information
    caller_frame = inspect.currentframe().f_back
    caller_info = inspect.getframeinfo(caller_frame)
    
    # Create detailed timing info
    current_time = time.perf_counter()
    elapsed = current_time - start_time if start_time else 0
    
    # This function name will appear in profiler with location info
    profiled_section.__name__ = (
        f"TIMER_{description}"
        f"_at_{caller_info.filename.split('/')[-1]}"
        f"_line_{caller_info.lineno}"
    )
    
    return elapsed, current_time

@contextmanager
def timer(description, detail_level='basic'):
    """
    Enhanced timer with detailed profiling
    Args:
        description: Label for this timing section
        detail_level: 'basic' or 'detailed' profiling info
    """
    # Create uniquely named function for this timing section
    timer_func = functools.partial(profiled_section, description)
    
    # Start timing
    start = time.perf_counter()
    _, section_start = timer_func(start)  # Start marker in profile
    
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        _, section_end = timer_func(start)  # End marker in profile
        
        # Print immediate feedback
        print(f"{description}: {elapsed:.4f} seconds")


def create_profile_visualizations(iteration_times, output_dir="profile_results"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot iteration times
    plt.figure(figsize=(12, 6))
    plt.plot(iteration_times, marker='o', alpha=0.5)
    plt.title('Iteration Times Over Run')
    plt.xlabel('Iteration Number')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.savefig(f"{output_dir}/iteration_times_{timestamp}.png")
    plt.close()

    # Create histogram of iteration times
    plt.figure(figsize=(10, 6))
    sns.histplot(iteration_times, bins=30, kde=True)
    plt.title('Distribution of Iteration Times')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Count')
    plt.savefig(f"{output_dir}/time_distribution_{timestamp}.png")
    plt.close()

    # Create box plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=iteration_times)
    plt.title('Iteration Times Box Plot')
    plt.ylabel('Time (seconds)')
    plt.savefig(f"{output_dir}/boxplot_{timestamp}.png")
    plt.close()