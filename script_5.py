from script_4 import RobustVideoSteganography
import math
import tempfile
import os
import shutil
import numpy as np
from typing import Optional
from PIL import Image


# Create a comprehensive example and usage demonstration
def create_test_video(output_path: str, width: int = 640, height: int = 480, 
                     duration: int = 5, fps: int = 25):
    """
    Create a test video with some motion for testing steganography
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Generate frames with moving object
        frames = []
        total_frames = duration * fps
        
        for frame_num in range(total_frames):
            fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
            ax.set_xlim(0, width)
            ax.set_ylim(0, height)
            ax.set_aspect('equal')
            
            # Create background
            ax.add_patch(plt.Rectangle((0, 0), width, height, facecolor='lightgray'))
            
            # Add moving circle
            x = (frame_num / total_frames) * (width - 100) + 50
            y = height // 2 + 50 * math.sin(frame_num * 0.1)
            circle = plt.Circle((x, y), 30, color='blue')
            ax.add_patch(circle)
            
            # Add some text
            ax.text(width//2, height-50, f'Frame {frame_num+1}', 
                   ha='center', va='center', fontsize=12)
            
            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
            
            # Save frame
            frame_path = os.path.join(temp_dir, f'frame_{frame_num+1:06d}.png')
            plt.savefig(frame_path, bbox_inches='tight', pad_inches=0, 
                       facecolor='white', dpi=100)
            plt.close(fig)
            
            frames.append(frame_path)
        
        # Create video from frames
        stego = RobustVideoSteganography()
        stego.create_video_ffmpeg(temp_dir, output_path, fps)
        
        print(f"Test video created: {output_path}")
        return output_path
        
    finally:
        # Clean up temporary frames
        shutil.rmtree(temp_dir, ignore_errors=True)

def demonstration_example():
    """
    Complete demonstration of the video steganography system
    """
    print("=" * 60)
    print("ROBUST VIDEO STEGANOGRAPHY DEMONSTRATION")
    print("=" * 60)
    
    # Initialize steganography system
    stego = RobustVideoSteganography(psychovisual_threshold=20)
    
    # Test message
    secret_message = "This is a secret message hidden in the video using DCT psychovisual steganography!"
    print(f"\nSecret message to embed:")
    print(f"'{secret_message}'")
    print(f"Message length: {len(secret_message)} characters")
    
    # Create temporary files
    temp_dir = tempfile.mkdtemp()
    try:
        original_video = os.path.join(temp_dir, 'original.mp4')
        stego_video = os.path.join(temp_dir, 'stego.mp4')
        
        print(f"\nCreating test video...")
        create_test_video(original_video, width=640, height=480, duration=3, fps=25)
        
        print(f"\nEmbedding message into video...")
        # Embed message
        embed_stats = stego.embed_message(original_video, secret_message, stego_video)
        
        print(f"\n--- EMBEDDING STATISTICS ---")
        for key, value in embed_stats.items():
            print(f"{key}: {value}")
        
        if embed_stats['success']:
            print(f"\n✓ Message successfully embedded!")
            
            print(f"\nExtracting message from stego video...")
            # Extract message
            extracted_message = stego.extract_message(stego_video)
            
            print(f"\n--- EXTRACTION RESULTS ---")
            print(f"Extracted message: '{extracted_message}'")
            print(f"Original length: {len(secret_message)}")
            print(f"Extracted length: {len(extracted_message)}")
            
            # Check if extraction was successful
            if extracted_message == secret_message:
                print(f"✓ Message extraction SUCCESSFUL!")
            else:
                print(f"✗ Message extraction FAILED!")
                print(f"Similarity: {len(set(secret_message) & set(extracted_message)) / len(set(secret_message)) * 100:.1f}%")
            
            # Analyze video quality
            print(f"\nAnalyzing video quality...")
            quality_stats = stego.analyze_video_quality(original_video, stego_video)
            
            print(f"\n--- QUALITY ANALYSIS ---")
            for key, value in quality_stats.items():
                if 'psnr' in key.lower():
                    print(f"{key}: {value:.2f} dB")
                else:
                    print(f"{key}: {value}")
        else:
            print(f"\n✗ Message embedding FAILED!")
            print(f"Only {embed_stats['embedding_ratio']:.1f}% of message was embedded")
    
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED")
    print("=" * 60)

# Save the complete implementation to a file for easy use
def save_complete_implementation():
    complete_code = '''#!/usr/bin/env python3
'''

    
    with open('robust_video_steganography.py', 'w') as f:
        f.write(complete_code)
    
    print("Complete implementation saved to 'robust_video_steganography.py'")

print("Demonstration functions ready!")
print("Note: The actual demonstration requires FFmpeg to be installed for video processing.")
print("Call demonstration_example() to run the complete test.")