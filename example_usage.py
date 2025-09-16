#!/usr/bin/env python3
"""
Example Usage Script for Robust Video Steganography
Demonstrates embedding and extracting messages from video files
"""

import os
import sys
from robust_video_steganography import RobustVideoSteganography

def check_requirements():
    """Check if required dependencies are available"""
    try:
        import numpy as np
        import scipy
        from PIL import Image
        print("✓ Python dependencies available")
    except ImportError as e:
        print(f"✗ Missing Python dependency: {e}")
        return False

    # Check FFmpeg
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ FFmpeg available")
        else:
            print("✗ FFmpeg not found")
            return False
    except FileNotFoundError:
        print("✗ FFmpeg not installed or not in PATH")
        return False

    return True

def create_sample_video(output_path: str = "sample_input.mp4"):
    """
    Create a simple test video with motion for demonstration
    Uses FFmpeg to generate a test pattern
    """
    import subprocess

    # Create a test video with moving pattern
    cmd = [
        'ffmpeg', '-f', 'lavfi',
        '-i', 'testsrc2=duration=5:size=640x480:rate=25',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        output_path, '-y'
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✓ Created sample video: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to create sample video: {e}")
        return None

def example_embedding():
    """Demonstrate message embedding"""
    print("\n" + "="*50)
    print("EMBEDDING EXAMPLE")
    print("="*50)

    # Initialize steganography system
    stego = RobustVideoSteganography(psychovisual_threshold=20)

    # Test message
    secret_message = "Hello, this is a secret message hidden using DCT steganography!"
    print(f"Secret message: '{secret_message}'")
    print(f"Message length: {len(secret_message)} characters")

    # Check if sample video exists, create if not
    input_video = "sample_input.mp4"
    if not os.path.exists(input_video):
        print("Creating sample video...")
        input_video = create_sample_video(input_video)
        if not input_video:
            return False

    stego_video = "stego_output.mp4"

    try:
        print("\nStarting embedding process...")
        embed_stats = stego.embed_message(input_video, secret_message, stego_video)

        print("\n--- EMBEDDING RESULTS ---")
        print(f"Success: {embed_stats['success']}")
        print(f"Message length: {embed_stats['message_length']} characters")
        print(f"Bits embedded: {embed_stats['bits_embedded']}/{embed_stats['binary_length']}")
        print(f"Embedding ratio: {embed_stats['embedding_ratio']:.1f}%")
        print(f"Frames used: {embed_stats['frames_used']}/{embed_stats['frames_processed']}")
        print(f"Total blocks used: {embed_stats['total_blocks_used']}")

        if embed_stats['success']:
            print(f"\n✓ Stego video created: {stego_video}")
            return True
        else:
            print(f"\n✗ Embedding failed - insufficient capacity")
            return False

    except Exception as e:
        print(f"\n✗ Embedding error: {e}")
        return False

def example_extraction():
    """Demonstrate message extraction"""
    print("\n" + "="*50)
    print("EXTRACTION EXAMPLE")
    print("="*50)

    stego_video = "stego_output.mp4"

    if not os.path.exists(stego_video):
        print(f"✗ Stego video not found: {stego_video}")
        print("Please run embedding example first.")
        return False

    # Initialize steganography system
    stego = RobustVideoSteganography(psychovisual_threshold=20)

    try:
        print("Starting extraction process...")
        extracted_message = stego.extract_message(stego_video)

        print("\n--- EXTRACTION RESULTS ---")
        print(f"Extracted message: '{extracted_message}'")
        print(f"Message length: {len(extracted_message)} characters")

        # Compare with original if known
        original_message = "Hello, this is a secret message hidden using DCT steganography!"
        if extracted_message == original_message:
            print("\n✓ Message extraction SUCCESSFUL!")
            print("✓ Extracted message matches original perfectly")
        elif len(extracted_message) > 0:
            print("\n⚠ Message extraction PARTIAL")
            print(f"Similarity: {sum(a==b for a,b in zip(original_message, extracted_message))/len(original_message)*100:.1f}%")
        else:
            print("\n✗ Message extraction FAILED")
            print("No message found in video")

        return len(extracted_message) > 0

    except Exception as e:
        print(f"\n✗ Extraction error: {e}")
        return False

def example_quality_analysis():
    """Demonstrate video quality analysis"""
    print("\n" + "="*50)
    print("QUALITY ANALYSIS EXAMPLE")
    print("="*50)

    original_video = "sample_input.mp4"
    stego_video = "stego_output.mp4"

    if not os.path.exists(original_video) or not os.path.exists(stego_video):
        print("✗ Required videos not found")
        print("Please run embedding example first.")
        return False

    # Initialize steganography system
    stego = RobustVideoSteganography()

    try:
        print("Analyzing video quality...")
        quality_stats = stego.analyze_video_quality(original_video, stego_video, sample_frames=5)

        print("\n--- QUALITY ANALYSIS RESULTS ---")
        print(f"Mean PSNR: {quality_stats['mean_psnr']:.2f} dB")
        print(f"Min PSNR: {quality_stats['min_psnr']:.2f} dB") 
        print(f"Max PSNR: {quality_stats['max_psnr']:.2f} dB")
        print(f"Frames analyzed: {quality_stats['frames_analyzed']}")

        # Interpret results
        mean_psnr = quality_stats['mean_psnr']
        if mean_psnr > 50:
            print("\n✓ EXCELLENT quality (PSNR > 50 dB)")
        elif mean_psnr > 40:
            print("\n✓ GOOD quality (PSNR > 40 dB)")
        elif mean_psnr > 30:
            print("\n⚠ ACCEPTABLE quality (PSNR > 30 dB)")
        else:
            print("\n✗ POOR quality (PSNR < 30 dB)")

        return True

    except Exception as e:
        print(f"\n✗ Quality analysis error: {e}")
        return False

def comprehensive_demo():
    """Run complete demonstration"""
    print("ROBUST VIDEO STEGANOGRAPHY - COMPREHENSIVE DEMO")
    print("Implementation based on DCT Psychovisual and Object Motion")
    print("="*60)

    # Check requirements
    if not check_requirements():
        print("\n✗ Requirements check failed")
        print("Please install missing dependencies and try again")
        return

    print("\n✓ All requirements satisfied")

    # Run examples
    success_count = 0

    # Embedding
    if example_embedding():
        success_count += 1

        # Extraction (only if embedding succeeded)
        if example_extraction():
            success_count += 1

            # Quality analysis (only if both succeeded)
            if example_quality_analysis():
                success_count += 1

    # Summary
    print("\n" + "="*60)
    print("DEMONSTRATION SUMMARY")
    print("="*60)
    print(f"Tests passed: {success_count}/3")

    if success_count == 3:
        print("\n🎉 ALL TESTS PASSED!")
        print("The video steganography system is working correctly.")
        print("\nGenerated files:")
        print("- sample_input.mp4 (original video)")
        print("- stego_output.mp4 (video with hidden message)")
    elif success_count > 0:
        print("\n⚠ PARTIAL SUCCESS")
        print("Some tests passed but there may be issues.")
    else:
        print("\n❌ ALL TESTS FAILED")
        print("Please check your installation and try again.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "embed":
            example_embedding()
        elif sys.argv[1] == "extract":
            example_extraction()
        elif sys.argv[1] == "quality":
            example_quality_analysis()
        elif sys.argv[1] == "requirements":
            check_requirements()
        else:
            print("Usage: python example_usage.py [embed|extract|quality|requirements]")
    else:
        comprehensive_demo()
