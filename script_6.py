# Create the complete implementation file
complete_implementation = '''#!/usr/bin/env python3


import numpy as np
import os
import struct
from typing import Tuple, List, Optional
from PIL import Image, ImageSequence
from scipy.fftpack import dct, idct
import hashlib
import json
import math
import subprocess
import tempfile
import shutil
from pathlib import Path

class RobustVideoSteganography:
    
    def __init__(self, psychovisual_threshold: int = 20):
        """
        Initialize the steganography system
        
        Args:
            psychovisual_threshold: Threshold for DCT coefficient modification (T value from paper)
        """
        self.psychovisual_threshold = psychovisual_threshold
        self.motion_threshold = 7  # Maximum motion value as per paper
        self.block_size = 8  # 8x8 DCT blocks
        
        # Selected DCT coefficients in middle frequency (from paper)
        # These are coefficients at positions 11, 12, 13, 14, 17, 16 in zig-zag order
        self.selected_positions = [11, 12, 13, 14, 17, 16]
        
        # Zig-zag order for DCT coefficients (8x8 block)
        self.zigzag_order = [
            (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
            (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
            (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
            (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
            (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
            (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
            (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
            (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
        ]

    def dct2(self, block: np.ndarray) -> np.ndarray:
        """2D DCT transform"""
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    def idct2(self, block: np.ndarray) -> np.ndarray:
        """2D Inverse DCT transform"""
        return idct(idct(block.T, norm='ortho').T, norm='ortho')
    
    def zigzag_scan(self, block: np.ndarray) -> np.ndarray:
        """Convert 8x8 block to 1D array using zig-zag order"""
        return np.array([block[i, j] for i, j in self.zigzag_order])
    
    def inverse_zigzag_scan(self, vector: np.ndarray) -> np.ndarray:
        """Convert 1D array back to 8x8 block using zig-zag order"""
        block = np.zeros((8, 8))
        for idx, (i, j) in enumerate(self.zigzag_order):
            if idx < len(vector):
                block[i, j] = vector[idx]
        return block

    def detect_motion_blocks(self, frame1: np.ndarray, frame2: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect blocks with object motion based on motion vector magnitude
        Returns list of (row, col) positions of blocks with motion <= motion_threshold
        """
        if frame1.shape != frame2.shape:
            frame2 = np.resize(frame2, frame1.shape)
        
        height, width = frame1.shape
        motion_blocks = []
        
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                block1 = frame1[i:i+self.block_size, j:j+self.block_size].astype(float)
                block2 = frame2[i:i+self.block_size, j:j+self.block_size].astype(float)
                
                # Calculate motion vector magnitude
                diff = block1 - block2
                motion_magnitude = np.sqrt(np.sum(diff**2))
                
                # Select blocks with motion <= threshold (as per paper)
                if motion_magnitude <= self.motion_threshold:
                    motion_blocks.append((i, j))
        
        return motion_blocks

    def text_to_binary(self, text: str) -> str:
        """Convert text message to binary string"""
        binary = ''.join(format(ord(char), '08b') for char in text)
        # Add delimiter to mark end of message
        binary += '1111111111111110'  # 16-bit delimiter
        return binary

    def binary_to_text(self, binary: str) -> str:
        """Convert binary string back to text"""
        # Find delimiter
        delimiter = '1111111111111110'
        if delimiter in binary:
            binary = binary[:binary.index(delimiter)]
        
        # Convert binary to text
        text = ''
        for i in range(0, len(binary), 8):
            if i + 8 <= len(binary):
                byte = binary[i:i+8]
                if len(byte) == 8:
                    text += chr(int(byte, 2))
        return text

    def embed_in_block(self, dct_block: np.ndarray, message_bits: str) -> np.ndarray:
        """
        Embed message bits into DCT block using psychovisual threshold
        Based on Algorithm 3 from the research paper
        """
        # Convert to zig-zag vector
        D = self.zigzag_scan(dct_block)
        
        # Setup thresholds f and s based on Algorithm 2 from paper
        T = self.psychovisual_threshold
        
        bit_index = 0
        for u in range(3):  # Process 3 pairs of coefficients
            if bit_index >= len(message_bits):
                break
                
            pos1 = self.selected_positions[u * 2]     # Even positions: 0, 2, 4
            pos2 = self.selected_positions[u * 2 + 1] # Odd positions: 1, 3, 5
            
            # Set thresholds based on coefficient signs
            f = T if D[pos1] >= 0 else -T
            s = T if D[pos2] >= 0 else -T
            
            message_bit = int(message_bits[bit_index])
            
            if message_bit == 1:
                if abs(D[pos1]) < abs(D[pos2]):
                    # Swap and modify
                    temp = D[pos1]
                    D[pos1] = D[pos2] + s
                    D[pos2] = temp
                else:
                    D[pos1] = D[pos1] + f
            else:  # message_bit == 0
                if abs(D[pos1]) < abs(D[pos2]):
                    D[pos1] = D[pos1] + s
                else:
                    # Swap and modify
                    temp = D[pos1]
                    D[pos1] = D[pos2]
                    D[pos2] = temp + f
            
            bit_index += 1
        
        # Convert back to 8x8 block
        modified_block = self.inverse_zigzag_scan(D)
        return modified_block
    
    def extract_from_block(self, dct_block: np.ndarray) -> str:
        """
        Extract message bits from DCT block
        Based on Algorithm 4 from the research paper
        """
        # Convert to zig-zag vector
        D = self.zigzag_scan(dct_block)
        
        extracted_bits = ''
        for u in range(3):  # Process 3 pairs of coefficients
            pos1 = self.selected_positions[u * 2]     # Even positions: 0, 2, 4
            pos2 = self.selected_positions[u * 2 + 1] # Odd positions: 1, 3, 5
            
            # Extract bit based on comparison (Algorithm 4)
            if D[pos1] < D[pos2]:
                extracted_bits += '1'
            else:
                extracted_bits += '0'
        
        return extracted_bits
    
    def embed_message_in_frame(self, frame: np.ndarray, message_bits: str, 
                              motion_blocks: List[Tuple[int, int]], 
                              start_bit: int) -> Tuple[np.ndarray, int]:
        """
        Embed message bits into a single frame using selected motion blocks
        Returns modified frame and next bit position
        """
        modified_frame = frame.copy().astype(float)
        current_bit = start_bit
        
        for block_row, block_col in motion_blocks:
            if current_bit >= len(message_bits):
                break
            
            # Extract 8x8 block
            block = frame[block_row:block_row+8, block_col:block_col+8].astype(float)
            
            # Apply DCT
            dct_block = self.dct2(block)
            
            # Embed bits (3 bits per block)
            end_bit = min(current_bit + 3, len(message_bits))
            bits_to_embed = message_bits[current_bit:end_bit]
            
            if bits_to_embed:
                # Embed bits in DCT coefficients
                modified_dct = self.embed_in_block(dct_block, bits_to_embed)
                
                # Apply inverse DCT
                reconstructed_block = self.idct2(modified_dct)
                
                # Clip values to valid range
                reconstructed_block = np.clip(reconstructed_block, 0, 255)
                
                # Replace block in frame
                modified_frame[block_row:block_row+8, block_col:block_col+8] = reconstructed_block
                
                current_bit = end_bit
        
        return modified_frame.astype(np.uint8), current_bit

    def extract_message_from_frame(self, frame: np.ndarray, 
                                  motion_blocks: List[Tuple[int, int]]) -> str:
        """
        Extract message bits from a single frame using selected motion blocks
        """
        extracted_bits = ''
        
        for block_row, block_col in motion_blocks:
            # Extract 8x8 block
            block = frame[block_row:block_row+8, block_col:block_col+8].astype(float)
            
            # Apply DCT
            dct_block = self.dct2(block)
            
            # Extract bits from DCT coefficients
            bits = self.extract_from_block(dct_block)
            extracted_bits += bits
        
        return extracted_bits

    def extract_frames_ffmpeg(self, video_path: str, output_dir: str) -> List[str]:
        """
        Extract frames from video using FFmpeg
        Returns list of frame file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Use FFmpeg to extract frames as PNG
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', 'scale=-1:-1',  # Keep original resolution
            f'{output_dir}/frame_%06d.png',
            '-y'  # Overwrite existing files
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to extract frames: {e.stderr}")
        
        # Get list of extracted frame files
        frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
        return [os.path.join(output_dir, f) for f in frame_files]
    
    def create_video_ffmpeg(self, frame_dir: str, output_video: str, fps: int = 25):
        """
        Create video from frames using FFmpeg
        """
        cmd = [
            'ffmpeg', '-r', str(fps),
            '-i', f'{frame_dir}/frame_%06d.png',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',  # High quality
            output_video,
            '-y'
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to create video: {e.stderr}")
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video information using FFprobe
        """
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            info = json.loads(result.stdout)
            
            # Extract video stream info
            video_stream = None
            for stream in info['streams']:
                if stream['codec_type'] == 'video':
                    video_stream = stream
                    break
            
            if video_stream is None:
                raise Exception("No video stream found")
            
            return {
                'fps': eval(video_stream.get('r_frame_rate', '25/1')),
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'duration': float(info['format'].get('duration', 0))
            }
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to get video info: {e.stderr}")

    def embed_message(self, input_video: str, message: str, output_video: str) -> dict:
        """
        Main method to embed message into video
        
        Args:
            input_video: Path to input MP4 video
            message: Text message to embed
            output_video: Path to output video
            
        Returns:
            Dictionary with embedding statistics
        """
        print(f"Starting message embedding process...")
        print(f"Input video: {input_video}")
        print(f"Message length: {len(message)} characters")
        
        # Convert message to binary
        binary_message = self.text_to_binary(message)
        print(f"Binary message length: {len(binary_message)} bits")
        
        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            frames_dir = os.path.join(temp_dir, 'frames')
            stego_frames_dir = os.path.join(temp_dir, 'stego_frames')
            os.makedirs(stego_frames_dir, exist_ok=True)
            
            # Extract frames
            print("Extracting frames from video...")
            frame_files = self.extract_frames_ffmpeg(input_video, frames_dir)
            print(f"Extracted {len(frame_files)} frames")
            
            # Get video info
            video_info = self.get_video_info(input_video)
            fps = video_info['fps']
            
            # Process frames for embedding
            current_bit = 0
            total_blocks_used = 0
            frames_used = 0
            
            for i, frame_file in enumerate(frame_files):
                if current_bit >= len(binary_message):
                    # Copy remaining frames without modification
                    shutil.copy(frame_file, os.path.join(stego_frames_dir, f'frame_{i+1:06d}.png'))
                    continue
                
                # Load frame
                frame_img = Image.open(frame_file).convert('L')  # Convert to grayscale
                frame = np.array(frame_img)
                
                # For first frame, just copy (reference frame for motion detection)
                if i == 0:
                    prev_frame = frame
                    shutil.copy(frame_file, os.path.join(stego_frames_dir, f'frame_{i+1:06d}.png'))
                    continue
                
                # Detect motion blocks
                motion_blocks = self.detect_motion_blocks(prev_frame, frame)
                
                if len(motion_blocks) > 0:
                    # Embed message in current frame
                    modified_frame, next_bit = self.embed_message_in_frame(
                        frame, binary_message, motion_blocks, current_bit
                    )
                    
                    # Save modified frame
                    modified_img = Image.fromarray(modified_frame.astype(np.uint8), mode='L')
                    modified_img.save(os.path.join(stego_frames_dir, f'frame_{i+1:06d}.png'))
                    
                    blocks_used = min(len(motion_blocks), (len(binary_message) - current_bit + 2) // 3)
                    total_blocks_used += blocks_used
                    current_bit = next_bit
                    frames_used += 1
                    
                    print(f"Frame {i+1}: Used {blocks_used}/{len(motion_blocks)} motion blocks, "
                          f"embedded bits {current_bit}/{len(binary_message)}")
                else:
                    # No motion blocks, copy original frame
                    shutil.copy(frame_file, os.path.join(stego_frames_dir, f'frame_{i+1:06d}.png'))
                
                prev_frame = frame
            
            # Create output video
            print("Creating output video...")
            self.create_video_ffmpeg(stego_frames_dir, output_video, int(fps))
            
            # Embedding statistics
            embedding_ratio = (current_bit / len(binary_message)) * 100 if binary_message else 0
            
            stats = {
                'message_length': len(message),
                'binary_length': len(binary_message),
                'bits_embedded': current_bit,
                'embedding_ratio': embedding_ratio,
                'frames_processed': len(frame_files),
                'frames_used': frames_used,
                'total_blocks_used': total_blocks_used,
                'success': current_bit >= len(binary_message)
            }
            
            print(f"Embedding completed: {embedding_ratio:.1f}% of message embedded")
            return stats

    def extract_message(self, stego_video: str, max_frames: Optional[int] = None) -> str:
        """
        Main method to extract hidden message from video
        
        Args:
            stego_video: Path to stego video file
            max_frames: Maximum number of frames to process (None for all)
            
        Returns:
            Extracted message text
        """
        print(f"Starting message extraction process...")
        print(f"Stego video: {stego_video}")
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            frames_dir = os.path.join(temp_dir, 'frames')
            
            # Extract frames
            print("Extracting frames from stego video...")
            frame_files = self.extract_frames_ffmpeg(stego_video, frames_dir)
            
            if max_frames:
                frame_files = frame_files[:max_frames]
            
            print(f"Processing {len(frame_files)} frames")
            
            # Extract bits from frames
            extracted_bits = ''
            frames_processed = 0
            
            for i, frame_file in enumerate(frame_files):
                # Load frame
                frame_img = Image.open(frame_file).convert('L')  # Convert to grayscale
                frame = np.array(frame_img)
                
                # For first frame, just store as reference
                if i == 0:
                    prev_frame = frame
                    continue
                
                # Detect motion blocks (same as embedding)
                motion_blocks = self.detect_motion_blocks(prev_frame, frame)
                
                if len(motion_blocks) > 0:
                    # Extract message bits from current frame
                    frame_bits = self.extract_message_from_frame(frame, motion_blocks)
                    extracted_bits += frame_bits
                    frames_processed += 1
                    
                    print(f"Frame {i+1}: Extracted {len(frame_bits)} bits from {len(motion_blocks)} blocks")
                
                prev_frame = frame
                
                # Check if we have the delimiter (end of message marker)
                delimiter = '1111111111111110'
                if delimiter in extracted_bits:
                    print(f"Found message delimiter at bit position {extracted_bits.index(delimiter)}")
                    break
            
            print(f"Total bits extracted: {len(extracted_bits)}")
            print(f"Frames processed: {frames_processed}")
            
            # Convert binary to text
            if extracted_bits:
                message = self.binary_to_text(extracted_bits)
                print(f"Extracted message: '{message}'")
                return message
            else:
                print("No message bits found")
                return ""

    def calculate_psnr(self, original: np.ndarray, modified: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio between original and modified frames
        """
        mse = np.mean((original.astype(float) - modified.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        return psnr
    
    def analyze_video_quality(self, original_video: str, stego_video: str, 
                            sample_frames: int = 10) -> dict:
        """
        Analyze quality metrics between original and stego video
        """
        print("Analyzing video quality...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            orig_dir = os.path.join(temp_dir, 'original')
            stego_dir = os.path.join(temp_dir, 'stego')
            
            # Extract sample frames
            orig_frames = self.extract_frames_ffmpeg(original_video, orig_dir)
            stego_frames = self.extract_frames_ffmpeg(stego_video, stego_dir)
            
            # Sample frames evenly
            step = max(1, len(orig_frames) // sample_frames)
            sampled_indices = list(range(0, len(orig_frames), step))[:sample_frames]
            
            psnr_values = []
            
            for idx in sampled_indices:
                if idx < len(stego_frames):
                    # Load frames
                    orig_frame = np.array(Image.open(orig_frames[idx]).convert('L'))
                    stego_frame = np.array(Image.open(stego_frames[idx]).convert('L'))
                    
                    # Calculate PSNR
                    psnr = self.calculate_psnr(orig_frame, stego_frame)
                    psnr_values.append(psnr)
            
            return {
                'mean_psnr': np.mean(psnr_values),
                'min_psnr': np.min(psnr_values),
                'max_psnr': np.max(psnr_values),
                'frames_analyzed': len(psnr_values)
            }


def example_usage():
    """
    Example usage of the video steganography system
    """
    # Initialize steganography system
    stego = RobustVideoSteganography(psychovisual_threshold=20)
    
    # Example 1: Embed message
    input_video = "input.mp4"
    secret_message = "This is a secret message!"
    stego_video = "stego_output.mp4"
    
    try:
        # Embed message
        embed_stats = stego.embed_message(input_video, secret_message, stego_video)
        print("Embedding statistics:", embed_stats)
        
        if embed_stats['success']:
            # Extract message
            extracted_message = stego.extract_message(stego_video)
            print(f"Extracted message: '{extracted_message}'")
            
            # Analyze quality
            quality_stats = stego.analyze_video_quality(input_video, stego_video)
            print("Quality statistics:", quality_stats)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    example_usage()
'''

# Save to file
with open('robust_video_steganography.py', 'w') as f:
    f.write(complete_implementation)

print("✓ Complete implementation saved to 'robust_video_steganography.py'")
print(f"File size: {len(complete_implementation)} characters")