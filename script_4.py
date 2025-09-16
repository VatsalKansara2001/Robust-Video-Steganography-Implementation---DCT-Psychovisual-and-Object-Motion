from script_3 import RobustVideoSteganography
import os
import numpy as np
import tempfile
from typing import Optional
from PIL import Image
import math

# Implement the extraction method
class RobustVideoSteganography(RobustVideoSteganography):
    """Extended with message extraction method"""
    
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

print("Extraction and quality analysis methods implemented successfully")