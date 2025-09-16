# Implement main video processing methods
import subprocess
import tempfile
import shutil
from pathlib import Path
from script_2 import RobustVideoSteganography
import os
import numpy as np
from typing import List
from PIL import Image
import json


class RobustVideoSteganography(RobustVideoSteganography):
    """Extended with main video processing methods"""
    
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

print("Main embedding method implemented successfully")