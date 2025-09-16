
import numpy as np
import os
import struct
from typing import Tuple, List, Optional
from PIL import Image, ImageSequence
from scipy.fftpack import dct, idct
import hashlib
import json
import math

class RobustVideoSteganography:
    def __init__(self, psychovisual_threshold: int = 20):
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

print("Video steganography class methods defined successfully")