
from script_1 import RobustVideoSteganography
import numpy as np
from typing import List, Tuple

class RobustVideoSteganography(RobustVideoSteganography):
    """Extended with embedding and extraction methods"""
    
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

print("Embedding and extraction methods implemented successfully")