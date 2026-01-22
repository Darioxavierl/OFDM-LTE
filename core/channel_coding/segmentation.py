"""
Code Block Segmentation for LTE Turbo Coding
=============================================

3GPP TS 36.212 Section 5.1.2

Segments transport blocks into code blocks for Turbo encoding.
Maximum code block size: Z = 6144 bits (including CRC)

Key parameters:
- If B ≤ Z: No segmentation needed, single code block
- If B > Z: Segment into C code blocks, add CRC-24B to each
- Filler bits added to reach proper block sizes from interleaver table

Segmentation process:
1. Calculate number of code blocks C
2. Find suitable block sizes K+ and K- from interleaver table
3. Calculate number of filler bits F
4. Distribute filler bits to first code blocks
5. Add CRC-24B to each code block (if C > 1)
"""

import numpy as np
from typing import Tuple, List
from .crc import attach_crc24b


# =============================================================================
# Turbo Code Internal Interleaver Sizes (3GPP TS 36.212 Table 5.1.3-3)
# =============================================================================
# These are the valid code block sizes K for turbo coding
# Taken from the turbo code internal interleaver table

TURBO_INTERLEAVER_SIZES = [
    40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160,
    168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 264, 272, 280,
    288, 296, 304, 312, 320, 328, 336, 344, 352, 360, 368, 376, 384, 392, 400,
    408, 416, 424, 432, 440, 448, 456, 464, 472, 480, 488, 496, 504, 512, 528,
    544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704, 720, 736, 752, 768,
    784, 800, 816, 832, 848, 864, 880, 896, 912, 928, 944, 960, 976, 992, 1008,
    1024, 1056, 1088, 1120, 1152, 1184, 1216, 1248, 1280, 1312, 1344, 1376,
    1408, 1440, 1472, 1504, 1536, 1568, 1600, 1632, 1664, 1696, 1728, 1760,
    1792, 1824, 1856, 1888, 1920, 1952, 1984, 2016, 2048, 2112, 2176, 2240,
    2304, 2368, 2432, 2496, 2560, 2624, 2688, 2752, 2816, 2880, 2944, 3008,
    3072, 3136, 3200, 3264, 3328, 3392, 3456, 3520, 3584, 3648, 3712, 3776,
    3840, 3904, 3968, 4032, 4096, 4160, 4224, 4288, 4352, 4416, 4480, 4544,
    4608, 4672, 4736, 4800, 4864, 4928, 4992, 5056, 5120, 5184, 5248, 5312,
    5376, 5440, 5504, 5568, 5632, 5696, 5760, 5824, 5888, 5952, 6016, 6080,
    6144
]


def find_interleaver_size(min_size: int) -> int:
    """
    Find the smallest valid interleaver size >= min_size
    
    Parameters:
    -----------
    min_size : int
        Minimum required size
    
    Returns:
    --------
    int : Valid interleaver size from table
    """
    for size in TURBO_INTERLEAVER_SIZES:
        if size >= min_size:
            return size
    
    # Should never reach here for valid LTE parameters
    raise ValueError(f"No valid interleaver size found for min_size={min_size}")


def segment_code_blocks(transport_block_with_crc: np.ndarray) -> Tuple[List[np.ndarray], dict]:
    """
    Segment transport block into code blocks for Turbo coding
    
    Follows 3GPP TS 36.212 Section 5.1.2 exactly.
    
    Process:
    --------
    1. If B ≤ Z (6144): Single code block, no additional CRC
    2. If B > Z: 
       - Calculate C (number of code blocks)
       - Add CRC-24B to each code block
       - Add filler bits if needed to reach valid interleaver sizes
    
    Parameters:
    -----------
    transport_block_with_crc : np.ndarray
        Transport block with CRC-24A already attached
        Size B (includes the 24-bit CRC-24A)
    
    Returns:
    --------
    tuple : (code_blocks, metadata)
        - code_blocks: List of code block bit arrays
        - metadata: Dictionary with segmentation information
            - num_blocks: Number of code blocks (C)
            - block_sizes: List of block sizes [K1, K2, ...]
            - num_filler_bits: Total filler bits added
            - filler_positions: Where filler bits were added
            - original_size: Original transport block size
    
    Example:
    --------
    >>> tb_with_crc = np.random.randint(0, 2, 5000)  # Transport block with CRC-24A
    >>> blocks, info = segment_code_blocks(tb_with_crc)
    >>> print(f"Segmented into {info['num_blocks']} blocks")
    """
    B = len(transport_block_with_crc)
    Z = 6144  # Maximum code block size
    
    # Case 1: Small enough for single code block (no segmentation)
    if B <= Z:
        # No segmentation needed
        # Find valid interleaver size K >= B
        K = find_interleaver_size(B)
        
        # Calculate filler bits needed
        F = K - B
        
        # Create code block with filler bits at the beginning
        code_block = np.zeros(K, dtype=np.uint8)
        code_block[F:] = transport_block_with_crc
        
        # Metadata
        metadata = {
            'num_blocks': 1,
            'block_sizes': [K],
            'num_filler_bits': F,
            'filler_positions': list(range(F)) if F > 0 else [],
            'filler_per_block': [F],  # NEW: filler bits per block
            'original_size': B,
            'segmented': False
        }
        
        return [code_block], metadata
    
    # Case 2: Need segmentation into multiple code blocks
    else:
        # Calculate number of code blocks
        # L = 24 bits (CRC-24B per code block when C > 1)
        L = 24
        C = int(np.ceil(B / (Z - L)))
        
        # Calculate code block sizes
        # B' = B + C*L (total bits including CRC-24B for each block)
        B_prime = B + C * L
        
        # Find K+ (smallest valid size >= B'/C)
        K_plus = find_interleaver_size(int(np.ceil(B_prime / C)))
        
        # Find K- (largest valid size < K+)
        K_minus_idx = TURBO_INTERLEAVER_SIZES.index(K_plus) - 1
        if K_minus_idx >= 0:
            K_minus = TURBO_INTERLEAVER_SIZES[K_minus_idx]
        else:
            K_minus = K_plus  # Edge case: K+ is already the smallest
        
        # Calculate number of blocks of each size
        # C- blocks of size K-, C+ blocks of size K+
        # Such that: C-*K- + C+*K+ = B'
        delta_K = K_plus - K_minus
        
        if delta_K > 0:
            C_minus = (C * K_plus - B_prime) // delta_K
            C_plus = C - C_minus
        else:
            # All blocks same size
            C_minus = 0
            C_plus = C
        
        # Total filler bits
        F = C * K_plus - B_prime + C_minus * (K_plus - K_minus)
        
        # Now segment the transport block
        code_blocks = []
        bit_position = 0
        filler_positions = []
        block_sizes = []
        
        # Calculate how to distribute B bits across C blocks
        # Each block r gets: bits_per_block[r] information bits
        bits_per_block = []
        remaining_bits = B
        
        for r in range(C):
            if r < C_minus:
                K_r = K_minus
            else:
                K_r = K_plus
            
            # Available space for information in this block (excluding CRC-24B)
            available_space = K_r - L
            
            # Allocate bits to this block
            if r == C - 1:
                # Last block gets all remaining bits
                bits_for_this_block = remaining_bits
            else:
                # Distribute evenly
                blocks_left = C - r
                bits_for_this_block = min(available_space, remaining_bits // blocks_left)
            
            bits_per_block.append(bits_for_this_block)
            remaining_bits -= bits_for_this_block
        
        # Now create the code blocks
        filler_per_block = []  # Track filler bits per block
        
        for r in range(C):
            if r < C_minus:
                K_r = K_minus
            else:
                K_r = K_plus
            
            info_bits_this_block = bits_per_block[r]
            
            # Create code block without CRC (K_r - L bits)
            code_block_no_crc = np.zeros(K_r - L, dtype=np.uint8)
            
            # Calculate filler bits for this block
            F_r = (K_r - L) - info_bits_this_block
            filler_per_block.append(F_r)
            
            # Debug last block
            if r == C - 1:
                print(f"[segment DEBUG] Last block (r={r}): K={K_r}, info_bits={info_bits_this_block}, F_r={F_r}")
                print(f"  bit_position={bit_position}/{B}, remaining={B - bit_position}")
            
            # Add filler bits at beginning (zeros)
            # Then add information bits
            if info_bits_this_block > 0:
                code_block_no_crc[F_r:F_r + info_bits_this_block] = transport_block_with_crc[bit_position:bit_position + info_bits_this_block]
            
            # Track filler positions
            if F_r > 0:
                filler_positions.extend(range(len(filler_positions), len(filler_positions) + F_r))
            
            bit_position += info_bits_this_block
            
            # Attach CRC-24B
            code_block_with_crc = attach_crc24b(code_block_no_crc)
            
            code_blocks.append(code_block_with_crc)
            block_sizes.append(K_r)
        
        metadata = {
            'num_blocks': C,
            'block_sizes': block_sizes,
            'num_filler_bits': F,
            'filler_positions': filler_positions,
            'filler_per_block': filler_per_block,  # NEW: filler bits per block
            'original_size': B,
            'segmented': True,
            'K_plus': K_plus,
            'K_minus': K_minus,
            'C_plus': C_plus,
            'C_minus': C_minus
        }
        
        return code_blocks, metadata


def desegment_code_blocks(code_blocks: List[np.ndarray], metadata: dict) -> np.ndarray:
    """
    Reassemble code blocks back into transport block
    
    Reverses the segmentation process:
    1. Remove CRC-24B from each code block (if segmented)
    2. Remove filler bits
    3. Concatenate blocks
    
    Parameters:
    -----------
    code_blocks : List[np.ndarray]
        List of decoded code blocks
    metadata : dict
        Segmentation metadata from segment_code_blocks
    
    Returns:
    --------
    np.ndarray : Reconstructed transport block with CRC-24A
    
    Example:
    --------
    >>> blocks, info = segment_code_blocks(tb_with_crc)
    >>> reconstructed = desegment_code_blocks(blocks, info)
    >>> np.array_equal(reconstructed, tb_with_crc)
    True
    """
    B = metadata['original_size']
    
    if not metadata['segmented']:
        # Single block case
        # Remove filler bits
        F = metadata['num_filler_bits']
        code_block = code_blocks[0]
        
        # Extract original transport block (skip filler bits)
        transport_block = code_block[F:F + B]
        
        return transport_block
    
    else:
        # Multiple blocks case
        C = metadata['num_blocks']
        L = 24  # CRC-24B length
        
        # Reassemble
        transport_block = []
        
        # Calculate bits per block (same logic as segmentation)
        bits_per_block = []
        remaining_bits = B
        
        for r in range(C):
            K_r = metadata['block_sizes'][r]
            available_space = K_r - L
            
            if r == C - 1:
                bits_for_this_block = remaining_bits
            else:
                blocks_left = C - r
                bits_for_this_block = min(available_space, remaining_bits // blocks_left)
            
            bits_per_block.append(bits_for_this_block)
            remaining_bits -= bits_for_this_block
        
        # Extract information bits from each block
        for r in range(C):
            code_block = code_blocks[r]
            K_r = metadata['block_sizes'][r]
            
            # Remove CRC-24B (last 24 bits)
            code_block_no_crc = code_block[:-L]
            
            info_bits_this_block = bits_per_block[r]
            
            # Calculate filler bits for this block
            F_r = (K_r - L) - info_bits_this_block
            
            # Debug last block
            if r == C - 1:
                errors_before_crc_removal = np.sum(code_block[:F_r] != 0)  # filler should be 0
                errors_in_filler_area = np.sum(code_block_no_crc[:F_r] != 0)
                print(f"[desegment DEBUG] Last block (r={r}): K={K_r}, info_bits={info_bits_this_block}, F_r={F_r}")
                print(f"  code_block length={len(code_block)}, code_block_no_crc length={len(code_block_no_crc)}")
                print(f"  Errors in filler area (should be 0): {errors_in_filler_area}/{F_r}")
                print(f"  Extracting bits from [{F_r}:{F_r + info_bits_this_block}]")
            
            # Extract information bits (skip filler bits at beginning)
            info_bits = code_block_no_crc[F_r:F_r + info_bits_this_block]
            
            transport_block.append(info_bits)
        
        # Concatenate all blocks
        return np.concatenate(transport_block)


def get_segmentation_info(transport_block_size: int) -> dict:
    """
    Get segmentation information without actually segmenting
    
    Useful for planning rate matching and resource allocation
    
    Parameters:
    -----------
    transport_block_size : int
        Size of transport block (with CRC-24A)
    
    Returns:
    --------
    dict : Segmentation information
        - num_blocks: Number of code blocks
        - block_sizes: Sizes of code blocks
        - total_coded_bits: Total bits after turbo encoding (rate 1/3)
    """
    B = transport_block_size
    Z = 6144
    
    if B <= Z:
        K = find_interleaver_size(B)
        return {
            'num_blocks': 1,
            'block_sizes': [K],
            'total_coded_bits': 3 * K + 12  # Rate 1/3 + 12 tail bits
        }
    else:
        L = 24
        C = int(np.ceil(B / (Z - L)))
        B_prime = B + C * L
        K_plus = find_interleaver_size(int(np.ceil(B_prime / C)))
        
        K_minus_idx = TURBO_INTERLEAVER_SIZES.index(K_plus) - 1
        if K_minus_idx >= 0:
            K_minus = TURBO_INTERLEAVER_SIZES[K_minus_idx]
        else:
            K_minus = K_plus
        
        delta_K = K_plus - K_minus
        
        if delta_K > 0:
            C_minus = (C * K_plus - B_prime) // delta_K
            C_plus = C - C_minus
        else:
            C_minus = 0
            C_plus = C
        
        block_sizes = [K_minus] * C_minus + [K_plus] * C_plus
        total_coded_bits = sum(3 * K + 12 for K in block_sizes)
        
        return {
            'num_blocks': C,
            'block_sizes': block_sizes,
            'total_coded_bits': total_coded_bits
        }


if __name__ == '__main__':
    """Self-test"""
    print("=" * 70)
    print("Code Block Segmentation Self-Test")
    print("=" * 70)
    
    # Test 1: Small block (no segmentation)
    print("\n[Test 1] Small block (no segmentation)")
    tb_small = np.random.randint(0, 2, 5000, dtype=np.uint8)
    blocks, info = segment_code_blocks(tb_small)
    print(f"  Input size:      {len(tb_small)} bits")
    print(f"  Number of blocks: {info['num_blocks']}")
    print(f"  Block size:       {info['block_sizes'][0]} bits")
    print(f"  Filler bits:      {info['num_filler_bits']}")
    
    # Desegment
    reconstructed = desegment_code_blocks(blocks, info)
    assert np.array_equal(reconstructed, tb_small), "Reconstruction failed!"
    print(f"  ✓ Reconstruction successful")
    
    # Test 2: Large block (needs segmentation)
    print("\n[Test 2] Large block (needs segmentation)")
    tb_large = np.random.randint(0, 2, 20000, dtype=np.uint8)
    blocks, info = segment_code_blocks(tb_large)
    print(f"  Input size:       {len(tb_large)} bits")
    print(f"  Number of blocks: {info['num_blocks']}")
    print(f"  Block sizes:      {info['block_sizes']}")
    print(f"  Filler bits:      {info['num_filler_bits']}")
    print(f"  K+:               {info.get('K_plus', 'N/A')}")
    print(f"  K-:               {info.get('K_minus', 'N/A')}")
    
    # Desegment
    reconstructed = desegment_code_blocks(blocks, info)
    assert np.array_equal(reconstructed, tb_large), "Reconstruction failed!"
    print(f"  ✓ Reconstruction successful")
    
    print("\n" + "=" * 70)
    print("Segmentation tests passed! ✓")
    print("=" * 70)
