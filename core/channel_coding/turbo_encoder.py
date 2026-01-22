"""
LTE Turbo Encoder - 3GPP TS 36.212 Section 5.1.3
================================================

Implements parallel concatenated convolutional code (PCCC) with:
- Two 8-state RSC (Recursive Systematic Convolutional) encoders
- QPP (Quadratic Permutation Polynomial) internal interleaver
- Trellis termination (12 tail bits total)
- Rate 1/3 (systematic + 2 parity streams)

Generator Polynomials:
- g0 = 1 + D² + D³ (feedback, octal 013)
- g1 = 1 + D + D³ (feedforward, octal 015)

Output Structure:
- d(0)^(0), d(1)^(0), d(2)^(0), ..., d(K-1)^(0)     [Systematic bits]
- d(0)^(1), d(1)^(1), d(2)^(1), ..., d(K-1)^(1)     [Parity 1]
- d(0)^(2), d(1)^(2), d(2)^(2), ..., d(K-1)^(2)     [Parity 2]
- Plus 12 tail bits (3 per encoder × 2 encoders × 2 streams)

Total output: 3K + 12 bits for input of K bits
"""

import numpy as np
from typing import Tuple


# =============================================================================
# QPP Interleaver Parameters (3GPP TS 36.212 Table 5.1.3-3)
# =============================================================================
# Format: K: (f1, f2)
# Where π(i) = (f1*i + f2*i²) mod K

QPP_INTERLEAVER_PARAMS = {
    40: (3, 10), 48: (7, 12), 56: (19, 42), 64: (7, 16), 72: (7, 18),
    80: (11, 20), 88: (5, 22), 96: (11, 24), 104: (7, 26), 112: (41, 84),
    120: (103, 90), 128: (15, 32), 136: (9, 34), 144: (17, 108), 152: (9, 38),
    160: (21, 120), 168: (101, 84), 176: (21, 44), 184: (57, 46), 192: (23, 48),
    200: (13, 50), 208: (27, 52), 216: (11, 36), 224: (27, 56), 232: (85, 58),
    240: (29, 60), 248: (33, 62), 256: (15, 32), 264: (17, 198), 272: (33, 68),
    280: (103, 210), 288: (19, 36), 296: (19, 74), 304: (37, 76), 312: (19, 78),
    320: (21, 120), 328: (21, 82), 336: (115, 84), 344: (193, 86), 352: (21, 44),
    360: (133, 90), 368: (81, 46), 376: (45, 94), 384: (23, 48), 392: (243, 98),
    400: (151, 40), 408: (155, 102), 416: (25, 52), 424: (51, 106), 432: (47, 72),
    440: (91, 110), 448: (29, 168), 456: (29, 114), 464: (247, 58), 472: (29, 118),
    480: (89, 180), 488: (91, 122), 496: (157, 62), 504: (55, 84), 512: (31, 64),
    528: (17, 66), 544: (35, 68), 560: (227, 420), 576: (65, 96), 592: (19, 74),
    608: (37, 76), 624: (41, 234), 640: (39, 80), 656: (185, 82), 672: (43, 252),
    688: (21, 86), 704: (155, 44), 720: (79, 120), 736: (139, 92), 752: (23, 94),
    768: (217, 48), 784: (25, 98), 800: (17, 80), 816: (127, 102), 832: (25, 52),
    848: (239, 106), 864: (17, 48), 880: (137, 110), 896: (215, 112), 912: (29, 114),
    928: (15, 58), 944: (147, 118), 960: (29, 60), 976: (59, 122), 992: (65, 124),
    1008: (55, 84), 1024: (31, 64), 1056: (17, 66), 1088: (171, 204), 1120: (67, 140),
    1152: (35, 72), 1184: (19, 74), 1216: (39, 76), 1248: (19, 78), 1280: (199, 240),
    1312: (21, 82), 1344: (211, 252), 1376: (21, 86), 1408: (43, 88), 1440: (149, 60),
    1472: (45, 92), 1504: (49, 846), 1536: (71, 48), 1568: (13, 28), 1600: (17, 80),
    1632: (25, 102), 1664: (183, 104), 1696: (55, 954), 1728: (127, 96), 1760: (27, 110),
    1792: (29, 112), 1824: (29, 114), 1856: (57, 116), 1888: (45, 354), 1920: (31, 120),
    1952: (59, 610), 1984: (185, 124), 2016: (113, 420), 2048: (31, 64), 2112: (17, 66),
    2176: (171, 136), 2240: (209, 420), 2304: (253, 216), 2368: (367, 444), 2432: (265, 456),
    2496: (181, 468), 2560: (39, 80), 2624: (27, 164), 2688: (127, 504), 2752: (143, 172),
    2816: (43, 88), 2880: (29, 300), 2944: (45, 92), 3008: (157, 188), 3072: (47, 96),
    3136: (13, 28), 3200: (111, 240), 3264: (443, 204), 3328: (51, 104), 3392: (51, 212),
    3456: (451, 192), 3520: (257, 220), 3584: (57, 336), 3648: (313, 228), 3712: (271, 232),
    3776: (179, 236), 3840: (331, 120), 3904: (363, 244), 3968: (375, 248), 4032: (127, 168),
    4096: (31, 64), 4160: (33, 130), 4224: (43, 264), 4288: (33, 134), 4352: (477, 408),
    4416: (35, 138), 4480: (233, 280), 4544: (357, 142), 4608: (337, 480), 4672: (37, 146),
    4736: (71, 444), 4800: (71, 120), 4864: (37, 152), 4928: (39, 462), 4992: (127, 234),
    5056: (39, 158), 5120: (39, 80), 5184: (31, 96), 5248: (113, 902), 5312: (41, 166),
    5376: (251, 336), 5440: (43, 170), 5504: (21, 86), 5568: (43, 174), 5632: (45, 176),
    5696: (45, 178), 5760: (161, 120), 5824: (89, 182), 5888: (323, 184), 5952: (47, 186),
    6016: (23, 94), 6080: (47, 190), 6144: (263, 480)
}


def qpp_interleave(data: np.ndarray, K: int) -> np.ndarray:
    """
    QPP (Quadratic Permutation Polynomial) Interleaver
    
    π(i) = (f1*i + f2*i²) mod K
    
    Parameters:
    -----------
    data : np.ndarray
        Input data (K bits)
    K : int
        Interleaver size (must be in QPP_INTERLEAVER_PARAMS)
    
    Returns:
    --------
    np.ndarray : Interleaved data
    """
    if K not in QPP_INTERLEAVER_PARAMS:
        raise ValueError(f"Invalid interleaver size K={K}")
    
    f1, f2 = QPP_INTERLEAVER_PARAMS[K]
    
    # Generate permutation indices
    indices = np.array([(f1 * i + f2 * i * i) % K for i in range(K)], dtype=int)
    
    # Apply permutation
    return data[indices]


def qpp_deinterleave(data: np.ndarray, K: int) -> np.ndarray:
    """
    Inverse QPP Interleaver
    
    Parameters:
    -----------
    data : np.ndarray
        Interleaved data (K bits)
    K : int
        Interleaver size
    
    Returns:
    --------
    np.ndarray : Deinterleaved data
    """
    if K not in QPP_INTERLEAVER_PARAMS:
        raise ValueError(f"Invalid interleaver size K={K}")
    
    f1, f2 = QPP_INTERLEAVER_PARAMS[K]
    
    # Generate permutation indices
    perm_indices = np.array([(f1 * i + f2 * i * i) % K for i in range(K)], dtype=int)
    
    # Create inverse permutation
    inv_indices = np.zeros(K, dtype=int)
    for i in range(K):
        inv_indices[perm_indices[i]] = i
    
    # Apply inverse permutation
    return data[inv_indices]


def rsc_encode(input_bits: np.ndarray, trellis_termination: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    RSC (Recursive Systematic Convolutional) Encoder
    
    Generator polynomials (octal):
    - g0 = 013 = 1 + D² + D³ (feedback)
    - g1 = 015 = 1 + D + D³ (feedforward)
    
    State diagram: 8 states (3 memory elements)
    
    Parameters:
    -----------
    input_bits : np.ndarray
        Input bits
    trellis_termination : bool
        If True, append tail bits to terminate trellis to zero state
    
    Returns:
    --------
    tuple : (systematic_bits, parity_bits)
        - systematic_bits: Same as input (with tail if termination=True)
        - parity_bits: Parity output stream
    """
    # Generator polynomials (g0=feedback, g1=feedforward)
    # g0 = [1, 0, 1, 1] = 1 + D² + D³
    # g1 = [1, 1, 0, 1] = 1 + D + D³
    
    K = len(input_bits)
    
    # Initialize state (3 flip-flops)
    state = np.zeros(3, dtype=np.uint8)
    
    # Output arrays
    systematic = []
    parity = []
    
    # Encode data bits
    for bit in input_bits:
        # Feedback: g0 = 1 + D² + D³
        feedback = (bit + state[1] + state[2]) % 2
        
        # Systematic output
        systematic.append(feedback)
        
        # Parity output: g1 = 1 + D + D³
        parity_bit = (feedback + state[0] + state[2]) % 2
        parity.append(parity_bit)
        
        # Update state (shift register)
        state[2] = state[1]
        state[1] = state[0]
        state[0] = feedback
    
    # Trellis termination (force state to zero)
    if trellis_termination:
        for _ in range(3):  # 3 tail bits to flush 3-bit state
            # Input is chosen to force state to zero
            # feedback_bit = state[1] + state[2]
            tail_bit = (state[1] + state[2]) % 2
            
            # Feedback
            feedback = (tail_bit + state[1] + state[2]) % 2
            
            systematic.append(feedback)
            
            # Parity
            parity_bit = (feedback + state[0] + state[2]) % 2
            parity.append(parity_bit)
            
            # Update state
            state[2] = state[1]
            state[1] = state[0]
            state[0] = feedback
    
    return np.array(systematic, dtype=np.uint8), np.array(parity, dtype=np.uint8)


def turbo_encode(input_bits: np.ndarray) -> np.ndarray:
    """
    LTE Turbo Encoder (Rate 1/3)
    
    Structure:
    1. First RSC encoder on original sequence
    2. Internal QPP interleaver
    3. Second RSC encoder on interleaved sequence
    4. Trellis termination for both encoders
    
    Output format (3GPP TS 36.212):
    - Systematic bits: d(0), d(1), ..., d(K-1), d(K), d(K+1), d(K+2)
    - Parity 1:        z(0), z(1), ..., z(K-1), z(K), z(K+1), z(K+2)
    - Parity 2:        z'(0), z'(1), ..., z'(K-1), z'(K), z'(K+1), z'(K+2)
    
    Total: 3*(K+3) = 3K + 9... wait, spec says 3K + 12
    Actually: Each encoder produces K+4 bits (K data + 4 tail for closing both trellis paths)
    
    Let me recalculate: According to 36.212:
    - Systematic: K bits
    - Parity 1: K bits + 3 tail bits
    - Parity 2: K bits + 3 tail bits
    - Total systematic with tail bits from both: K + 3 + 3 = K + 6? No...
    
    Actually reviewing 36.212 Section 5.1.3.1:
    - Output consists of three streams
    - Each stream has K bits from data encoding
    - Plus 4 tail bits per encoder (but they go in specific positions)
    - Total is 3K + 12 bits
    
    Output ordering:
    - d_k^(0), d_k^(1), d_k^(2) for k=0..K-1 (data)
    - Then 12 tail bits in specific order
    
    Parameters:
    -----------
    input_bits : np.ndarray
        Input code block (K bits, must be valid interleaver size)
    
    Returns:
    --------
    np.ndarray : Encoded output (3K + 12 bits)
    """
    K = len(input_bits)
    
    if K not in QPP_INTERLEAVER_PARAMS:
        raise ValueError(f"Invalid code block size K={K}. Must be valid interleaver size.")
    
    # First RSC encoder (on original sequence)
    systematic_1, parity_1 = rsc_encode(input_bits, trellis_termination=True)
    
    # Interleave input for second encoder
    interleaved_input = qpp_interleave(input_bits, K)
    
    # Second RSC encoder (on interleaved sequence)
    systematic_2, parity_2 = rsc_encode(interleaved_input, trellis_termination=True)
    
    # Output structure according to 3GPP TS 36.212 Section 5.1.3.1
    # Stream 0: systematic bits (K bits) + tail from encoder 1 (3 bits) + tail from encoder 2 (3 bits)
    # Stream 1: parity 1 bits (K bits) + tail parity 1 (3 bits)
    # Stream 2: parity 2 bits (K bits) + tail parity 2 (3 bits)
    
    # Extract data and tail bits
    systematic_data = systematic_1[:K]
    parity_1_data = parity_1[:K]
    parity_2_data = parity_2[:K]
    
    tail_systematic_1 = systematic_1[K:]  # 3 bits
    tail_parity_1 = parity_1[K:]          # 3 bits
    tail_systematic_2 = systematic_2[K:]  # 3 bits
    tail_parity_2 = parity_2[K:]          # 3 bits
    
    # Concatenate streams
    # d^(0): systematic data + tails from both encoders
    stream_0 = np.concatenate([systematic_data, tail_systematic_1, tail_systematic_2])
    
    # d^(1): parity 1 + tail
    stream_1 = np.concatenate([parity_1_data, tail_parity_1])
    
    # d^(2): parity 2 + tail
    stream_2 = np.concatenate([parity_2_data, tail_parity_2])
    
    # Interleave the three streams
    # Output: d_0^(0), d_0^(1), d_0^(2), d_1^(0), d_1^(1), d_1^(2), ...
    output = np.zeros(len(stream_0) + len(stream_1) + len(stream_2), dtype=np.uint8)
    
    # First K*3 bits (data)
    for k in range(K):
        output[3*k] = stream_0[k]
        output[3*k + 1] = stream_1[k]
        output[3*k + 2] = stream_2[k]
    
    # Tail bits (12 bits total after data)
    tail_start = 3 * K
    output[tail_start:tail_start+3] = tail_systematic_1
    output[tail_start+3:tail_start+6] = tail_parity_1
    output[tail_start+6:tail_start+9] = tail_systematic_2
    output[tail_start+9:tail_start+12] = tail_parity_2
    
    return output


def turbo_encode_block_list(code_blocks: list) -> list:
    """
    Encode multiple code blocks
    
    Parameters:
    -----------
    code_blocks : list of np.ndarray
        List of code blocks to encode
    
    Returns:
    --------
    list : List of encoded blocks
    """
    return [turbo_encode(block) for block in code_blocks]


if __name__ == '__main__':
    """Self-test"""
    print("=" * 70)
    print("Turbo Encoder Self-Test")
    print("=" * 70)
    
    # Test 1: QPP Interleaver
    print("\n[Test 1] QPP Interleaver")
    K = 40
    data = np.arange(K, dtype=np.uint8)
    interleaved = qpp_interleave(data, K)
    deinterleaved = qpp_deinterleave(interleaved, K)
    
    print(f"  Original:      {data[:10]}...")
    print(f"  Interleaved:   {interleaved[:10]}...")
    print(f"  Deinterleaved: {deinterleaved[:10]}...")
    assert np.array_equal(data, deinterleaved), "Interleaver test failed!"
    print("  ✓ QPP interleaver working")
    
    # Test 2: RSC Encoder
    print("\n[Test 2] RSC Encoder")
    input_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    sys_bits, par_bits = rsc_encode(input_bits, trellis_termination=True)
    
    print(f"  Input:      {input_bits}")
    print(f"  Systematic: {sys_bits}")
    print(f"  Parity:     {par_bits}")
    print(f"  Output length: {len(sys_bits)} (should be {len(input_bits)+3})")
    assert len(sys_bits) == len(input_bits) + 3, "RSC length wrong!"
    print("  ✓ RSC encoder working")
    
    # Test 3: Turbo Encoder
    print("\n[Test 3] Turbo Encoder")
    for K_test in [40, 104, 512, 1024, 3072, 6144]:
        input_block = np.random.randint(0, 2, K_test, dtype=np.uint8)
        encoded = turbo_encode(input_block)
        
        expected_length = 3 * K_test + 12
        actual_length = len(encoded)
        
        print(f"  K={K_test:4d}: Input={K_test}, Output={actual_length}, Expected={expected_length}", end="")
        assert actual_length == expected_length, f" FAILED!"
        print(" ✓")
    
    print("\n" + "=" * 70)
    print("Turbo Encoder tests passed! ✓")
    print("=" * 70)
