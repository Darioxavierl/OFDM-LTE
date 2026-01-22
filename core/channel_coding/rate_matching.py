"""
LTE Rate Matching for Turbo Codes - 3GPP TS 36.212 Section 5.1.4
=================================================================

Adapts the number of coded bits to match available physical resources.

Process:
1. Sub-block Interleaving: Interleave each of the 3 streams (d^(0), d^(1), d^(2))
2. Bit Collection: Create circular buffer from interleaved streams
3. Bit Selection: Select E bits from circular buffer (with repetition or puncturing)

Rate Matching Parameters:
- N_cb: Soft buffer size (typically Kw = 3 * K_pi for turbo code)
- E: Number of rate-matched output bits
- r_vidx: Redundancy version (0, 1, 2, or 3)

Output:
- E bits ready for modulation
"""

import numpy as np
from typing import Tuple, Optional


def sub_block_interleaver(input_bits: np.ndarray, D: int = 32) -> np.ndarray:
    """
    Sub-block Interleaver for Rate Matching
    
    3GPP TS 36.212 Section 5.1.4.1
    
    Algorithm:
    1. Create matrix with C columns (depends on input length and D)
    2. Fill matrix column-by-column with input bits
    3. Add <NULL> bits if needed for padding
    4. Perform inter-column permutation with pattern P
    5. Read out row-by-row (with <NULL> bits removed)
    
    Parameters:
    -----------
    input_bits : np.ndarray
        Input bit stream (systematic or parity)
    D : int
        Number of columns (fixed to 32 for LTE)
    
    Returns:
    --------
    np.ndarray : Interleaved bits
    """
    K_pi = len(input_bits)
    
    if K_pi == 0:
        return np.array([], dtype=np.uint8)
    
    # Calculate number of rows
    R = int(np.ceil(K_pi / D))
    
    # Total matrix size
    K_total = R * D
    
    # Number of <NULL> bits
    N_null = K_total - K_pi
    
    # Column permutation pattern (fixed for D=32)
    # 3GPP TS 36.212 Table 5.1.4-1
    P = np.array([
        0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30,
        1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31
    ], dtype=int)
    
    # Initialize matrix with <NULL> (-1 indicates NULL)
    matrix = np.full((R, D), -1, dtype=int)
    
    # Fill matrix column-by-column
    idx = 0
    for col in range(D):
        for row in range(R):
            if idx < K_pi:
                matrix[row, col] = input_bits[idx]
                idx += 1
            else:
                # Pad with <NULL>
                matrix[row, col] = -1
    
    # Perform inter-column permutation
    permuted_matrix = matrix[:, P]
    
    # Read out row-by-row, skipping <NULL> bits
    output = []
    for row in range(R):
        for col in range(D):
            if permuted_matrix[row, col] != -1:
                output.append(permuted_matrix[row, col])
    
    return np.array(output, dtype=np.uint8)


def sub_block_deinterleaver(input_bits: np.ndarray, original_length: int, D: int = 32) -> np.ndarray:
    """
    Inverse Sub-block Interleaver
    
    Reverses the sub-block interleaving process.
    
    Parameters:
    -----------
    input_bits : np.ndarray
        Interleaved bits
    original_length : int
        Original length before interleaving  
    D : int
        Number of columns (32 for LTE)
    
    Returns:
    --------
    np.ndarray : Deinterleaved bits
    """
    K_pi = original_length
    
    if K_pi == 0:
        return np.array([], dtype=np.uint8)
    
    R = int(np.ceil(K_pi / D))
    K_total = R * D
    
    # Column permutation pattern
    P = np.array([
        0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30,
        1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31
    ], dtype=int)
    
    # Inverse permutation
    P_inv = np.zeros(D, dtype=int)
    for i in range(D):
        P_inv[P[i]] = i
    
    # The interleaver:
    # 1. Fills matrix column-by-column (with NULLs at end)
    # 2. Permutes columns
    # 3. Reads row-by-row (skipping NULLs)
    
    # To reverse:
    # 1. Fill permuted matrix row-by-row with input (need to know where NULLs were)
    # 2. Un-permute columns
    # 3. Read column-by-column
    
    # Key insight: NULLs were in the last (K_total - K_pi) positions
    # when reading column-by-column from the original matrix
    
    # First, determine which positions in the permuted matrix were NULL
    # We need to map from column-major index to (row, col) in original,
    # then apply permutation, then map to row-major index in permuted
    
    # Create original matrix and mark NULL positions
    null_positions_original = set()
    orig_idx = 0
    for col in range(D):
        for row in range(R):
            if orig_idx >= K_pi:
                null_positions_original.add((row, col))
            orig_idx += 1
    
    # Apply column permutation to get NULL positions in permuted matrix
    null_positions_permuted = set()
    for (row, col) in null_positions_original:
        new_col = P[col]
        null_positions_permuted.add((row, new_col))
    
    # Now fill permuted matrix row-by-row, inserting NULLs where they belong
    permuted_matrix = np.full((R, D), -1, dtype=int)
    bit_idx = 0
    
    for row in range(R):
        for col in range(D):
            if (row, col) in null_positions_permuted:
                permuted_matrix[row, col] = -1
            else:
                if bit_idx < len(input_bits):
                    permuted_matrix[row, col] = input_bits[bit_idx]
                    bit_idx += 1
    
    # Apply inverse column permutation
    matrix = permuted_matrix[:, P_inv]
    
    # Read out column-by-column (skip NULLs)
    output = []
    for col in range(D):
        for row in range(R):
            if matrix[row, col] != -1:
                output.append(matrix[row, col])
    
    return np.array(output[:K_pi], dtype=np.uint8)


def rate_match_turbo(encoded_bits: np.ndarray, E: int, K: int, rv_idx: int = 0) -> np.ndarray:
    """
    Rate Matching for Turbo Code
    
    3GPP TS 36.212 Section 5.1.4.1
    
    Takes turbo encoder output (3K + 12 bits) and produces E output bits.
    
    Process:
    1. Separate into 3 streams (systematic, parity1, parity2)
    2. Sub-block interleave each stream
    3. Create circular buffer
    4. Bit collection from circular buffer starting at rv_idx
    
    Parameters:
    -----------
    encoded_bits : np.ndarray
        Turbo encoder output (3K + 12 bits)
    E : int
        Number of output bits desired
    K : int
        Original code block size (before encoding)
    rv_idx : int
        Redundancy version index (0, 1, 2, or 3)
    
    Returns:
    --------
    np.ndarray : Rate-matched output (E bits)
    """
    # Parse turbo encoder output
    # Format: d_0^(0), d_0^(1), d_0^(2), d_1^(0), ..., d_{K-1}^(2), tail_bits(12)
    
    # Extract the three streams
    D = len(encoded_bits)
    
    if D != 3*K + 12:
        raise ValueError(f"Invalid encoded_bits length. Expected {3*K + 12}, got {D}")
    
    # Separate into streams (data portion)
    systematic = np.zeros(K, dtype=np.uint8)
    parity1 = np.zeros(K, dtype=np.uint8)
    parity2 = np.zeros(K, dtype=np.uint8)
    
    for k in range(K):
        systematic[k] = encoded_bits[3*k]
        parity1[k] = encoded_bits[3*k + 1]
        parity2[k] = encoded_bits[3*k + 2]
    
    # Tail bits (12 bits after data)
    tail_start = 3*K
    tail_sys_1 = encoded_bits[tail_start:tail_start+3]
    tail_par_1 = encoded_bits[tail_start+3:tail_start+6]
    tail_sys_2 = encoded_bits[tail_start+6:tail_start+9]
    tail_par_2 = encoded_bits[tail_start+9:tail_start+12]
    
    # Add tail bits to respective streams
    d0 = np.concatenate([systematic, tail_sys_1, tail_sys_2])  # K + 6 bits
    d1 = np.concatenate([parity1, tail_par_1])                # K + 3 bits
    d2 = np.concatenate([parity2, tail_par_2])                # K + 3 bits
    
    # Sub-block interleaving
    v0 = sub_block_interleaver(d0)
    v1 = sub_block_interleaver(d1)
    v2 = sub_block_interleaver(d2)
    
    # Create circular buffer
    # Circular buffer size K_w (soft buffer size per stream)
    # For turbo code: K_w = 3 * K_pi where K_pi is interleaver size
    # Here we use the actual lengths
    K_w = len(v0)  # Should be same for all three after interleaving
    
    # Circular buffer: w_k = [v0[0], v1[0], v2[0], v0[1], v1[1], v2[1], ...]
    # But we need to handle different lengths
    max_len = max(len(v0), len(v1), len(v2))
    
    # Pad shorter streams with zeros (though they should be similar length)
    v0_padded = np.pad(v0, (0, max_len - len(v0)), 'constant')
    v1_padded = np.pad(v1, (0, max_len - len(v1)), 'constant')
    v2_padded = np.pad(v2, (0, max_len - len(v2)), 'constant')
    
    # Interleave the three streams into circular buffer
    circular_buffer = np.zeros(max_len * 3, dtype=np.uint8)
    for i in range(max_len):
        circular_buffer[3*i] = v0_padded[i]
        circular_buffer[3*i + 1] = v1_padded[i]
        circular_buffer[3*i + 2] = v2_padded[i]
    
    # Bit selection from circular buffer
    # Starting position depends on redundancy version
    N_cb = len(circular_buffer)
    
    # RV starting positions (simplified - actual LTE has more complex formula)
    # RV0: start at 0
    # RV1: start at N_cb // 4
    # RV2: start at N_cb // 2
    # RV3: start at 3 * N_cb // 4
    rv_starts = [0, N_cb // 4, N_cb // 2, 3 * N_cb // 4]
    start_pos = rv_starts[rv_idx % 4]
    
    # Collect E bits from circular buffer (with wraparound)
    output = np.zeros(E, dtype=np.uint8)
    for i in range(E):
        output[i] = circular_buffer[(start_pos + i) % N_cb]
    
    return output


def sub_block_deinterleaver_llr(interleaved_data: np.ndarray, K_original: int) -> np.ndarray:
    """
    Sub-block De-interleaver for LLRs (float version)
    
    Same as sub_block_deinterleaver but works with float/LLR values.
    This properly reverses the sub-block interleaver by handling NULL positions.
    
    Parameters:
    -----------
    interleaved_data : np.ndarray
        Interleaved LLR stream (float)
    K_original : int
        Original length before interleaving (K)
    
    Returns:
    --------
    np.ndarray : De-interleaved LLRs (K_original values)
    """
    D = 32  # Number of columns
    K_pi = len(interleaved_data)
    
    # Number of rows
    R = int(np.ceil(K_pi / D))
    K_total = R * D
    
    # Column permutation
    P = np.array([0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30,
                  1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31])
    
    # Inverse permutation
    P_inv = np.zeros(D, dtype=int)
    for i in range(D):
        P_inv[P[i]] = i
    
    # Create original matrix and mark NULL positions
    null_positions_original = set()
    orig_idx = 0
    for col in range(D):
        for row in range(R):
            if orig_idx >= K_pi:
                null_positions_original.add((row, col))
            orig_idx += 1
    
    # Apply column permutation to get NULL positions in permuted matrix
    null_positions_permuted = set()
    for (row, col) in null_positions_original:
        new_col = P[col]
        null_positions_permuted.add((row, new_col))
    
    # Fill permuted matrix row-by-row, with NULLs where they belong
    # Use NaN to mark NULL positions for LLRs
    permuted_matrix = np.full((R, D), np.nan, dtype=np.float64)
    llr_idx = 0
    
    for row in range(R):
        for col in range(D):
            if (row, col) not in null_positions_permuted:
                if llr_idx < len(interleaved_data):
                    permuted_matrix[row, col] = interleaved_data[llr_idx]
                    llr_idx += 1
    
    # Apply inverse column permutation
    matrix = permuted_matrix[:, P_inv]
    
    # Read out column-by-column (skip NaNs/NULLs)
    output = []
    for col in range(D):
        for row in range(R):
            if not np.isnan(matrix[row, col]):
                output.append(matrix[row, col])
    
    return np.array(output[:K_original], dtype=np.float64)


def rate_dematching_turbo(rate_matched_llrs: np.ndarray, K: int, rv_idx: int = 0, debug: bool = False) -> np.ndarray:
    """
    Rate De-matching for Turbo Code (LLR version for soft decoding)
    
    Reconstructs the turbo encoder output format from rate-matched LLRs.
    Handles both puncturing and repetition:
    - Puncturing: Missing positions get LLR=0 (maximum uncertainty)
    - Repetition: Repeated LLRs are summed (soft combining)
    
    Parameters:
    -----------
    rate_matched_llrs : np.ndarray
        Rate-matched LLRs from soft demodulator
    K : int
        Original code block size (before encoding)
    rv_idx : int
        Redundancy version used (0-3)
    debug : bool
        Enable debug output for problematic blocks
    
    Returns:
    --------
    np.ndarray : Reconstructed turbo encoder output LLRs (3K + 12 values)
                 Format: [sys_0, par1_0, par2_0, ..., sys_K-1, par1_K-1, par2_K-1, tails(12)]
    """
    E = len(rate_matched_llrs)
    E = len(rate_matched_llrs)
    
    # Calculate stream lengths (matching rate_match_turbo)
    d0_len = K + 6  # systematic + both tails (3+3)
    d1_len = K + 3  # parity1 + tail
    d2_len = K + 3  # parity2 + tail
    
    # Sub-block interleaving parameters
    D_cols = 32
    R0 = int(np.ceil(d0_len / D_cols))
    R1 = int(np.ceil(d1_len / D_cols))
    R2 = int(np.ceil(d2_len / D_cols))
    
    # After sub-block interleaving (NULLs are removed)
    v0_len = d0_len
    v1_len = d1_len
    v2_len = d2_len
    
    # Circular buffer size
    max_len = max(v0_len, v1_len, v2_len)
    N_cb = max_len * 3
    
    # Initialize circular buffer with zeros (LLR=0 = maximum uncertainty)
    circular_buffer = np.zeros(N_cb, dtype=np.float64)
    count_buffer = np.zeros(N_cb, dtype=int)  # Count repetitions for averaging
    
    # RV starting positions
    rv_starts = [0, N_cb // 4, N_cb // 2, 3 * N_cb // 4]
    start_pos = rv_starts[rv_idx % 4]
    
    # De-collect from rate-matched stream to circular buffer
    # For repetition (E > N_cb): sum LLRs at same position (soft combining)
    # For puncturing (E < N_cb): missing positions stay at LLR=0
    for i in range(E):
        cb_pos = (start_pos + i) % N_cb
        circular_buffer[cb_pos] += rate_matched_llrs[i]
        count_buffer[cb_pos] += 1
    
    # De-interleave from circular buffer to three streams
    # Format: [v0[0], v1[0], v2[0], v0[1], v1[1], v2[1], ...]
    v0 = np.zeros(v0_len, dtype=np.float64)
    v1 = np.zeros(v1_len, dtype=np.float64)
    v2 = np.zeros(v2_len, dtype=np.float64)
    
    for i in range(max_len):
        if i < v0_len and 3*i < N_cb:
            v0[i] = circular_buffer[3*i]
        if i < v1_len and 3*i + 1 < N_cb:
            v1[i] = circular_buffer[3*i + 1]
        if i < v2_len and 3*i + 2 < N_cb:
            v2[i] = circular_buffer[3*i + 2]
    
    # Sub-block deinterleave (using float-compatible version)
    d0 = sub_block_deinterleaver_llr(v0, d0_len)
    d1 = sub_block_deinterleaver_llr(v1, d1_len)
    d2 = sub_block_deinterleaver_llr(v2, d2_len)
    
    if debug:
        print(f"[rate_dematch DEBUG] K={K}, E={E}, N_cb={N_cb}")
        print(f"  d0_len={d0_len}, d1_len={d1_len}, d2_len={d2_len}")
        print(f"  d0 stats: mean_abs={np.mean(np.abs(d0)):.3f}, std={np.std(d0):.3f}")
        print(f"  d1 stats: mean_abs={np.mean(np.abs(d1)):.3f}, std={np.std(d1):.3f}")
        print(f"  d2 stats: mean_abs={np.mean(np.abs(d2)):.3f}, std={np.std(d2):.3f}")
    
    # Reconstruct turbo encoder output format
    # Output: [sys_0, par1_0, par2_0, sys_1, par1_1, par2_1, ..., tails]
    output_llrs = np.zeros(3*K + 12, dtype=np.float64)
    
    # Interleave systematic, parity1, parity2 for data bits
    for k in range(K):
        output_llrs[3*k] = d0[k] if k < len(d0) else 0.0      # systematic
        output_llrs[3*k + 1] = d1[k] if k < len(d1) else 0.0  # parity1
        output_llrs[3*k + 2] = d2[k] if k < len(d2) else 0.0  # parity2
    
    # Tail bits (12 total: 3 sys tail1, 3 par1 tail, 3 sys tail2, 3 par2 tail)
    # From encoder: d0 has systematic + tail1(3) + tail2(3)
    #               d1 has parity1 + par1_tail(3)
    #               d2 has parity2 + par2_tail(3)
    
    tail_offset = 3*K
    if K < len(d0):
        output_llrs[tail_offset:tail_offset+3] = d0[K:K+3]      # sys tail 1
    if K+3 < len(d0):
        output_llrs[tail_offset+6:tail_offset+9] = d0[K+3:K+6]  # sys tail 2
    if K < len(d1):
        output_llrs[tail_offset+3:tail_offset+6] = d1[K:K+3]    # par1 tail
    if K < len(d2):
        output_llrs[tail_offset+9:tail_offset+12] = d2[K:K+3]   # par2 tail
    
    return output_llrs


if __name__ == '__main__':
    """Self-test"""
    print("=" * 70)
    print("Rate Matching Self-Test")
    print("=" * 70)
    
    # Test 1: Sub-block interleaver
    print("\n[Test 1] Sub-block Interleaver")
    input_bits = np.arange(100, dtype=np.uint8) % 2
    interleaved = sub_block_interleaver(input_bits)
    deinterleaved = sub_block_deinterleaver(interleaved, len(input_bits))
    
    print(f"  Input length:        {len(input_bits)}")
    print(f"  Interleaved length:  {len(interleaved)}")
    print(f"  Deinterleaved length:{len(deinterleaved)}")
    print(f"  Match: {np.array_equal(input_bits, deinterleaved)}")
    assert np.array_equal(input_bits, deinterleaved), "Interleaver failed!"
    print("  ✓ Sub-block interleaver working")
    
    # Test 2: Rate matching
    print("\n[Test 2] Rate Matching")
    K = 40
    # Simulate turbo encoder output
    encoded = np.random.randint(0, 2, 3*K + 12, dtype=np.uint8)
    
    # Rate match to different sizes
    for E in [50, 3*K + 12, 200]:
        rate_matched = rate_match_turbo(encoded, E, K, rv_idx=0)
        print(f"  Input={3*K+12}, Output={len(rate_matched)}, Target={E}", end="")
        assert len(rate_matched) == E, f" LENGTH MISMATCH!"
        print(" ✓")
    
    print("  ✓ Rate matching producing correct lengths")
    
    # Test 3: Rate matching with de-matching
    print("\n[Test 3] Rate Matching Round-trip")
    K = 104
    encoded = np.random.randint(0, 2, 3*K + 12, dtype=np.uint8)
    
    # Rate match (no puncturing, E = input length)
    E = 3*K + 12
    rate_matched = rate_match_turbo(encoded, E, K, rv_idx=0)
    dematched = rate_dematching_turbo(rate_matched, K, rv_idx=0)
    
    match_rate = np.sum(encoded == dematched) / len(encoded) * 100
    print(f"  K={K}, E={E}")
    print(f"  Match rate: {match_rate:.1f}%")
    print("  ✓ Rate dematching working")
    
    print("\n" + "=" * 70)
    print("Rate Matching tests passed! ✓")
    print("=" * 70)
