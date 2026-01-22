"""
LTE Turbo Decoder - Log-MAP/BCJR Algorithm
==========================================

3GPP TS 36.212 Section 5.1.3 (Decoder side)

Implements iterative decoding of Turbo codes using:
- BCJR (Bahl-Cocke-Jelinek-Raviv) algorithm
- Log-MAP for numerical stability
- Two constituent RSC decoders
- Extrinsic information exchange
- 8 iterations (standard for LTE)

Input: Soft values (LLRs) from rate de-matching
Output: Hard decisions (decoded bits)

LLR Convention:
- LLR > 0 → bit = 0
- LLR < 0 → bit = 1
- |LLR| = confidence

Performance Modes:
- USE_MAX_LOG_MAP = True:  Fast mode (10-100x faster, ~0.5dB loss)
- USE_MAX_LOG_MAP = False: Exact mode (slower but optimal)
"""

import numpy as np
from typing import Tuple

# Performance mode switch
# True = Max-Log-MAP (fast approximation, ~0.5dB loss)
# False = True Log-MAP (exact but slower)
USE_MAX_LOG_MAP = True  # Default to fast mode

def set_decoder_mode(use_max_log_map: bool = True):
    """
    Configure Turbo decoder performance mode.
    
    Parameters:
    -----------
    use_max_log_map : bool
        True = Fast mode (Max-Log-MAP, 10-100x faster, ~0.5dB loss)
        False = Exact mode (True Log-MAP, optimal but slower)
    
    Example:
    --------
    >>> from core.channel_coding.turbo_decoder import set_decoder_mode
    >>> set_decoder_mode(True)   # Fast mode for large images
    >>> set_decoder_mode(False)  # Exact mode for best performance
    """
    global USE_MAX_LOG_MAP
    USE_MAX_LOG_MAP = use_max_log_map
    mode = "Max-Log-MAP (fast)" if use_max_log_map else "True Log-MAP (exact)"
    print(f"Turbo Decoder mode set to: {mode}")

# Try relative import first (when used as module)
try:
    from .turbo_encoder import qpp_interleave, qpp_deinterleave
except ImportError:
    # Fallback for direct execution
    from turbo_encoder import qpp_interleave, qpp_deinterleave


def log_sum_exp(a: float, b: float) -> float:
    """
    Numerically stable log(exp(a) + exp(b))
    
    Uses: log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
    
    Parameters:
    -----------
    a, b : float
        Values to sum in log domain
    
    Returns:
    --------
    float : log(exp(a) + exp(b))
    """
    # Handle -inf cases
    if np.isinf(a) and a < 0:
        return b
    if np.isinf(b) and b < 0:
        return a
    
    if a > b:
        return a + np.log1p(np.exp(b - a))
    else:
        return b + np.log1p(np.exp(a - b))


def max_star(a: float, b: float) -> float:
    """
    Max* operation: max*(a,b) = log(exp(a) + exp(b))
    
    Two implementations:
    - Max-Log-MAP: max*(a,b) ≈ max(a,b) [FAST, ~0.5dB loss]
    - True Log-MAP: max*(a,b) = log(exp(a) + exp(b)) [SLOW, optimal]
    
    Controlled by global USE_MAX_LOG_MAP flag.
    
    Parameters:
    -----------
    a, b : float
        Values
    
    Returns:
    --------
    float : max*(a,b)
    """
    if USE_MAX_LOG_MAP:
        # Max-Log-MAP approximation (10-100x faster)
        return max(a, b)
    else:
        # True Log-MAP (exact but slower)
        return log_sum_exp(a, b)


class LogMAPDecoder:
    """
    Log-MAP BCJR Decoder for 8-state RSC code
    
    Generator polynomials:
    - g0 = 013 (octal) = 1 + D² + D³ (feedback)
    - g1 = 015 (octal) = 1 + D + D³ (feedforward)
    
    States: 8 (3 memory elements)
    """
    
    def __init__(self):
        """Initialize decoder with trellis structure"""
        self.num_states = 8
        self.num_memory = 3
        
        # Build trellis structure
        self._build_trellis()
    
    def _build_trellis(self):
        """
        Build trellis structure for 8-state RSC code
        
        Match encoder implementation exactly:
        - State array: [s0, s1, s2] where s0 is D⁰ (most recent), s2 is D² (oldest)
        - State integer: bit 2 = s0, bit 1 = s1, bit 0 = s2
        - g0 = 1 + D² + D³ (feedback) → input + s1 + s2
        - g1 = 1 + D + D³ (parity) → feedback + s0 + s2
        """
        self.next_state = np.zeros((self.num_states, 2), dtype=int)
        self.output_systematic = np.zeros((self.num_states, 2), dtype=int)
        self.output_parity = np.zeros((self.num_states, 2), dtype=int)
        
        for state in range(self.num_states):
            # Extract state bits (match encoder array indexing)
            s0 = (state >> 2) & 1  # D⁰ (most recent)
            s1 = (state >> 1) & 1  # D¹
            s2 = state & 1         # D² (oldest)
            
            for input_bit in range(2):
                # Feedback: g0 = 1 + D² + D³ means input ⊕ D¹ ⊕ D²
                # In encoder: feedback = (bit + state[1] + state[2]) % 2
                feedback = (input_bit + s1 + s2) % 2
                
                # Systematic output equals feedback
                sys_out = feedback
                
                # Parity: g1 = 1 + D + D³ means feedback ⊕ D⁰ ⊕ D²
                # In encoder: parity = (feedback + state[0] + state[2]) % 2
                par_out = (feedback + s0 + s2) % 2
                
                # Next state after shift
                # encoder: state[2]=state[1], state[1]=state[0], state[0]=feedback
                next_s0 = feedback
                next_s1 = s0
                next_s2 = s1
                
                next_state_val = (next_s0 << 2) | (next_s1 << 1) | next_s2
                
                self.next_state[state, input_bit] = next_state_val
                self.output_systematic[state, input_bit] = sys_out
                self.output_parity[state, input_bit] = par_out
    
    def decode(self, 
               llr_systematic: np.ndarray,
               llr_parity: np.ndarray,
               llr_apriori: np.ndarray = None,
               return_extrinsic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Log-MAP BCJR Decoding
        
        Parameters:
        -----------
        llr_systematic : np.ndarray
            LLRs for systematic bits (K bits)
        llr_parity : np.ndarray
            LLRs for parity bits (K bits)
        llr_apriori : np.ndarray, optional
            A priori LLRs from other decoder (K bits)
            If None, initialized to zeros
        return_extrinsic : bool
            If True, return extrinsic information
            If False, return a posteriori LLRs
        
        Returns:
        --------
        tuple : (decoded_bits, llr_output)
            - decoded_bits: Hard decisions (K bits)
            - llr_output: Extrinsic LLRs or a posteriori LLRs
        """
        K = len(llr_systematic)
        
        if llr_apriori is None:
            llr_apriori = np.zeros(K)
        
        # Initialize forward and backward metrics
        alpha = np.full((K + 1, self.num_states), -np.inf)
        beta = np.full((K + 1, self.num_states), -np.inf)
        
        # Initial state = 0 (probability = 1, log-prob = 0)
        alpha[0, 0] = 0.0
        
        # Final state = 0 (after trellis termination)
        beta[K, 0] = 0.0
        
        # Compute branch metrics
        gamma = self._compute_gamma(llr_systematic, llr_parity, llr_apriori)
        
        # Forward recursion
        for k in range(K):
            for next_state in range(self.num_states):
                max_val = -np.inf
                for prev_state in range(self.num_states):
                    for input_bit in range(2):
                        if self.next_state[prev_state, input_bit] == next_state:
                            val = alpha[k, prev_state] + gamma[k, prev_state, input_bit]
                            max_val = max_star(max_val, val)
                alpha[k + 1, next_state] = max_val
        
        # Backward recursion
        for k in range(K - 1, -1, -1):
            for prev_state in range(self.num_states):
                max_val = -np.inf
                for input_bit in range(2):
                    next_state = self.next_state[prev_state, input_bit]
                    val = beta[k + 1, next_state] + gamma[k, prev_state, input_bit]
                    max_val = max_star(max_val, val)
                beta[k, prev_state] = max_val
        
        # Compute LLRs
        llr_aposteriori = np.zeros(K)
        
        for k in range(K):
            llr_0 = -np.inf  # Log probability of bit = 0
            llr_1 = -np.inf  # Log probability of bit = 1
            
            for state in range(self.num_states):
                for input_bit in range(2):
                    next_state = self.next_state[state, input_bit]
                    val = alpha[k, state] + gamma[k, state, input_bit] + beta[k + 1, next_state]
                    
                    if input_bit == 0:
                        llr_0 = max_star(llr_0, val)
                    else:
                        llr_1 = max_star(llr_1, val)
            
            # LLR = log(P(bit=0)/P(bit=1))
            llr_aposteriori[k] = llr_0 - llr_1
        
        # Compute extrinsic information
        if return_extrinsic:
            # Extrinsic = A posteriori - A priori - Channel
            llr_extrinsic = llr_aposteriori - llr_apriori - llr_systematic
            llr_output = llr_extrinsic
        else:
            llr_output = llr_aposteriori
        
        # Hard decisions
        decoded_bits = (llr_aposteriori < 0).astype(np.uint8)
        
        return decoded_bits, llr_output
    
    def _compute_gamma(self,
                       llr_systematic: np.ndarray,
                       llr_parity: np.ndarray,
                       llr_apriori: np.ndarray) -> np.ndarray:
        """
        Compute branch metrics (gamma)
        
        gamma[k, state, input] = branch metric for transition at time k
        
        Parameters:
        -----------
        llr_systematic : np.ndarray
            Systematic LLRs (K bits)
        llr_parity : np.ndarray
            Parity LLRs (K bits)
        llr_apriori : np.ndarray
            A priori LLRs (K bits)
        
        Returns:
        --------
        np.ndarray : Branch metrics (K, num_states, 2)
        """
        K = len(llr_systematic)
        gamma = np.zeros((K, self.num_states, 2))
        
        for k in range(K):
            for state in range(self.num_states):
                for input_bit in range(2):
                    # Expected outputs for this state and input
                    sys_out = self.output_systematic[state, input_bit]
                    par_out = self.output_parity[state, input_bit]
                    
                    # Branch metric components
                    # For bit x with LLR L: metric = (-1)^x * L/2
                    
                    # Systematic contribution
                    if sys_out == 0:
                        sys_metric = llr_systematic[k] / 2.0
                    else:
                        sys_metric = -llr_systematic[k] / 2.0
                    
                    # Parity contribution
                    if par_out == 0:
                        par_metric = llr_parity[k] / 2.0
                    else:
                        par_metric = -llr_parity[k] / 2.0
                    
                    # A priori contribution
                    if input_bit == 0:
                        apr_metric = llr_apriori[k] / 2.0
                    else:
                        apr_metric = -llr_apriori[k] / 2.0
                    
                    gamma[k, state, input_bit] = sys_metric + par_metric + apr_metric
        
        return gamma


def turbo_decode(llr_encoded: np.ndarray,
                 K: int,
                 num_iterations: int = 5,
                 debug: bool = False) -> np.ndarray:
    """
    Turbo Decoder with iterative Log-MAP decoding
    
    Parameters:
    -----------
    llr_encoded : np.ndarray
        LLRs from rate de-matching (3K + 12 values)
        Format: [sys_0, par1_0, par2_0, sys_1, par1_1, par2_1, ..., tails]
    K : int
        Code block size (before encoding)
    num_iterations : int
        Number of iterations (default: 5, optimized for speed)
        Standard LTE uses 8, but 5 iterations gives similar performance
    debug : bool
        Enable debug output for problematic blocks
    
    Returns:
    --------
    np.ndarray : Decoded bits (K bits)
    """
    # Check input dimensions
    expected_len = 3*K + 12
    if len(llr_encoded) != expected_len:
        if debug:
            print(f"[turbo_decode WARNING] Expected {expected_len} LLRs, got {len(llr_encoded)}")
    
    # Parse input LLRs
    # Extract systematic, parity1, parity2 (only K bits each, ignore tails for now)
    llr_systematic = np.zeros(K)
    llr_parity1 = np.zeros(K)
    llr_parity2 = np.zeros(K)
    
    for k in range(K):
        llr_systematic[k] = llr_encoded[3*k]
        llr_parity1[k] = llr_encoded[3*k + 1]
        llr_parity2[k] = llr_encoded[3*k + 2]
    
    # Tail bits at positions 3K to 3K+11
    # Format (from encoder):
    # - systematic tail 1 (3 bits): 3K, 3K+1, 3K+2
    # - parity 1 tail (3 bits): 3K+3, 3K+4, 3K+5  
    # - systematic tail 2 (3 bits): 3K+6, 3K+7, 3K+8
    # - parity 2 tail (3 bits): 3K+9, 3K+10, 3K+11
    
    # For now, extend LLRs with tail bits for better decoding
    llr_systematic = np.concatenate([llr_systematic, llr_encoded[3*K:3*K+3]])
    llr_parity1 = np.concatenate([llr_parity1, llr_encoded[3*K+3:3*K+6]])
    
    llr_systematic_forDec2 = np.concatenate([llr_systematic[:K], llr_encoded[3*K+6:3*K+9]])
    llr_parity2 = np.concatenate([llr_parity2, llr_encoded[3*K+9:3*K+12]])
    
    K_extended = K + 3
    
    # Initialize decoders
    decoder1 = LogMAPDecoder()
    decoder2 = LogMAPDecoder()
    
    # Initialize extrinsic information (only for data bits, not tails)
    extrinsic_1to2 = np.zeros(K)
    extrinsic_2to1 = np.zeros(K)
    
    # Iterative decoding
    for iteration in range(num_iterations):
        # Decoder 1: decode with original sequence (K+3 bits)
        extrinsic_1to2_extended = np.concatenate([extrinsic_2to1, np.zeros(3)])
        decoded_bits_1, extrinsic_1to2_full = decoder1.decode(
            llr_systematic,
            llr_parity1,
            llr_apriori=extrinsic_1to2_extended,
            return_extrinsic=True
        )
        
        # Keep only data bits for interleaving
        extrinsic_1to2 = extrinsic_1to2_full[:K]
        
        if debug and iteration == 0:
            print(f"  [iter 0] extrinsic_1to2 stats: mean_abs={np.mean(np.abs(extrinsic_1to2)):.3f}, std={np.std(extrinsic_1to2):.3f}")
        
        # Interleave extrinsic info for decoder 2
        extrinsic_1to2_interleaved = qpp_interleave(extrinsic_1to2, K)
        
        # Interleave systematic LLRs for decoder 2
        llr_systematic_interleaved = qpp_interleave(llr_systematic[:K], K)
        llr_systematic_forDec2_full = np.concatenate([llr_systematic_interleaved, llr_encoded[3*K+6:3*K+9]])
        
        # Decoder 2: decode with interleaved sequence
        extrinsic_1to2_extended = np.concatenate([extrinsic_1to2_interleaved, np.zeros(3)])
        decoded_bits_2, extrinsic_2to1_interleaved_full = decoder2.decode(
            llr_systematic_forDec2_full,
            llr_parity2,
            llr_apriori=extrinsic_1to2_extended,
            return_extrinsic=True
        )
        
        # Keep only data bits and deinterleave
        extrinsic_2to1_interleaved = extrinsic_2to1_interleaved_full[:K]
        extrinsic_2to1 = qpp_deinterleave(extrinsic_2to1_interleaved, K)
    
    # Final decoding (a posteriori)
    extrinsic_1to2_extended = np.concatenate([extrinsic_2to1, np.zeros(3)])
    decoded_bits, llr_aposteriori = decoder1.decode(
        llr_systematic,
        llr_parity1,
        llr_apriori=extrinsic_1to2_extended,
        return_extrinsic=False
    )
    
    # Return only data bits (not tail bits)
    return decoded_bits[:K]


if __name__ == '__main__':
    """Self-test with perfect channel (no noise)"""
    print("=" * 70)
    print("Turbo Decoder Self-Test")
    print("=" * 70)
    
    try:
        from .turbo_encoder import turbo_encode
    except ImportError:
        from turbo_encoder import turbo_encode
    
    # Test with perfect channel (LLRs from hard decisions)
    print("\n[Test 1] Perfect Channel (No Noise)")
    
    K = 40
    input_bits = np.random.randint(0, 2, K, dtype=np.uint8)
    
    # Encode
    encoded_bits = turbo_encode(input_bits)
    
    # Convert to LLRs (perfect: +inf for 0, -inf for 1)
    # Use large values instead of inf for numerical stability
    LLR_SCALE = 10.0
    llr_encoded = np.where(encoded_bits == 0, LLR_SCALE, -LLR_SCALE)
    
    # Decode
    decoded_bits = turbo_decode(llr_encoded, K, num_iterations=8)
    
    # Check BER
    errors = np.sum(input_bits != decoded_bits)
    ber = errors / K
    
    print(f"  K = {K}")
    print(f"  Input:   {input_bits[:20]}...")
    print(f"  Decoded: {decoded_bits[:20]}...")
    print(f"  Errors:  {errors}/{K}")
    print(f"  BER:     {ber:.6f}")
    
    if ber == 0:
        print("  ✓ Perfect decoding (BER = 0)")
    else:
        print(f"  ⚠ Some errors remain (BER = {ber:.6f})")
    
    # Test with different sizes
    print("\n[Test 2] Different Block Sizes")
    for K_test in [40, 104, 512]:
        input_bits = np.random.randint(0, 2, K_test, dtype=np.uint8)
        encoded_bits = turbo_encode(input_bits)
        llr_encoded = np.where(encoded_bits == 0, LLR_SCALE, -LLR_SCALE)
        decoded_bits = turbo_decode(llr_encoded, K_test, num_iterations=4)
        
        errors = np.sum(input_bits != decoded_bits)
        ber = errors / K_test
        
        print(f"  K={K_test:4d}: BER={ber:.6f} ({errors} errors)")
    
    print("\n" + "=" * 70)
    print("Turbo Decoder basic tests completed!")
    print("=" * 70)
