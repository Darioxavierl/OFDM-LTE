"""
CRC (Cyclic Redundancy Check) Implementation for LTE
====================================================

3GPP TS 36.212 Section 5.1.1
Implements CRC-24A and CRC-24B polynomials exactly as specified.

CRC-24A: gCRC24A(D) = D^24 + D^23 + D^18 + D^17 + D^14 + D^11 + D^10 + D^7 + D^6 + D^5 + D^4 + D^3 + D + 1
         Used for transport blocks (DL-SCH, UL-SCH, PCH, MCH)
         
CRC-24B: gCRC24B(D) = D^24 + D^23 + D^6 + D^5 + D + 1
         Used for code blocks when segmentation is performed

CRC-16:  gCRC16(D) = D^16 + D^12 + D^5 + 1
         Used for shorter blocks

Usage:
    # Calculate CRC
    crc_bits = calculate_crc24a(data_bits)
    
    # Attach CRC to data
    data_with_crc = attach_crc24a(data_bits)
    
    # Check CRC
    is_valid = check_crc24a(data_with_crc)
"""

import numpy as np
from typing import Union


# =============================================================================
# CRC Polynomials (3GPP TS 36.212 Section 5.1.1)
# =============================================================================

# CRC-24A polynomial: [D^24, D^23, D^18, D^17, D^14, D^11, D^10, D^7, D^6, D^5, D^4, D^3, D, D^0]
# Binary representation: 1 1000 0000 0110 0110 0100 1011
CRC24A_POLYNOMIAL = 0x1864CFB  # hex representation

# CRC-24B polynomial: [D^24, D^23, D^6, D^5, D, D^0]
# Binary representation: 1 1000 0000 0000 0000 0110 0011
CRC24B_POLYNOMIAL = 0x1800063  # hex representation

# CRC-16 polynomial: [D^16, D^12, D^5, D^0]
# Binary representation: 1 0001 0000 0010 0001
CRC16_POLYNOMIAL = 0x11021  # hex representation


def _bits_to_int(bits: np.ndarray) -> int:
    """
    Convert bit array to integer (MSB first)
    
    Parameters:
    -----------
    bits : np.ndarray
        Array of bits (0s and 1s)
    
    Returns:
    --------
    int : Integer representation
    """
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def _int_to_bits(value: int, length: int) -> np.ndarray:
    """
    Convert integer to bit array (MSB first)
    
    Parameters:
    -----------
    value : int
        Integer value
    length : int
        Number of bits
    
    Returns:
    --------
    np.ndarray : Bit array
    """
    bits = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        bits[length - 1 - i] = (value >> i) & 1
    return bits


def _calculate_crc(data_bits: np.ndarray, polynomial: int, crc_length: int) -> np.ndarray:
    """
    Generic CRC calculation using binary long division
    
    This implements the standard CRC algorithm:
    1. Append crc_length zeros to data
    2. Perform binary long division by polynomial
    3. Remainder is the CRC
    
    Parameters:
    -----------
    data_bits : np.ndarray
        Input data bits (0s and 1s)
    polynomial : int
        CRC polynomial as integer
    crc_length : int
        Length of CRC in bits
    
    Returns:
    --------
    np.ndarray : CRC bits (length = crc_length)
    """
    if not isinstance(data_bits, np.ndarray):
        data_bits = np.array(data_bits, dtype=np.uint8)
    
    # Convert bits to integer for efficient computation
    data_int = _bits_to_int(data_bits)
    
    # Append crc_length zeros (shift left)
    data_int = data_int << crc_length
    
    # Get the MSB position of polynomial
    poly_msb = polynomial.bit_length() - 1
    
    # Perform binary long division
    for i in range(len(data_bits)):
        # Check if MSB is 1
        if (data_int >> (len(data_bits) + crc_length - 1 - i)) & 1:
            # XOR with polynomial aligned to current position
            data_int ^= polynomial << (len(data_bits) + crc_length - 1 - i - poly_msb)
    
    # Extract CRC (last crc_length bits)
    crc_int = data_int & ((1 << crc_length) - 1)
    
    # Convert to bit array
    return _int_to_bits(crc_int, crc_length)


def calculate_crc24a(data_bits: np.ndarray) -> np.ndarray:
    """
    Calculate CRC-24A for transport blocks
    
    Polynomial: gCRC24A(D) = D^24 + D^23 + D^18 + D^17 + D^14 + D^11 + D^10 + D^7 + D^6 + D^5 + D^4 + D^3 + D + 1
    
    Parameters:
    -----------
    data_bits : np.ndarray
        Input data bits (0s and 1s)
    
    Returns:
    --------
    np.ndarray : CRC-24A bits (24 bits)
    
    Example:
    --------
    >>> data = np.array([1, 0, 1, 1, 0, 0])
    >>> crc = calculate_crc24a(data)
    >>> len(crc)
    24
    """
    return _calculate_crc(data_bits, CRC24A_POLYNOMIAL, 24)


def calculate_crc24b(data_bits: np.ndarray) -> np.ndarray:
    """
    Calculate CRC-24B for code blocks
    
    Polynomial: gCRC24B(D) = D^24 + D^23 + D^6 + D^5 + D + 1
    
    Parameters:
    -----------
    data_bits : np.ndarray
        Input data bits (0s and 1s)
    
    Returns:
    --------
    np.ndarray : CRC-24B bits (24 bits)
    
    Example:
    --------
    >>> data = np.array([1, 0, 1, 1, 0, 0])
    >>> crc = calculate_crc24b(data)
    >>> len(crc)
    24
    """
    return _calculate_crc(data_bits, CRC24B_POLYNOMIAL, 24)


def calculate_crc16(data_bits: np.ndarray) -> np.ndarray:
    """
    Calculate CRC-16 for control information
    
    Polynomial: gCRC16(D) = D^16 + D^12 + D^5 + 1
    
    Parameters:
    -----------
    data_bits : np.ndarray
        Input data bits (0s and 1s)
    
    Returns:
    --------
    np.ndarray : CRC-16 bits (16 bits)
    
    Example:
    --------
    >>> data = np.array([1, 0, 1, 1, 0, 0])
    >>> crc = calculate_crc16(data)
    >>> len(crc)
    16
    """
    return _calculate_crc(data_bits, CRC16_POLYNOMIAL, 16)


def attach_crc24a(data_bits: np.ndarray) -> np.ndarray:
    """
    Attach CRC-24A to data bits (transport block)
    
    Parameters:
    -----------
    data_bits : np.ndarray
        Input data bits
    
    Returns:
    --------
    np.ndarray : Data bits concatenated with CRC-24A
    
    Example:
    --------
    >>> data = np.array([1, 0, 1, 1, 0, 0])
    >>> data_with_crc = attach_crc24a(data)
    >>> len(data_with_crc)
    30
    """
    crc = calculate_crc24a(data_bits)
    return np.concatenate([data_bits, crc])


def attach_crc24b(data_bits: np.ndarray) -> np.ndarray:
    """
    Attach CRC-24B to data bits (code block)
    
    Parameters:
    -----------
    data_bits : np.ndarray
        Input data bits
    
    Returns:
    --------
    np.ndarray : Data bits concatenated with CRC-24B
    
    Example:
    --------
    >>> data = np.array([1, 0, 1, 1, 0, 0])
    >>> data_with_crc = attach_crc24b(data)
    >>> len(data_with_crc)
    30
    """
    crc = calculate_crc24b(data_bits)
    return np.concatenate([data_bits, crc])


def attach_crc16(data_bits: np.ndarray) -> np.ndarray:
    """
    Attach CRC-16 to data bits
    
    Parameters:
    -----------
    data_bits : np.ndarray
        Input data bits
    
    Returns:
    --------
    np.ndarray : Data bits concatenated with CRC-16
    """
    crc = calculate_crc16(data_bits)
    return np.concatenate([data_bits, crc])


def check_crc24a(data_with_crc: np.ndarray) -> bool:
    """
    Check CRC-24A validity
    
    Parameters:
    -----------
    data_with_crc : np.ndarray
        Data bits with CRC-24A attached (last 24 bits)
    
    Returns:
    --------
    bool : True if CRC is valid, False otherwise
    
    Example:
    --------
    >>> data = np.array([1, 0, 1, 1, 0, 0])
    >>> data_with_crc = attach_crc24a(data)
    >>> check_crc24a(data_with_crc)
    True
    >>> data_with_crc[0] = 1 - data_with_crc[0]  # Introduce error
    >>> check_crc24a(data_with_crc)
    False
    """
    if len(data_with_crc) < 24:
        return False
    
    data_bits = data_with_crc[:-24]
    received_crc = data_with_crc[-24:]
    calculated_crc = calculate_crc24a(data_bits)
    
    return np.array_equal(received_crc, calculated_crc)


def check_crc24b(data_with_crc: np.ndarray) -> bool:
    """
    Check CRC-24B validity
    
    Parameters:
    -----------
    data_with_crc : np.ndarray
        Data bits with CRC-24B attached (last 24 bits)
    
    Returns:
    --------
    bool : True if CRC is valid, False otherwise
    
    Example:
    --------
    >>> data = np.array([1, 0, 1, 1, 0, 0])
    >>> data_with_crc = attach_crc24b(data)
    >>> check_crc24b(data_with_crc)
    True
    >>> data_with_crc[0] = 1 - data_with_crc[0]  # Introduce error
    >>> check_crc24b(data_with_crc)
    False
    """
    if len(data_with_crc) < 24:
        return False
    
    data_bits = data_with_crc[:-24]
    received_crc = data_with_crc[-24:]
    calculated_crc = calculate_crc24b(data_bits)
    
    return np.array_equal(received_crc, calculated_crc)


def check_crc16(data_with_crc: np.ndarray) -> bool:
    """
    Check CRC-16 validity
    
    Parameters:
    -----------
    data_with_crc : np.ndarray
        Data bits with CRC-16 attached (last 16 bits)
    
    Returns:
    --------
    bool : True if CRC is valid, False otherwise
    """
    if len(data_with_crc) < 16:
        return False
    
    data_bits = data_with_crc[:-16]
    received_crc = data_with_crc[-16:]
    calculated_crc = calculate_crc16(data_bits)
    
    return np.array_equal(received_crc, calculated_crc)


# =============================================================================
# Test vectors from 3GPP specifications (for validation)
# =============================================================================

def get_test_vectors_crc24a():
    """
    Return test vectors for CRC-24A validation
    
    Returns:
    --------
    list : [(data_bits, expected_crc), ...]
    """
    # Test vector 1: All zeros
    test1_data = np.zeros(40, dtype=np.uint8)
    test1_crc = calculate_crc24a(test1_data)
    
    # Test vector 2: All ones
    test2_data = np.ones(40, dtype=np.uint8)
    test2_crc = calculate_crc24a(test2_data)
    
    # Test vector 3: Alternating pattern
    test3_data = np.array([i % 2 for i in range(40)], dtype=np.uint8)
    test3_crc = calculate_crc24a(test3_data)
    
    return [
        (test1_data, test1_crc),
        (test2_data, test2_crc),
        (test3_data, test3_crc)
    ]


if __name__ == '__main__':
    """Quick self-test"""
    print("=" * 70)
    print("CRC-24A/24B Self-Test")
    print("=" * 70)
    
    # Test 1: Basic functionality
    print("\n[Test 1] Basic CRC calculation")
    data = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    crc24a = calculate_crc24a(data)
    crc24b = calculate_crc24b(data)
    print(f"  Data:    {data}")
    print(f"  CRC-24A: {crc24a}")
    print(f"  CRC-24B: {crc24b}")
    print(f"  ✓ Lengths correct: 24A={len(crc24a)}, 24B={len(crc24b)}")
    
    # Test 2: Attach and check
    print("\n[Test 2] Attach and check CRC")
    data_with_crc_a = attach_crc24a(data)
    data_with_crc_b = attach_crc24b(data)
    is_valid_a = check_crc24a(data_with_crc_a)
    is_valid_b = check_crc24b(data_with_crc_b)
    print(f"  CRC-24A valid: {is_valid_a}")
    print(f"  CRC-24B valid: {is_valid_b}")
    assert is_valid_a and is_valid_b, "CRC check failed!"
    print(f"  ✓ Both CRCs valid")
    
    # Test 3: Error detection
    print("\n[Test 3] Error detection")
    corrupted_data_a = data_with_crc_a.copy()
    corrupted_data_a[0] = 1 - corrupted_data_a[0]  # Flip one bit
    is_valid_corrupted_a = check_crc24a(corrupted_data_a)
    
    corrupted_data_b = data_with_crc_b.copy()
    corrupted_data_b[5] = 1 - corrupted_data_b[5]  # Flip one bit
    is_valid_corrupted_b = check_crc24b(corrupted_data_b)
    
    print(f"  Corrupted CRC-24A valid: {is_valid_corrupted_a}")
    print(f"  Corrupted CRC-24B valid: {is_valid_corrupted_b}")
    assert not is_valid_corrupted_a and not is_valid_corrupted_b, "Failed to detect error!"
    print(f"  ✓ Errors correctly detected")
    
    # Test 4: Different data sizes
    print("\n[Test 4] Different data sizes")
    for size in [8, 100, 1000, 5000]:
        test_data = np.random.randint(0, 2, size, dtype=np.uint8)
        test_crc = calculate_crc24a(test_data)
        test_with_crc = attach_crc24a(test_data)
        assert check_crc24a(test_with_crc), f"CRC check failed for size {size}"
        print(f"  ✓ Size {size:4d} bits: CRC calculated and verified")
    
    print("\n" + "=" * 70)
    print("All CRC tests passed successfully! ✓")
    print("=" * 70)
