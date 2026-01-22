"""
LTE Channel Coding Module - 3GPP TS 36.212 Compliant
=====================================================

Implements complete LTE channel coding chain:
- CRC attachment (CRC-24A, CRC-24B)
- Code block segmentation
- Turbo encoding (rate 1/3)
- Rate matching (circular buffer)
- Turbo decoding (MAP/Log-MAP)

All implementations follow 3GPP TS 36.212 specifications exactly.
"""

from .crc import calculate_crc24a, calculate_crc24b, attach_crc24a, attach_crc24b, check_crc24a, check_crc24b
from .segmentation import segment_code_blocks, desegment_code_blocks, get_segmentation_info
from .turbo_encoder import turbo_encode, qpp_interleave, qpp_deinterleave
from .turbo_decoder import turbo_decode, LogMAPDecoder
from .rate_matching import rate_match_turbo, rate_dematching_turbo, sub_block_interleaver, sub_block_deinterleaver

__all__ = [
    'calculate_crc24a',
    'calculate_crc24b', 
    'attach_crc24a',
    'attach_crc24b',
    'check_crc24a',
    'check_crc24b',
    'segment_code_blocks',
    'desegment_code_blocks',
    'get_segmentation_info',
    'turbo_encode',
    'turbo_decode',
    'LogMAPDecoder',
    'qpp_interleave',
    'qpp_deinterleave',
    'rate_match_turbo',
    'rate_dematching_turbo',
    'sub_block_interleaver',
    'sub_block_deinterleaver'
]
