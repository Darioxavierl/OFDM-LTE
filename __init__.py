"""
OFDM-SC: LTE-based OFDM and Single-Carrier FDM Library

A comprehensive Python library for simulating OFDM and SC-FDM systems
based on LTE specifications. Includes channel simulation, PAPR analysis,
and signal metrics calculation.

Main Components:
    - config: LTE configuration and parameters
    - modulator: QAM and OFDM/SC-FDM modulation
    - channel: AWGN and Rayleigh multipath channels
    - demodulator: OFDM/SC-FDM demodulation and equalization
    - signal_analysis: BER, PAPR, and metrics computation
    - ofdm_module: Main API for end-to-end simulation

Quick Start:
    from module import OFDMModule
    
    # Create module with default 5MHz QPSK configuration
    ofdm = OFDMModule()
    
    # Transmit 1000 bits
    bits = np.random.randint(0, 2, 1000)
    results = ofdm.transmit(bits, snr_db=15)
    
    # Access results
    ber = results['ber']
    papr = results['papr_db']

Author: OFDM-SC Project
Version: 1.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "OFDM-SC Project"
__all__ = [
    'config',
    'OFDMModule',
    'LTEConfig',
    'MODULATION_SCHEMES',
    'ITU_CHANNEL_MODELS'
]

from .config import (
    LTEConfig,
    MODULATION_SCHEMES,
    ITU_CHANNEL_MODELS,
    LTE_PROFILES,
    CP_VALUES,
    SUBCARRIER_SPACING
)

# Lazy imports to reduce initial load time
_ofdm_module = None
_signal_analysis = None


def get_ofdm_module():
    """Get or lazily load OFDMModule"""
    global _ofdm_module
    if _ofdm_module is None:
        from .ofdm_module import OFDMModule as OM
        _ofdm_module = OM
    return _ofdm_module


def get_signal_analysis():
    """Get or lazily load signal analysis utilities"""
    global _signal_analysis
    if _signal_analysis is None:
        from . import signal_analysis as sa
        _signal_analysis = sa
    return _signal_analysis


# Make OFDMModule directly accessible
def OFDMModule(*args, **kwargs):
    """
    Create an OFDM/SC-FDM module instance
    
    Args:
        config: LTEConfig object (optional, default: 5MHz QPSK)
        channel_type: 'awgn' or 'rayleigh_mp' (default: 'awgn')
        itu_profile: ITU profile name for Rayleigh (default: 'Vehicular_A')
        enable_sc_fdm: Enable SC-FDM mode (default: False for OFDM)
    
    Returns:
        OFDMModule instance
    
    Example:
        module = OFDMModule(enable_sc_fdm=True)
    """
    OM = get_ofdm_module()
    return OM(*args, **kwargs)


if __name__ == '__main__':
    print(f"OFDM-SC Module v{__version__}")
    print("Available modulations:", MODULATION_SCHEMES)
    print("ITU profiles:", list(ITU_CHANNEL_MODELS.keys()))
