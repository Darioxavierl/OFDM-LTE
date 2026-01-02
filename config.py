"""
LTE Parameters and Configuration Module
Self-contained configuration for OFDM-SC library
"""
import numpy as np

# ============================================================================
# Predefined LTE Profiles
# ============================================================================

LTE_PROFILES = {
    1.25: {'Nc': 76, 'N': 128},
    2.5: {'Nc': 150, 'N': 256},
    5.0: {'Nc': 300, 'N': 512},
    10.0: {'Nc': 600, 'N': 1024},
    15.0: {'Nc': 900, 'N': 2048},
    20.0: {'Nc': 1200, 'N': 2048}
}

# Cyclic Prefix values (in microseconds)
CP_VALUES = {
    'normal': 4.7,
    'extended_15khz': 16.6,
    'extended_7.5khz': 33.0
}

# Available modulation schemes
MODULATION_SCHEMES = ['QPSK', '16-QAM', '64-QAM']

# Subcarrier spacing (in kHz)
SUBCARRIER_SPACING = [15.0, 7.5]

# ITU-R M.1225 Channel Models for Rayleigh Multipath
ITU_CHANNEL_MODELS = {
    'Pedestrian_A': {
        'delays_us': [0.0, 0.11, 0.19, 0.41],
        'power_db': [0.0, -9.7, -19.2, -22.8],
        'description': 'Pedestrian, low velocity, short distance'
    },
    'Pedestrian_B': {
        'delays_us': [0.0, 0.2, 0.8, 1.2, 2.3, 3.7],
        'power_db': [0.0, -0.9, -4.9, -8.0, -7.8, -23.9],
        'description': 'Pedestrian, high velocity'
    },
    'Vehicular_A': {
        'delays_us': [0.0, 0.31, 0.71, 1.09, 1.73, 2.51],
        'power_db': [0.0, -1.0, -9.0, -10.0, -15.0, -20.0],
        'description': 'Vehicular, low velocity, short distance'
    },
    'Vehicular_B': {
        'delays_us': [0.0, 0.3, 0.7, 1.09, 1.73, 2.51, 3.7, 4.53],
        'power_db': [0.0, -1.0, -9.0, -10.0, -13.0, -16.0, -21.6, -24.0],
        'description': 'Vehicular, high velocity, long distance'
    },
    'Bad_Urban': {
        'delays_us': [0.0, 0.1, 0.3, 0.5, 0.9, 1.3, 1.9, 2.6],
        'power_db': [0.0, -3.0, -5.0, -7.0, -9.0, -11.0, -13.0, -15.0],
        'description': 'Urban with severe multipath'
    }
}


class LTEConfig:
    """
    LTE OFDM Configuration Manager
    
    Handles all LTE-specific parameters and derives secondary parameters
    from primary configuration inputs.
    
    Example:
        config = LTEConfig(bandwidth=5.0, delta_f=15.0, modulation='QPSK')
        print(config)  # Display configuration
        print(config.get_info())  # Get info dict
    """
    
    def __init__(self, bandwidth=5.0, delta_f=15.0, modulation='QPSK', cp_type='normal'):
        """
        Initialize LTE Configuration
        
        Args:
            bandwidth (float): Bandwidth in MHz. Typical: 1.25, 2.5, 5, 10, 15, 20
            delta_f (float): Subcarrier spacing in kHz. Typical: 15.0 or 7.5
            modulation (str): Modulation scheme. Options: 'QPSK', '16-QAM', '64-QAM'
            cp_type (str): Cyclic prefix type. Options: 'normal', 'extended'
        
        Raises:
            ValueError: If invalid modulation scheme provided
        """
        self.bandwidth = bandwidth
        self.delta_f = delta_f  # kHz
        self.modulation = modulation
        self.cp_type = cp_type
        
        # Validate inputs
        if modulation not in MODULATION_SCHEMES:
            raise ValueError(f"Unsupported modulation: {modulation}. Options: {MODULATION_SCHEMES}")
        
        # Calculate derived parameters
        self._calculate_parameters()
    
    def _calculate_parameters(self):
        """Calculate all derived parameters based on primary configuration"""
        # Get LTE profile (number of useful subcarriers and FFT size)
        if self.bandwidth in LTE_PROFILES:
            profile = LTE_PROFILES[self.bandwidth]
            self.Nc = profile['Nc']      # Number of useful subcarriers
            self.N = profile['N']        # FFT size
        else:
            # Calculate manually if no predefined profile
            self.Nc = int((self.bandwidth * 1e3) / self.delta_f)
            self.N = self._next_power_of_2(self.Nc)
        
        # Sampling frequency (Hz)
        self.fs = self.N * self.delta_f * 1e3  # kHz to Hz
        
        # Sampling period (seconds)
        self.Ts = 1 / self.fs
        
        # Duration of OFDM symbol without CP (seconds)
        self.T_symbol = self.N * self.Ts
        
        # Cyclic prefix duration and length
        self.cp_duration = self._get_cp_duration()  # microseconds
        self.cp_length = int(self.cp_duration * 1e-6 * self.fs)  # samples
        
        # Bits per symbol based on modulation
        self.bits_per_symbol = self._get_bits_per_symbol()
        
        # Total samples per OFDM symbol (including CP)
        self.samples_per_ofdm_symbol = self.N + self.cp_length
    
    def _next_power_of_2(self, x):
        """Calculate next power of 2"""
        return int(2**np.ceil(np.log2(x)))
    
    def _get_cp_duration(self):
        """Get cyclic prefix duration based on configuration"""
        if self.cp_type == 'normal':
            return CP_VALUES['normal']
        elif self.cp_type == 'extended':
            if self.delta_f == 15.0:
                return CP_VALUES['extended_15khz']
            else:  # 7.5 kHz
                return CP_VALUES['extended_7.5khz']
        return CP_VALUES['normal']
    
    def _get_bits_per_symbol(self):
        """Get bits per symbol for current modulation scheme"""
        bits_map = {
            'QPSK': 2,
            '16-QAM': 4,
            '64-QAM': 6
        }
        return bits_map.get(self.modulation, 2)
    
    def get_info(self):
        """
        Get comprehensive configuration information
        
        Returns:
            dict: Dictionary with all configuration parameters
        """
        return {
            'Bandwidth (MHz)': self.bandwidth,
            'Subcarrier Spacing (kHz)': self.delta_f,
            'Modulation': self.modulation,
            'CP Type': self.cp_type,
            'Useful Subcarriers (Nc)': self.Nc,
            'FFT Points (N)': self.N,
            'Sampling Frequency (MHz)': self.fs / 1e6,
            'Sampling Period (ns)': self.Ts * 1e9,
            'OFDM Symbol Duration (μs)': self.T_symbol * 1e6,
            'CP Duration (μs)': self.cp_duration,
            'CP Length (samples)': self.cp_length,
            'Bits per Symbol': self.bits_per_symbol,
            'Samples per OFDM Symbol': self.samples_per_ofdm_symbol
        }
    
    def __str__(self):
        """String representation of configuration"""
        info = self.get_info()
        lines = ["LTE OFDM Configuration:"]
        lines.extend([f"  {k}: {v}" for k, v in info.items()])
        return "\n".join(lines)
    
    def __repr__(self):
        """Representation string"""
        return (f"LTEConfig(bandwidth={self.bandwidth}, delta_f={self.delta_f}, "
                f"modulation='{self.modulation}', cp_type='{self.cp_type}')")
    
    def copy(self):
        """Create a copy of this configuration"""
        return LTEConfig(
            bandwidth=self.bandwidth,
            delta_f=self.delta_f,
            modulation=self.modulation,
            cp_type=self.cp_type
        )


# Utility functions for common configurations

def create_config_5MHz_QPSK():
    """Create 5 MHz QPSK configuration (most common)"""
    return LTEConfig(bandwidth=5.0, delta_f=15.0, modulation='QPSK', cp_type='normal')


def create_config_20MHz_16QAM():
    """Create 20 MHz 16-QAM configuration"""
    return LTEConfig(bandwidth=20.0, delta_f=15.0, modulation='16-QAM', cp_type='normal')


def create_config_10MHz_64QAM():
    """Create 10 MHz 64-QAM configuration"""
    return LTEConfig(bandwidth=10.0, delta_f=15.0, modulation='64-QAM', cp_type='normal')
