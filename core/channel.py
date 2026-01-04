"""
Canal AWGN y Rayleigh Multitrayecto - Simulación de canales de comunicación
"""
import numpy as np
import os
from core.rayleighchannel import RayleighChannel
from config import ITU_CHANNEL_MODELS


class AWGNChannel:
    """Canal AWGN que añade ruido gaussiano a la señal"""
    
    def __init__(self, snr_db=10.0):
        """
        Inicializa el canal AWGN
        
        Args:
            snr_db: Relación Señal-Ruido en dB
        """
        self.snr_db = snr_db
        self.snr_linear = 10 ** (snr_db / 10)
        self.noise_power = None
    
    def set_snr(self, snr_db):
        """
        Establece el SNR del canal
        
        Args:
            snr_db: Relación Señal-Ruido en dB
        """
        self.snr_db = snr_db
        self.snr_linear = 10 ** (snr_db / 10)
    
    def transmit(self, signal):
        """
        Transmite una señal a través del canal AWGN
        
        Args:
            signal: Señal a transmitir (puede ser real o compleja)
            
        Returns:
            tuple: (signal_received, noise_added)
        """
        # Calcular potencia de señal
        if np.iscomplexobj(signal):
            signal_power = np.mean(np.abs(signal) ** 2)
        else:
            signal_power = np.mean(signal ** 2)
        
        # Calcular potencia de ruido necesaria
        # SNR = signal_power / noise_power
        noise_power = signal_power / self.snr_linear
        self.noise_power = noise_power
        
        # Generar ruido blanco gaussiano complejo
        if np.iscomplexobj(signal):
            # Para señales complejas: ruido en componente I y Q
            noise_real = np.random.normal(0, np.sqrt(noise_power / 2), len(signal))
            noise_imag = np.random.normal(0, np.sqrt(noise_power / 2), len(signal))
            noise = noise_real + 1j * noise_imag
        else:
            # Para señales reales
            noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        
        # Señal recibida = Señal transmitida + Ruido
        signal_received = signal + noise
        
        return signal_received, noise
    
    def get_noise_power(self):
        """Retorna la potencia de ruido del último símbolo transmitido"""
        return self.noise_power
    
    def get_snr_info(self):
        """Retorna información del SNR actual"""
        return {
            'SNR (dB)': self.snr_db,
            'SNR (lineal)': self.snr_linear,
            'Potencia de ruido': self.noise_power
        }


class RayleighMultiPathChannel:
    """
    Canal Rayleigh Multitrayecto basado en perfiles ITU-R M.1225
    Simula múltiples caminos con retardos y ganancias realistas
    """
    
    def __init__(self, snr_db=10.0, fs=None, itu_profile='Vehicular_A', fD=None,
                 frequency_ghz=None, velocity_kmh=None, verbose=True):
        """
        Inicializa el canal Rayleigh multitrayecto
        
        Args:
            snr_db: Relación Señal-Ruido en dB
            fs: Frecuencia de muestreo (Hz)
            itu_profile: Nombre del perfil ITU-R M.1225
            fD: Frecuencia Doppler máxima (Hz). Si es None, se calcula automáticamente
            frequency_ghz: Frecuencia portadora en GHz (opcional, usa esta si se proporciona)
            velocity_kmh: Velocidad en km/h (opcional, usa esta si se proporciona)
            verbose: Si True, imprime información del canal (default True)
        """
        self.snr_db = snr_db
        self.snr_linear = 10 ** (snr_db / 10)
        self.itu_profile = itu_profile
        self.fs = fs
        self.noise_power = None
        self.frequency_ghz = frequency_ghz
        self.velocity_kmh = velocity_kmh
        self.verbose = verbose
        
        # Obtener retardos y ganancias del perfil desde config
        delays, gains = self._get_itu_profile_params(itu_profile)
        
        # Si no se especifica fD, usar valor típico para el perfil
        if fD is None:
            # Si se proporcionan frecuencia y velocidad, usarlas
            if frequency_ghz is not None and velocity_kmh is not None:
                fc = frequency_ghz * 1e9
                v = velocity_kmh / 3.6  # Convertir a m/s
            else:
                # Usar velocidad típica del perfil (de la descripción)
                profile_info = ITU_CHANNEL_MODELS.get(itu_profile, {})
                description = profile_info.get('description', '')
                
                # Velocidad predeterminada basada en el perfil
                if 'Pedestrian' in itu_profile:
                    v_max_kmh = 5.0  # Peatonal: ~5 km/h
                elif 'Vehicular_A' in itu_profile:
                    v_max_kmh = 30.0  # Vehicular A: ~30 km/h
                elif 'Vehicular_B' in itu_profile:
                    v_max_kmh = 120.0  # Vehicular B: ~120 km/h
                else:
                    v_max_kmh = 10.0  # Default
                
                v = v_max_kmh / 3.6
                
                # Usar frecuencia portadora 2 GHz por defecto
                fc = 2e9
            
            # Calcular Doppler
            c = 3e8
            fD = (v * fc) / c
        
        # Crear canal Rayleigh
        self.rayleigh = RayleighChannel(fs, fD, delays, gains)
        
        # Imprimir información solo si verbose=True
        if self.verbose:
            print(f"[RayleighMultiPathChannel] Perfil: {itu_profile}")
            print(f"  - Retardos: {[f'{d*1e6:.2f}µs' for d in delays]}")
            print(f"  - Ganancias: {[f'{g:.1f}dB' for g in gains]}")
            print(f"  - Doppler máximo: {fD:.1f} Hz")
            print(f"  - SNR: {snr_db} dB")
            if frequency_ghz is not None:
                print(f"  - Frecuencia: {frequency_ghz:.2f} GHz")
            if velocity_kmh is not None:
                print(f"  - Velocidad: {velocity_kmh:.1f} km/h")

    
    def _get_itu_profile_params(self, profile_name):
        """
        Obtiene los parámetros de retardo y ganancia desde config.ITU_CHANNEL_MODELS
        
        Args:
            profile_name: Nombre del perfil (ej: 'Pedestrian_A')
            
        Returns:
            tuple: (delays_segundos, gains_lineales)
        """
        if profile_name not in ITU_CHANNEL_MODELS:
            raise ValueError(f"Perfil ITU no encontrado: {profile_name}. "
                           f"Opciones disponibles: {list(ITU_CHANNEL_MODELS.keys())}")
        
        profile_data = ITU_CHANNEL_MODELS[profile_name]
        
        # Convertir delays de microsegundos a segundos
        delays_us = np.array(profile_data['delays_us'])
        delays_s = delays_us * 1e-6
        
        # Convertir ganancias de dB a lineal
        power_db = np.array(profile_data['power_db'])
        gains_linear = 10 ** (power_db / 20)  # Convertir de dB a lineal
        
        return delays_s, gains_linear
    
    def set_snr(self, snr_db):
        """Establece el SNR del canal"""
        self.snr_db = snr_db
        self.snr_linear = 10 ** (snr_db / 10)


    
    def set_profile(self, itu_profile):
        """Cambia el perfil ITU en tiempo real"""
        self.itu_profile = itu_profile
        delays, gains = self._get_itu_profile_params(itu_profile)
        self.rayleigh.delays = np.array(delays)
        self.rayleigh.gains = gains
        self.rayleigh.num_paths = len(delays)
    
    def transmit(self, signal):
        """
        Transmite a través del canal Rayleigh multitrayecto + AWGN
        
        Args:
            signal: Señal a transmitir (compleja)
            
        Returns:
            tuple: (signal_received, channel_response_info)
        """
        # Aplicar canal Rayleigh (múltiples caminos + Jakes fading)
        signal_rayleigh = self.rayleigh.filter(signal)
        
        # Calcular potencia de señal después de Rayleigh
        if np.iscomplexobj(signal_rayleigh):
            signal_power = np.mean(np.abs(signal_rayleigh) ** 2)
        else:
            signal_power = np.mean(signal_rayleigh ** 2)
        
        # Calcular potencia de ruido necesaria para el SNR deseado
        noise_power = signal_power / self.snr_linear
        self.noise_power = noise_power
        
        # Generar ruido blanco gaussiano complejo
        noise_real = np.random.normal(0, np.sqrt(noise_power / 2), len(signal_rayleigh))
        noise_imag = np.random.normal(0, np.sqrt(noise_power / 2), len(signal_rayleigh))
        noise = noise_real + 1j * noise_imag
        
        # Señal recibida = Rayleigh(señal) + Ruido
        signal_received = signal_rayleigh + noise
        
        return signal_received, None
    
    def get_channel_info(self):
        """Retorna información del canal"""
        return {
            'type': 'Rayleigh MultiPath (ITU-R M.1225)',
            'profile': self.itu_profile,
            'SNR_dB': self.snr_db,
            'num_paths': self.rayleigh.num_paths,
            'delays_us': self.rayleigh.delays * 1e6,
            'gains_dB': 20 * np.log10(self.rayleigh.gains)
        }


class FadingChannel:
    """Canal con desvanecimiento (extensión futura)"""
    
    def __init__(self, snr_db=10.0, fading_type='rayleigh'):
        """
        Inicializa el canal con desvanecimiento
        
        Args:
            snr_db: Relación Señal-Ruido en dB
            fading_type: Tipo de desvanecimiento ('rayleigh', 'rician', etc.)
        """
        self.snr_db = snr_db
        self.snr_linear = 10 ** (snr_db / 10)
        self.fading_type = fading_type
        self.awgn_channel = AWGNChannel(snr_db)
    
    def transmit(self, signal):
        """
        Transmite a través del canal con desvanecimiento
        
        Args:
            signal: Señal a transmitir
            
        Returns:
            tuple: (signal_received, channel_response)
        """
        # Generar respuesta del canal según tipo
        if self.fading_type == 'rayleigh':
            h_real = np.random.normal(0, 1/np.sqrt(2), len(signal))
            h_imag = np.random.normal(0, 1/np.sqrt(2), len(signal))
            channel_response = h_real + 1j * h_imag
        else:
            # Por defecto, usar desvanecimiento de Rayleigh
            h_real = np.random.normal(0, 1/np.sqrt(2), len(signal))
            h_imag = np.random.normal(0, 1/np.sqrt(2), len(signal))
            channel_response = h_real + 1j * h_imag
        
        # Aplicar desvanecimiento
        signal_faded = signal * channel_response
        
        # Añadir ruido AWGN
        signal_received, _ = self.awgn_channel.transmit(signal_faded)
        
        return signal_received, channel_response


class ChannelSimulator:
    """Simulador de canal que gestiona múltiples canales"""
    
    def __init__(self, channel_type='awgn', snr_db=10.0, fs=None, itu_profile='Vehicular_A',
                 frequency_ghz=None, velocity_kmh=None, verbose=True):
        """
        Inicializa el simulador de canal
        
        Args:
            channel_type: 'awgn', 'fading', o 'rayleigh_mp'
            snr_db: SNR en dB
            fs: Frecuencia de muestreo (necesario para Rayleigh)
            itu_profile: Perfil ITU para Rayleigh multitrayecto
            frequency_ghz: Frecuencia portadora en GHz (para Rayleigh)
            velocity_kmh: Velocidad en km/h (para Rayleigh)
            verbose: Si True, imprime información del canal (default True)
        """
        self.channel_type = channel_type
        self.fs = fs
        self.itu_profile = itu_profile
        self.frequency_ghz = frequency_ghz
        self.velocity_kmh = velocity_kmh
        
        if channel_type == 'awgn':
            self.channel = AWGNChannel(snr_db)
        elif channel_type == 'fading':
            self.channel = FadingChannel(snr_db)
        elif channel_type == 'rayleigh_mp':
            if fs is None:
                raise ValueError("Se requiere fs (frecuencia de muestreo) para canal Rayleigh")
            self.channel = RayleighMultiPathChannel(
                snr_db, fs, itu_profile, 
                frequency_ghz=frequency_ghz, 
                velocity_kmh=velocity_kmh,
                verbose=verbose  # Pass verbose flag
            )
        else:
            raise ValueError(f"Tipo de canal desconocido: {channel_type}")

    
    def transmit(self, signal):
        """
        Transmite a través del canal
        
        Args:
            signal: Señal a transmitir
            
        Returns:
            Signal recibida
        """
        received, _ = self.channel.transmit(signal)
        return received
    
    def set_snr(self, snr_db):
        """Establece el SNR"""
        self.channel.set_snr(snr_db)
    
    def set_channel_type(self, channel_type, **kwargs):
        """
        Cambia el tipo de canal en tiempo real
        
        Args:
            channel_type: 'awgn', 'fading', o 'rayleigh_mp'
            **kwargs: parámetros adicionales (itu_profile, etc.)
        """
        self.channel_type = channel_type
        
        if channel_type == 'awgn':
            self.channel = AWGNChannel(self.channel.snr_db if hasattr(self.channel, 'snr_db') else 10.0)
        elif channel_type == 'fading':
            self.channel = FadingChannel(self.channel.snr_db if hasattr(self.channel, 'snr_db') else 10.0)
        elif channel_type == 'rayleigh_mp':
            if self.fs is None:
                raise ValueError("Se requiere fs para cambiar a canal Rayleigh")
            itu_profile = kwargs.get('itu_profile', self.itu_profile)
            self.itu_profile = itu_profile
            snr_db = self.channel.snr_db if hasattr(self.channel, 'snr_db') else 10.0
            self.channel = RayleighMultiPathChannel(snr_db, self.fs, itu_profile)
        else:
            raise ValueError(f"Tipo de canal desconocido: {channel_type}")
    
    def set_itu_profile(self, itu_profile):
        """Cambia el perfil ITU si el canal actual es Rayleigh"""
        if isinstance(self.channel, RayleighMultiPathChannel):
            self.channel.set_profile(itu_profile)
            self.itu_profile = itu_profile
        else:
            raise ValueError("Solo se puede cambiar perfil ITU en canal Rayleigh")
    
    def get_channel(self):
        """Retorna el objeto del canal actual"""
        return self.channel
    
    def get_channel_info(self):
        """Retorna información del canal actual"""
        if hasattr(self.channel, 'get_channel_info'):
            return self.channel.get_channel_info()
        else:
            return {
                'type': self.channel_type,
                'SNR_dB': self.channel.snr_db if hasattr(self.channel, 'snr_db') else None
            }
