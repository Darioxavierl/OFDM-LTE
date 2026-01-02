import numpy as np
import scipy.signal as signal
from typing import Optional, Dict, List, Any, Tuple

class RayleighChannel:
    def __init__(self, Fs, fD, delays, gains):
        """
        Fs: frecuencia de muestreo (Hz)
        fD: Doppler máximo (Hz)
        delays: array de retardos (s)
        gains: array de ganancias (dB)
        """
        self.Fs = Fs
        self.fD = fD
        self.delays = np.array(delays)
        self.gains = 10**(np.array(gains)/20)
        assert len(self.delays) == len(self.gains), "delays y gains deben tener la misma longitud"
        self.num_paths = len(delays)
    
    def jakes_fading(self, N, N_s=16):
        """
        Genera un proceso Rayleigh fading basado en el modelo clásico de Jakes.
        
        N     : número de muestras
        N_s   : número de ondas sinusoidales para aproximar el espectro Doppler
        """
        t = np.arange(N) / self.Fs
        fD = self.fD

        # Generamos fases aleatorias
        phi_n = 2 * np.pi * np.random.rand(N_s)
        alpha_n = 2 * np.pi * np.arange(1, N_s+1) / N_s  # ángulos uniformemente distribuidos

        # Inicializamos señal compleja
        h = np.zeros(N, dtype=complex)

        for n in range(N_s):
            h += np.exp(1j * (2*np.pi*fD*np.cos(alpha_n[n])*t + phi_n[n]))

        # Normalizamos para que la potencia media sea 1
        h = h * np.sqrt(2 / N_s)
        return h
    
    def filter(self, x):
        """
        Aplica el canal Rayleigh a una señal de entrada x.
        """
        N = len(x)
        y = np.zeros(N, dtype=complex)

        for i in range(self.num_paths):
            delay_samples = int(np.round(self.delays[i] * self.Fs))
            fading = self.jakes_fading(N)
            # aplicar retardo y fading
            x_delayed = np.concatenate([np.zeros(delay_samples), x])[:N]
            y += self.gains[i] * fading * x_delayed

        return y
    
    def large_scale_fading(self, d, fc, PL0=30, n=3.5, sigma=4, d0=100):
        """
        Modelo log-distance + shadowing
        d: distancia (m)
        fc: frecuencia portadora (Hz)
        PL0: pérdida en dB a 1m (referencia)
        n: exponente de pérdida
        sigma: desviación estándar del shadowing en dB
        """
        # pérdida determinística
        PL_dB = PL0 + 10*n*np.log10(d/d0)
        # shadowing log-normal
        shadowing = np.random.normal(0, sigma)
        total_loss_dB = PL_dB + shadowing
        return 10**(-total_loss_dB/20)  # factor lineal
    
    def channel_response(self, freqs, h_taps, N_freq=None):
        """
        Calcula la respuesta en frecuencia del canal a partir de la salida temporal del canal.

        freqs : array
            Array de frecuencias
        h_taps : array
            Taps del canal
        N_freq : int, opcional
            Número de puntos para la FFT.
        """
        Hf = np.zeros_like(freqs, dtype=complex)

        for i in range(self.num_paths):
            Hf += h_taps[i] * np.exp(-1j * 2 * np.pi * freqs * self.delays[i])

        return Hf

    
    def impulse_response(self, N=1):
        """
        Devuelve la respuesta al impulso compleja del canal Rayleigh (taps).
        Cada camino se representa con su ganancia y retardo correspondiente.
        """
        taps = []
        delays_out = []

        for i in range(self.num_paths):
            fading = self.jakes_fading(N)[0]  # una muestra instantánea de fading
            tap_value = self.gains[i] * fading
            taps.append(tap_value)
            delays_out.append(self.delays[i])

        return np.array(delays_out), np.array(taps)
