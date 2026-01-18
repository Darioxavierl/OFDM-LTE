"""
Layer Mapper/Demapper para Spatial Multiplexing (TS 36.211 Section 6.3.3)

Mapea símbolos QAM a múltiples capas espaciales (layers) para transmisión MIMO.
Soporta rank 1-8 según especificación LTE.

En LTE, el layer mapping distribuye símbolos entre capas para aprovechar 
diversidad espacial y multiplexación.
"""

import numpy as np


class LayerMapper:
    """
    Mapea símbolos modulados a capas espaciales para transmisión MIMO.
    
    Mapping según TS 36.211 Table 6.3.3.2-1:
    - Rank 1: Todos los símbolos en una capa
    - Rank 2-8: Símbolos distribuidos round-robin entre capas
    """
    
    def __init__(self, num_layers):
        """
        Args:
            num_layers (int): Número de capas espaciales (rank). Rango: 1-8
        """
        if num_layers < 1 or num_layers > 8:
            raise ValueError(f"num_layers debe estar en [1,8], recibido: {num_layers}")
        
        self.num_layers = num_layers
        
        print(f"[LayerMapper] Inicializado con {num_layers} capas espaciales")
    
    def map_to_layers(self, symbols):
        """
        Mapea símbolos a capas espaciales.
        
        Distribución round-robin según LTE:
        - symbols[0] → layer 0
        - symbols[1] → layer 1
        - symbols[2] → layer 2 (si rank >= 3)
        - ...
        - symbols[num_layers] → layer 0
        - symbols[num_layers+1] → layer 1
        - etc.
        
        Args:
            symbols: Array de símbolos complejos [num_symbols]
        
        Returns:
            layers: Array [num_layers, symbols_per_layer]
                    Símbolos distribuidos entre capas
        
        Example:
            >>> mapper = LayerMapper(num_layers=2)
            >>> symbols = np.array([s0, s1, s2, s3, s4, s5])
            >>> layers = mapper.map_to_layers(symbols)
            >>> # layers[0] = [s0, s2, s4]
            >>> # layers[1] = [s1, s3, s5]
        """
        symbols = np.asarray(symbols)
        num_symbols = len(symbols)
        
        if self.num_layers == 1:
            # Rank 1: todos los símbolos en una sola capa
            return symbols.reshape(1, -1)
        
        # Calcular símbolos por capa
        symbols_per_layer = num_symbols // self.num_layers
        remainder = num_symbols % self.num_layers
        
        if remainder != 0:
            # Pad con ceros si no es divisible
            pad_size = self.num_layers - remainder
            symbols = np.concatenate([symbols, np.zeros(pad_size, dtype=symbols.dtype)])
            symbols_per_layer = len(symbols) // self.num_layers
        
        # Reshape para distribuir símbolos
        # Método: reshape + transpose para round-robin
        symbols_reshaped = symbols.reshape(symbols_per_layer, self.num_layers).T
        
        return symbols_reshaped  # [num_layers, symbols_per_layer]
    
    def demap_from_layers(self, layers, original_length=None):
        """
        Inverso de map_to_layers: reconstruye símbolos desde capas.
        
        Args:
            layers: Array [num_layers, symbols_per_layer] de símbolos por capa
            original_length: Longitud original antes de padding (None = sin truncar)
        
        Returns:
            symbols: Array [num_symbols] reconstruido en orden original
        
        Example:
            >>> layers = np.array([[s0, s2, s4],
            ...                    [s1, s3, s5]])
            >>> symbols = mapper.demap_from_layers(layers)
            >>> # symbols = [s0, s1, s2, s3, s4, s5]
        """
        layers = np.asarray(layers)
        
        if self.num_layers == 1:
            # Rank 1: simplemente flatten
            symbols = layers.flatten()
        else:
            # Transpose + flatten para deshacer el round-robin
            symbols = layers.T.flatten()
        
        # Truncar si había padding
        if original_length is not None:
            symbols = symbols[:original_length]
        
        return symbols
    
    def get_symbols_per_layer(self, total_symbols):
        """
        Calcula cuántos símbolos irán en cada capa (con padding si necesario).
        
        Args:
            total_symbols: Número total de símbolos a mapear
        
        Returns:
            symbols_per_layer: Símbolos que tendrá cada capa
        """
        if self.num_layers == 1:
            return total_symbols
        
        symbols_per_layer = int(np.ceil(total_symbols / self.num_layers))
        return symbols_per_layer
    
    def get_padded_length(self, total_symbols):
        """
        Calcula la longitud total después de padding para ser divisible por num_layers.
        
        Args:
            total_symbols: Número original de símbolos
        
        Returns:
            padded_length: Longitud después de padding
        """
        if self.num_layers == 1:
            return total_symbols
        
        remainder = total_symbols % self.num_layers
        if remainder == 0:
            return total_symbols
        else:
            return total_symbols + (self.num_layers - remainder)


class LayerDemapper:
    """
    Alias para compatibilidad. Usa LayerMapper.demap_from_layers()
    """
    def __init__(self, num_layers):
        self.mapper = LayerMapper(num_layers)
    
    def demap(self, layers, original_length=None):
        return self.mapper.demap_from_layers(layers, original_length)


def test_layer_mapper():
    """
    Tests unitarios del LayerMapper
    """
    print("\n=== Test LayerMapper ===")
    
    # Test 1: Rank 1
    print("\n1. Test Rank-1:")
    mapper1 = LayerMapper(num_layers=1)
    symbols1 = np.array([1+1j, 2+2j, 3+3j, 4+4j])
    layers1 = mapper1.map_to_layers(symbols1)
    recovered1 = mapper1.demap_from_layers(layers1)
    print(f"   Símbolos: {symbols1}")
    print(f"   Layers shape: {layers1.shape}")
    print(f"   Recuperados: {recovered1}")
    assert np.allclose(symbols1, recovered1), "Error en rank-1"
    print("   ✓ Rank-1 OK")
    
    # Test 2: Rank 2
    print("\n2. Test Rank-2:")
    mapper2 = LayerMapper(num_layers=2)
    symbols2 = np.array([1, 2, 3, 4, 5, 6])
    layers2 = mapper2.map_to_layers(symbols2)
    print(f"   Símbolos: {symbols2}")
    print(f"   Layer 0: {layers2[0]}")
    print(f"   Layer 1: {layers2[1]}")
    recovered2 = mapper2.demap_from_layers(layers2, original_length=len(symbols2))
    print(f"   Recuperados: {recovered2}")
    assert np.allclose(symbols2, recovered2), "Error en rank-2"
    print("   ✓ Rank-2 OK")
    
    # Test 3: Rank 4
    print("\n3. Test Rank-4:")
    mapper4 = LayerMapper(num_layers=4)
    symbols4 = np.arange(16)
    layers4 = mapper4.map_to_layers(symbols4)
    print(f"   Símbolos: {symbols4}")
    print(f"   Layers shape: {layers4.shape}")
    for i, layer in enumerate(layers4):
        print(f"   Layer {i}: {layer}")
    recovered4 = mapper4.demap_from_layers(layers4)
    assert np.allclose(symbols4, recovered4), "Error en rank-4"
    print("   ✓ Rank-4 OK")
    
    # Test 4: Longitud no divisible (con padding)
    print("\n4. Test con Padding:")
    mapper2b = LayerMapper(num_layers=2)
    symbols_odd = np.array([1, 2, 3, 4, 5])  # 5 símbolos, no divisible por 2
    layers_odd = mapper2b.map_to_layers(symbols_odd)
    print(f"   Símbolos originales (5): {symbols_odd}")
    print(f"   Layers shape (con padding): {layers_odd.shape}")
    recovered_odd = mapper2b.demap_from_layers(layers_odd, original_length=len(symbols_odd))
    print(f"   Recuperados (truncados): {recovered_odd}")
    assert len(recovered_odd) == len(symbols_odd), "Error en truncado"
    assert np.allclose(symbols_odd, recovered_odd), "Error en recuperación con padding"
    print("   ✓ Padding OK")
    
    print("\n=== Todos los tests pasaron ✓ ===\n")


if __name__ == '__main__':
    test_layer_mapper()
