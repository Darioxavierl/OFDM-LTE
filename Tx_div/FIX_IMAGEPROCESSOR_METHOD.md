# Fix: Error en Carga de Imagen para Barrido SNR

## Problema Reportado

El usuario reportó que al ejecutar el barrido SNR con imagen cargada, el mensaje de debug mostraba que estaba usando bits aleatorios en lugar de los bits de la imagen.

## Causa del Error

El código estaba usando un método **inexistente** de `ImageProcessor`:

```python
# ❌ INCORRECTO
img = ImageProcessor.load_image(self.current_image_path)
bits, metadata = ImageProcessor.image_to_bits(img)
```

**Problema:** `ImageProcessor` NO tiene un método `load_image()`. 

## Métodos Disponibles en ImageProcessor

```python
class ImageProcessor:
    @staticmethod
    def image_to_bits(image_path):  # ← Recibe PATH directamente
        """Convierte imagen a bits"""
        
    @staticmethod
    def bits_to_image(bits, metadata):
        """Convierte bits a imagen"""
        
    @staticmethod
    def calculate_psnr(original_img, reconstructed_img):
        """Calcula PSNR"""
        
    @staticmethod
    def calculate_ssim(original_img, reconstructed_img):
        """Calcula SSIM"""
```

**Nota:** `image_to_bits()` recibe la **ruta (path)** directamente, no un objeto Image.

## Solución Implementada

**Línea ~791 en `run_sweep_simulation()`:**

```python
# ✅ CORRECTO
if self.current_image_path:
    try:
        bits, metadata = ImageProcessor.image_to_bits(self.current_image_path)
        params['bits'] = bits
        params['metadata'] = metadata
        print(f"[INFO] ✓ Imagen cargada exitosamente en params:")
        print(f"  - Ruta: {self.current_image_path}")
        print(f"  - Bits: {len(bits):,}")
        print(f"  - Dimensiones: {metadata['width']}×{metadata['height']}")
    except Exception as e:
        print(f"[WARNING] No se pudo cargar imagen: {e}")
        import traceback
        traceback.print_exc()
```

**Cambio:** 
- Llamada directa a `ImageProcessor.image_to_bits(path)`
- Se eliminó la línea inexistente `load_image()`
- Se agregó traceback para debugging

## Verificación

### Antes (con error):
```
[WARNING] No se pudo cargar imagen, usando bits aleatorios: 
    type object 'ImageProcessor' has no attribute 'load_image'

======================================================================
[DEBUG TxDiv] Barrido SNR - SIN IMAGEN (usando bits aleatorios)
======================================================================
  Total bits aleatorios: 50,000 bits
```

### Ahora (corregido):
```
[INFO] ✓ Imagen cargada exitosamente en params:
  - Ruta: img/entre-ciel-et-terre.jpg
  - Bits: 1,920,000
  - Dimensiones: 800×600

======================================================================
[DEBUG TxDiv] Barrido SNR - IMAGEN CARGADA
======================================================================
  Dimensiones imagen: 800×600
  Total bits de imagen: 1,920,000 bits
  Tamaño datos: 234.38 KB
  ✓ Se enviarán TODOS los bits de la imagen en cada SNR
======================================================================
```

## Otros Usos Correctos en el Código

### Simulación Única (línea ~59)
```python
# ✅ Ya estaba correcto
if 'image_path' in self.params:
    bits, metadata = ImageProcessor.image_to_bits(self.params['image_path'])
```

### MIMO Comparison (línea ~313)
```python
# ✅ Ya estaba correcto
bits, metadata = ImageProcessor.image_to_bits(self.params['image_path'])
```

## Estado
✅ **RESUELTO** - Ahora el barrido SNR carga correctamente los bits de la imagen
