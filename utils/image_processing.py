"""
Utilidades para procesamiento de imágenes
"""
import numpy as np
from PIL import Image
import io


class ImageProcessor:
    """Clase para convertir imágenes a bits y viceversa"""
    
    @staticmethod
    def image_to_bits(image_path):
        """
        Convierte una imagen a array de bits
        
        Args:
            image_path: Ruta de la imagen
            
        Returns:
            tuple: (bits_array, metadata_dict)
        """
        # Cargar imagen
        img = Image.open(image_path)
        
        # Convertir a RGB si es necesario
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convertir a numpy array
        img_array = np.array(img)
        
        # Metadata
        height, width, channels = img_array.shape
        metadata = {
            'height': height,
            'width': width,
            'channels': channels,
            'dtype': str(img_array.dtype)
        }
        
        # Convertir píxeles a bits
        # Cada píxel tiene 3 canales (RGB), cada canal es uint8 (8 bits)
        img_flat = img_array.flatten()
        
        # Convertir cada byte a 8 bits
        bits = np.unpackbits(img_flat)
        
        return bits, metadata
    
    @staticmethod
    def bits_to_image(bits, metadata):
        """
        Reconstruye una imagen desde bits
        
        Args:
            bits: Array de bits
            metadata: Diccionario con información de la imagen
            
        Returns:
            PIL Image
        """
        height = metadata['height']
        width = metadata['width']
        channels = metadata['channels']
        
        # Calcular número total de bits esperados
        expected_bits = height * width * channels * 8
        
        # Truncar o rellenar si es necesario
        if len(bits) < expected_bits:
            # Rellenar con ceros
            bits = np.pad(bits, (0, expected_bits - len(bits)), 'constant')
        else:
            bits = bits[:expected_bits]
        
        # Convertir bits a bytes
        img_flat = np.packbits(bits)
        
        # Reshape a imagen
        try:
            img_array = img_flat.reshape(height, width, channels)
            
            # Convertir a imagen PIL
            img = Image.fromarray(img_array.astype(np.uint8), 'RGB')
            
            return img
        except Exception as e:
            print(f"Error al reconstruir imagen: {e}")
            # Retornar imagen en blanco del tamaño correcto
            return Image.new('RGB', (width, height), color='black')
    
    @staticmethod
    def calculate_psnr(original_img, reconstructed_img):
        """
        Calcula el Peak Signal-to-Noise Ratio entre dos imágenes
        
        Args:
            original_img: Imagen original (PIL Image o numpy array)
            reconstructed_img: Imagen reconstruida (PIL Image o numpy array)
            
        Returns:
            PSNR en dB
        """
        # Convertir a numpy arrays si son PIL Images
        if isinstance(original_img, Image.Image):
            original_img = np.array(original_img)
        if isinstance(reconstructed_img, Image.Image):
            reconstructed_img = np.array(reconstructed_img)
        
        # Asegurar mismo tamaño
        if original_img.shape != reconstructed_img.shape:
            # Resize reconstructed to match original
            reconstructed_pil = Image.fromarray(reconstructed_img)
            reconstructed_pil = reconstructed_pil.resize(
                (original_img.shape[1], original_img.shape[0])
            )
            reconstructed_img = np.array(reconstructed_pil)
        
        # Calcular MSE (Mean Squared Error)
        mse = np.mean((original_img.astype(float) - reconstructed_img.astype(float)) ** 2)
        
        if mse == 0:
            return float('inf')  # Imágenes idénticas
        
        # Calcular PSNR
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        return psnr
    
    @staticmethod
    def calculate_ssim(original_img, reconstructed_img):
        """
        Calcula el Structural Similarity Index entre dos imágenes
        
        Args:
            original_img: Imagen original (PIL Image o numpy array)
            reconstructed_img: Imagen reconstruida (PIL Image o numpy array)
            
        Returns:
            SSIM (0-1, donde 1 es idéntico)
        """
        try:
            from skimage.metrics import structural_similarity as ssim
            
            # Convertir a numpy arrays si son PIL Images
            if isinstance(original_img, Image.Image):
                original_img = np.array(original_img)
            if isinstance(reconstructed_img, Image.Image):
                reconstructed_img = np.array(reconstructed_img)
            
            # Asegurar mismo tamaño
            if original_img.shape != reconstructed_img.shape:
                reconstructed_pil = Image.fromarray(reconstructed_img)
                reconstructed_pil = reconstructed_pil.resize(
                    (original_img.shape[1], original_img.shape[0])
                )
                reconstructed_img = np.array(reconstructed_pil)
            
            # Calcular SSIM (multicanal para RGB)
            ssim_value = ssim(original_img, reconstructed_img, 
                            channel_axis=2, data_range=255)
            
            return ssim_value
        except ImportError:
            print("scikit-image no disponible para calcular SSIM")
            return None
    
    @staticmethod
    def save_comparison(original_path, reconstructed_img, output_path):
        """
        Guarda una comparación lado a lado de la imagen original y reconstruida
        
        Args:
            original_path: Ruta de la imagen original
            reconstructed_img: Imagen reconstruida (PIL Image)
            output_path: Ruta para guardar la comparación
        """
        original_img = Image.open(original_path)
        
        # Asegurar mismo tamaño
        if original_img.size != reconstructed_img.size:
            reconstructed_img = reconstructed_img.resize(original_img.size)
        
        # Crear imagen de comparación
        width, height = original_img.size
        comparison = Image.new('RGB', (width * 2, height))
        
        comparison.paste(original_img, (0, 0))
        comparison.paste(reconstructed_img, (width, 0))
        
        comparison.save(output_path)
        
        return comparison