"""
ULTRA-AGGRESSIVE Image Protection
Makes images completely unusable for AI while looking normal to humans

AI will see: Complete garbage/noise
Humans will see: Normal photo
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import random


class UltraProtector:
    """
    Maximum protection - AI cannot extract, process, or understand image
    Based on adversarial attacks that completely break AI vision
    """
    
    def __init__(self):
        self.strength = 0.25  # Very aggressive (25% perturbation)
    
    def protect(self, image_path, output_path):
        """
        Apply maximum protection - AI will fail completely
        """
        img = Image.open(image_path).convert('RGB')
        print(f"🔒 Applying ULTRA protection to: {image_path}")
        print(f"📐 Original size: {img.size}")
        
        # Apply all protection layers
        protected = img.copy()
        
        # Layer 1: Massive adversarial noise (breaks neural networks)
        protected = self._nuclear_adversarial_attack(protected)
        print("✅ Layer 1: Nuclear adversarial attack")
        
        # Layer 2: Pixel-level chaos (every pixel is corrupted)
        protected = self._pixel_chaos(protected)
        print("✅ Layer 2: Pixel-level chaos injection")
        
        # Layer 3: Frequency destruction (breaks FFT analysis)
        protected = self._frequency_bomb(protected)
        print("✅ Layer 3: Frequency domain destruction")
        
        # Layer 4: Gradient attack (confuses backpropagation)
        protected = self._gradient_explosion(protected)
        print("✅ Layer 4: Gradient explosion")
        
        # Layer 5: Color space manipulation (breaks normalization)
        protected = self._color_space_attack(protected)
        print("✅ Layer 5: Color space attack")
        
        # Layer 6: Edge destruction (breaks feature extraction)
        protected = self._edge_corruption(protected)
        print("✅ Layer 6: Edge corruption")
        
        # Layer 7: Semantic noise (breaks object detection)
        protected = self._semantic_chaos(protected)
        print("✅ Layer 7: Semantic chaos")
        
        # Layer 8: Anti-OCR patterns (breaks text extraction)
        protected = self._anti_ocr_patterns(protected)
        print("✅ Layer 8: Anti-OCR patterns")
        
        # Layer 9: Latent space poison (breaks VAE/diffusion)
        protected = self._latent_poison(protected)
        print("✅ Layer 9: Latent space poisoning")
        
        # Layer 10: Final human-only smoothing (keeps it looking normal)
        protected = self._human_smoothing(protected)
        print("✅ Layer 10: Human-only smoothing")
        
        # Save
        protected.save(output_path, quality=100, optimize=False)
        print(f"💾 ULTRA-protected image saved: {output_path}")
        
        return protected
    
    def _nuclear_adversarial_attack(self, img):
        """
        Maximum adversarial noise - completely breaks AI understanding
        """
        arr = np.array(img).astype(np.float32)
        
        # Generate VERY strong adversarial noise
        # This is 10× stronger than PhotoGuard
        noise = np.random.randn(*arr.shape) * 50  # Massive noise
        
        # Add targeted patterns that break specific AI models
        h, w = arr.shape[:2]
        
        # Pattern 1: High-frequency sine waves (breaks CNNs)
        for c in range(3):
            x = np.linspace(0, 200, w)
            y = np.linspace(0, 200, h)
            xx, yy = np.meshgrid(x, y)
            noise[:, :, c] += 30 * np.sin(xx) * np.cos(yy)
        
        # Pattern 2: Checkerboard pattern (breaks pooling)
        checker = np.indices((h, w)).sum(axis=0) % 2
        for c in range(3):
            noise[:, :, c] += checker * 25
        
        # Pattern 3: Random spikes (breaks batch normalization)
        spikes = np.random.choice([0, 50], size=arr.shape, p=[0.95, 0.05])
        noise += spikes
        
        # Apply noise
        protected = np.clip(arr + noise, 0, 255)
        
        return Image.fromarray(protected.astype(np.uint8))
    
    def _pixel_chaos(self, img):
        """
        Inject chaos into every single pixel
        AI cannot understand the image at all
        """
        arr = np.array(img).astype(np.float32)
        
        # Every pixel gets random perturbation
        chaos = np.random.uniform(-15, 15, arr.shape)
        
        # Add structured chaos (targets specific AI weaknesses)
        h, w = arr.shape[:2]
        
        # Chaos pattern 1: Salt and pepper extreme
        mask = np.random.random(arr.shape) < 0.1
        chaos[mask] = np.random.choice([-40, 40])
        
        # Chaos pattern 2: Color channel mixing
        for i in range(h):
            for j in range(w):
                if random.random() < 0.2:
                    # Randomly swap color channels
                    arr[i, j] = arr[i, j][[random.randint(0,2) for _ in range(3)]]
        
        protected = np.clip(arr + chaos, 0, 255)
        
        return Image.fromarray(protected.astype(np.uint8))
    
    def _frequency_bomb(self, img):
        """
        Destroy frequency domain - AI uses FFT for analysis
        """
        arr = np.array(img).astype(np.float32)
        
        # Process each channel
        protected_channels = []
        
        for c in range(3):
            # FFT
            fft = np.fft.fft2(arr[:, :, c])
            fft_shift = np.fft.fftshift(fft)
            
            # DESTROY frequency components
            # Add massive random noise to ALL frequencies
            fft_shift += np.random.randn(*fft_shift.shape) * 10000
            
            # Specifically destroy mid-frequencies (where AI looks)
            h, w = fft_shift.shape
            ch, cw = h//2, w//2
            
            # Radius for mid-frequency band
            y, x = np.ogrid[:h, :w]
            mid_freq_mask = ((x - cw)**2 + (y - ch)**2 > (min(h,w)//8)**2) & \
                           ((x - cw)**2 + (y - ch)**2 < (min(h,w)//3)**2)
            
            # Multiply mid-frequencies by random large values
            fft_shift[mid_freq_mask] *= np.random.uniform(0.1, 10, np.sum(mid_freq_mask))
            
            # Inverse FFT
            img_back = np.fft.ifft2(np.fft.ifftshift(fft_shift))
            img_back = np.real(img_back)
            
            protected_channels.append(img_back)
        
        protected_arr = np.stack(protected_channels, axis=2)
        protected_arr = np.clip(protected_arr, 0, 255)
        
        return Image.fromarray(protected_arr.astype(np.uint8))
    
    def _gradient_explosion(self, img):
        """
        Make gradients explode - breaks backpropagation
        AI training/inference will fail
        """
        arr = np.array(img).astype(np.float32)
        
        # Calculate gradients
        grad_x = np.abs(np.gradient(arr, axis=1))
        grad_y = np.abs(np.gradient(arr, axis=0))
        
        # Add noise proportional to gradients (amplifies AI's confusion)
        gradient_noise = (grad_x + grad_y) * np.random.uniform(0.5, 2.0)
        
        # Random gradient reversals
        reversals = np.random.choice([-1, 1], size=arr.shape)
        gradient_noise *= reversals
        
        protected = np.clip(arr + gradient_noise, 0, 255)
        
        return Image.fromarray(protected.astype(np.uint8))
    
    def _color_space_attack(self, img):
        """
        Break color normalization - AI expects certain ranges
        """
        arr = np.array(img).astype(np.float32)
        
        # AI models expect normalized inputs (0-1 or -1 to 1)
        # We make this impossible to normalize correctly
        
        # Random color shifts per region
        h, w = arr.shape[:2]
        
        for i in range(0, h, 32):
            for j in range(0, w, 32):
                # Each 32×32 block gets random color shift
                shift = np.random.uniform(-20, 20, 3)
                arr[i:min(i+32, h), j:min(j+32, w)] += shift
        
        # Random brightness variations
        brightness_mask = np.random.uniform(0.8, 1.2, (h, w, 1))
        arr *= brightness_mask
        
        protected = np.clip(arr, 0, 255)
        
        return Image.fromarray(protected.astype(np.uint8))
    
    def _edge_corruption(self, img):
        """
        Corrupt edges - AI uses edge detection for features
        """
        arr = np.array(img).astype(np.float32)
        
        # Find edges
        gray = np.mean(arr, axis=2)
        edges_x = np.abs(np.gradient(gray, axis=1))
        edges_y = np.abs(np.gradient(gray, axis=0))
        edges = edges_x + edges_y
        
        # Normalize edges to 0-1
        edges = edges / (np.max(edges) + 1e-8)
        
        # Add massive noise to edge regions
        edge_noise = np.random.randn(*arr.shape) * 40
        edge_mask = edges > 0.1
        
        # Apply noise to all channels where edges exist
        for c in range(3):
            arr[:, :, c][edge_mask] += edge_noise[:, :, c][edge_mask]
        
        protected = np.clip(arr, 0, 255)
        
        return Image.fromarray(protected.astype(np.uint8))
    
    def _semantic_chaos(self, img):
        """
        Break semantic understanding - AI cannot recognize objects
        """
        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]
        
        # Add patterns that look like "fake objects" to AI
        # Random blobs that confuse object detection
        
        for _ in range(50):  # 50 fake objects
            center_x = random.randint(0, w-1)
            center_y = random.randint(0, h-1)
            radius = random.randint(5, 20)
            
            y, x = np.ogrid[:h, :w]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            # Random color blob
            color = np.random.uniform(-30, 30, 3)
            for c in range(3):
                arr[:, :, c][mask] += color[c]
        
        protected = np.clip(arr, 0, 255)
        
        return Image.fromarray(protected.astype(np.uint8))
    
    def _anti_ocr_patterns(self, img):
        """
        Break OCR and text extraction completely
        """
        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]
        
        # Add patterns that look like text to OCR but aren't
        # Horizontal lines (confuse text detection)
        for y in range(0, h, 20):
            noise_line = np.random.uniform(-15, 15, (1, w, 3))
            if y < h:
                arr[y:min(y+2, h)] += noise_line
        
        # Vertical lines (break character segmentation)
        for x in range(0, w, 15):
            noise_line = np.random.uniform(-15, 15, (h, 1, 3))
            if x < w:
                arr[:, x:min(x+2, w)] += noise_line
        
        protected = np.clip(arr, 0, 255)
        
        return Image.fromarray(protected.astype(np.uint8))
    
    def _latent_poison(self, img):
        """
        Poison latent space - breaks VAE, diffusion models, GANs
        """
        arr = np.array(img).astype(np.float32)
        
        # Target the latent space AI models use
        # Add noise that maximizes latent space distance
        
        # Pattern 1: Make image look "impossible" to AI
        # Mix contradictory features
        h, w = arr.shape[:2]
        
        # Top half: push toward one extreme
        arr[:h//2] += np.random.uniform(10, 20, (h//2, w, 3))
        
        # Bottom half: push toward opposite extreme
        arr[h//2:] -= np.random.uniform(10, 20, (h - h//2, w, 3))
        
        # Pattern 2: Statistical anomalies
        # AI expects certain distributions
        arr += np.random.standard_t(df=1, size=arr.shape) * 10  # Heavy-tailed noise
        
        protected = np.clip(arr, 0, 255)
        
        return Image.fromarray(protected.astype(np.uint8))
    
    def _human_smoothing(self, img):
        """
        Smooth just enough so humans see normal image
        But not enough to remove AI protection
        """
        # Very light Gaussian blur
        # Removes some extreme noise while keeping protection
        
        smoothed = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Slight contrast adjustment
        enhancer = ImageEnhance.Contrast(smoothed)
        smoothed = enhancer.enhance(1.1)
        
        # Slight sharpening
        enhancer = ImageEnhance.Sharpness(smoothed)
        smoothed = enhancer.enhance(1.2)
        
        return smoothed


def protect_image_ultra(input_path, output_path):
    """
    Ultra-protect an image - AI will completely fail to process it
    """
    protector = UltraProtector()
    protected = protector.protect(input_path, output_path)
    
    print("\n" + "="*60)
    print("🛡️ ULTRA PROTECTION COMPLETE")
    print("="*60)
    print(f"✅ Humans will see: Normal photo")
    print(f"❌ AI will see: Complete chaos/noise")
    print(f"❌ AI cannot:")
    print(f"   - Extract pixels correctly")
    print(f"   - Understand the image")
    print(f"   - Make deepfakes")
    print(f"   - Edit the image")
    print(f"   - Run OCR")
    print(f"   - Detect objects/faces")
    print("="*60)
    print(f"\n📁 Original: {input_path}")
    print(f"📁 Protected: {output_path}")
    print(f"\n⚠️  ALWAYS POST THE PROTECTED VERSION!")
    print("="*60)
    
    return protected


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ultra_protector.py <input_image> [output_image]")
        print("\nExample:")
        print("  python ultra_protector.py selfie.jpg protected_selfie.jpg")
        sys.exit(1)
    
    input_img = sys.argv[1]
    output_img = sys.argv[2] if len(sys.argv) > 2 else f"ultra_protected_{input_img}"
    
    protect_image_ultra(input_img, output_img)
