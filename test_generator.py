# test_generator.py
"""
Simple test image generator for testing the detection system
"""
import qrcode
import barcode
from barcode.writer import ImageWriter
import os
from PIL import Image, ImageDraw


class TestGenerator:
    """Generate test images with barcodes and QR codes"""
    
    def __init__(self, output_dir="test_images"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_qr_code(self, text, filename):
        """Create a QR code image"""
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(text)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        img = img.resize((300, 300))
        img.save(os.path.join(self.output_dir, filename))
        return img
    
    def create_barcode(self, text, filename, code_type='code128'):
        """Create a barcode image"""
        try:
            if code_type == 'ean13' and len(text) != 12:
                text = text.ljust(12, '0')[:12]
            
            BarcodeClass = barcode.get_barcode_class(code_type)
            barcode_instance = BarcodeClass(text, writer=ImageWriter())
            
            full_path = os.path.join(self.output_dir, filename)
            barcode_instance.save(full_path.rsplit('.', 1)[0])
            
            # Rename file if needed
            png_path = full_path.rsplit('.', 1)[0] + '.png'
            if os.path.exists(png_path) and png_path != full_path:
                os.rename(png_path, full_path)
            
            return Image.open(full_path)
        except Exception as e:
            print(f"Error creating barcode: {e}")
            return None
    
    def create_simple_test_set(self):
        """Create a simple set of test images"""
        print("Creating test images...")
        
        # Simple QR code
        self.create_qr_code("Hello World!", "qr_simple.png")
        
        # URL QR code
        self.create_qr_code("https://www.example.com", "qr_url.png")
        
        # Simple barcode
        self.create_barcode("123456789012", "barcode_ean13.png", "ean13")
        
        # Code128 barcode
        self.create_barcode("ABC12345", "barcode_code128.png", "code128")
        
        # Combined image with both codes
        self.create_combined_image()
        
        print(f"Test images created in {self.output_dir}")
    
    def create_combined_image(self):
        """Create an image with both QR code and barcode"""
        # Create blank image
        canvas = Image.new('RGB', (600, 400), 'white')
        draw = ImageDraw.Draw(canvas)
        
        # Add QR code
        qr_img = self.create_qr_code("QR Code Test", "temp_qr.png")
        qr_img = qr_img.resize((150, 150))
        canvas.paste(qr_img, (50, 50))
        
        # Add barcode
        barcode_img = self.create_barcode("456789123456", "temp_barcode.png", "code128")
        if barcode_img:
            barcode_img = barcode_img.resize((200, 80))
            canvas.paste(barcode_img, (350, 100))
        
        # Add labels
        draw.text((50, 220), "QR Code", fill="black")
        draw.text((350, 200), "Barcode", fill="black")
        
        canvas.save(os.path.join(self.output_dir, "combined.png"))


def main():
    """Create test images"""
    generator = TestGenerator()
    generator.create_simple_test_set()


if __name__ == "__main__":
    main()