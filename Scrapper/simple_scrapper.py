"""
Simple script to scrape barcode and QR code images for manual inspection
"""
import requests
import os
import time
from PIL import Image


class SimpleImageScraper:
    """Simple scraper to download barcode and QR code images"""

    def __init__(self, output_dir="downloaded_images"):
        self.output_dir = output_dir
        os.makedirs(f"{output_dir}/barcodes", exist_ok=True)
        os.makedirs(f"{output_dir}/qr_codes", exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        self.downloaded = 0

    def download_from_wikimedia(self):
        """Download from Wikimedia Commons (free to use)"""
        print("Downloading from Wikimedia Commons...")

        # Direct barcode image links
        barcode_urls = [
            "https://upload.wikimedia.org/wikipedia/commons/3/32/UPC-A-036000291452.svg",
            "https://upload.wikimedia.org/wikipedia/commons/2/28/EAN-13-5901234123457.svg",
            "https://upload.wikimedia.org/wikipedia/commons/0/0b/UPC_barcode.png",
            "https://upload.wikimedia.org/wikipedia/commons/5/50/EAN-13_barcode.svg",
            "https://upload.wikimedia.org/wikipedia/commons/9/99/CODE128_Barcode.svg",
            "https://upload.wikimedia.org/wikipedia/commons/2/24/ITF-14_barcode.svg",
            "https://upload.wikimedia.org/wikipedia/commons/1/1b/Code_39.svg"
        ]

        # Direct QR code image links
        qr_urls = [
            "https://upload.wikimedia.org/wikipedia/commons/d/d8/QR_Code_Example.svg",
            "https://upload.wikimedia.org/wikipedia/commons/7/7d/Qr-code-hello-world.svg",
            "https://upload.wikimedia.org/wikipedia/commons/4/4f/QR_code_example.png",
            "https://upload.wikimedia.org/wikipedia/commons/2/2f/Wikimedia-QR-code.png",
            "https://upload.wikimedia.org/wikipedia/commons/0/0b/QR_code_for_mobile_English_Wikipedia.svg"
        ]

        # Download barcodes
        print("  Downloading barcode images...")
        for i, url in enumerate(barcode_urls):
            self.download_image(url, f"barcodes/wikimedia_barcode_{i + 1}")
            time.sleep(1)

        # Download QR codes
        print("  Downloading QR code images...")
        for i, url in enumerate(qr_urls):
            self.download_image(url, f"qr_codes/wikimedia_qr_{i + 1}")
            time.sleep(1)

    def download_sample_images(self):
        """Download from various free sources"""
        print("Downloading sample images...")

        sample_images = [
            {
                'url': 'https://www.python-barcode.org/_images/ean13.png',
                'filename': 'barcodes/sample_ean13'
            },
            {
                'url': 'https://chart.googleapis.com/chart?chs=300x300&cht=qr&chl=Hello%20World&choe=UTF-8',
                'filename': 'qr_codes/google_qr_hello'
            },
            {
                'url': 'https://chart.googleapis.com/chart?chs=300x300&cht=qr&chl=https://example.com&choe=UTF-8',
                'filename': 'qr_codes/google_qr_url'
            },
            {
                'url': 'https://barcode.orcascan.com/img/code128.png',
                'filename': 'barcodes/sample_code128'
            }
        ]

        for img in sample_images:
            try:
                self.download_image(img['url'], img['filename'])
                time.sleep(2)  # Be respectful
            except:
                print(f"    Skipped: {img['filename']}")

    def download_image(self, url, filename_base):
        """Download a single image"""
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                # Determine file extension
                content_type = response.headers.get('content-type', '')
                if 'image/png' in content_type:
                    ext = '.png'
                elif 'image/svg' in content_type:
                    ext = '.svg'
                elif 'image/jpeg' in content_type or 'image/jpg' in content_type:
                    ext = '.jpg'
                else:
                    ext = '.png'  # Default

                filename = f"{filename_base}{ext}"
                filepath = os.path.join(self.output_dir, filename)

                with open(filepath, 'wb') as f:
                    f.write(response.content)

                try:
                    img = Image.open(filepath)
                    img.verify()
                    print(f"    ✓ Downloaded: {filename}")
                    self.downloaded += 1
                except Exception:
                    os.remove(filepath)
                    print(f"    ✗ Invalid image: {filename}")
            else:
                print(f"    ✗ Failed to download: {url}")
        except Exception as e:
            print(f"    ✗ Error downloading {url}: {e}")

    def create_sample_codes(self):
        """Create some sample codes for testing"""
        print("Creating sample codes...")

        try:
            import qrcode
            import barcode
            from barcode.writer import ImageWriter

            # QR samples
            qr_samples = [
                ("Hello World!", "sample_hello"),
                ("https://www.example.com", "sample_url"),
                ("Contact: John Doe\nPhone: +1234567890\nEmail: john@example.com", "sample_contact"),
                ("WiFi:T:WPA;S:MyNetwork;P:password123;;", "sample_wifi")
            ]

            for data, name in qr_samples:
                qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L)
                qr.add_data(data)
                qr.make(fit=True)
                img = qr.make_image(fill_color="black", back_color="white")
                filepath = os.path.join(self.output_dir, f"qr_codes/{name}.png")
                img.save(filepath)
                print(f"    ✓ Created QR: {name}.png")

            # Barcode samples
            barcode_samples = [
                ("123456789012", "ean13", "sample_ean13"),
                ("12345678", "ean8", "sample_ean8"),
                ("HELLO123", "code39", "sample_code39"),
                ("Sample123", "code128", "sample_code128")
            ]

            for data, code_type, name in barcode_samples:
                try:
                    BarcodeClass = barcode.get_barcode_class(code_type)
                    barcode_instance = BarcodeClass(data, writer=ImageWriter())
                    filepath = os.path.join(self.output_dir, f"barcodes/{name}")
                    barcode_instance.save(filepath)
                    print(f"    ✓ Created Barcode: {name}.png")
                except Exception as e:
                    print(f"    ✗ Error creating {name}: {e}")

        except ImportError:
            print("    qrcode or python-barcode not installed. Skipping sample creation.")

    def run_scraping(self):
        """Run all scraping methods"""
        print("Starting image scraping...")
        print("=" * 40)

        self.download_from_wikimedia()
        self.download_sample_images()
        self.create_sample_codes()

        print("=" * 40)
        print(f"Scraping complete! Downloaded {self.downloaded} images")
        print(f"Images saved in '{self.output_dir}' folder:")
        print(f"  - Barcodes: {self.output_dir}/barcodes/")
        print(f"  - QR Codes: {self.output_dir}/qr_codes/")
        print("\nYou can now manually select and organize these images.")


def main():
    """Main function"""
    scraper = SimpleImageScraper()
    scraper.run_scraping()


if __name__ == "__main__":
    main()
