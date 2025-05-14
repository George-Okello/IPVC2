# simple_gui.py
"""
Simple Tkinter GUI for the detection system
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import os
import threading
from detector import CodeDetector


class SimpleGUI:
    """Simple GUI for barcode and QR code detection"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Barcode & QR Code Detector")
        self.root.geometry("800x600")
        
        self.detector = CodeDetector()
        self.current_image = None
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Title
        title = ttk.Label(self.root, text="Barcode & QR Code Detector", 
                         font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # Button frame
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Select Image", 
                  command=self.select_image).pack(side=tk.LEFT, padx=5)
        
        self.process_button = ttk.Button(button_frame, text="Process Image", 
                                        command=self.process_image, state="disabled")
        self.process_button.pack(side=tk.LEFT, padx=5)
        
        # Image display
        self.image_frame = ttk.LabelFrame(self.root, text="Image", padding=10)
        self.image_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.image_label = ttk.Label(self.image_frame, text="No image selected", 
                                    relief="sunken")
        self.image_label.pack(fill="both", expand=True)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(fill="x", padx=10, pady=5)
        
        # Results
        results_frame = ttk.LabelFrame(self.root, text="Results", padding=10)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=8)
        self.results_text.pack(fill="both", expand=True)
        
        # Status bar
        self.status = ttk.Label(self.root, text="Ready", relief="sunken")
        self.status.pack(fill="x", side=tk.BOTTOM)
    
    def select_image(self):
        """Select an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image = file_path
            self.display_image(file_path)
            self.process_button.config(state="normal")
            self.status.config(text=f"Selected: {os.path.basename(file_path)}")
    
    def display_image(self, image_path):
        """Display image in the GUI"""
        try:
            # Load and resize image
            image = Image.open(image_path)
            image.thumbnail((400, 300), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep reference
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")
    
    def process_image(self):
        """Process the selected image"""
        if not self.current_image:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        # Start processing in thread
        self.process_button.config(state="disabled")
        self.progress.start()
        self.status.config(text="Processing...")
        
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        
        thread = threading.Thread(target=self._process_thread)
        thread.daemon = True
        thread.start()
    
    def _process_thread(self):
        """Process image in background thread"""
        try:
            # Process image
            result = self.detector.detect_codes(self.current_image)
            
            # Update GUI in main thread
            self.root.after(0, self._update_results, result)
        except Exception as e:
            self.root.after(0, self._show_error, str(e))
        finally:
            self.root.after(0, self._finish_processing)
    
    def _update_results(self, result):
        """Update results display"""
        # Format results
        text = f"Processing time: {result['processing_time']:.3f} seconds\n"
        text += f"Barcode regions: {len(result['barcode_regions'])}\n"
        text += f"QR code regions: {len(result['qr_regions'])}\n"
        text += f"Codes decoded: {result['total_codes']}\n\n"
        
        if result['detected_codes']:
            text += "Detected Codes:\n"
            text += "-" * 30 + "\n"
            for i, code in enumerate(result['detected_codes'], 1):
                text += f"{i}. {code['type']}: {code['data']}\n"
        else:
            text += "No codes detected\n"
        
        self.results_text.insert(1.0, text)
        
        # Show visualization if available
        vis_path = os.path.splitext(self.current_image)[0] + '_detected.jpg'
        if os.path.exists(vis_path):
            self.display_image(vis_path)
    
    def _show_error(self, error_msg):
        """Show error message"""
        messagebox.showerror("Error", f"Processing failed: {error_msg}")
    
    def _finish_processing(self):
        """Clean up after processing"""
        self.process_button.config(state="normal")
        self.progress.stop()
        self.status.config(text="Processing complete")


def main():
    """Run the GUI application"""
    root = tk.Tk()
    app = SimpleGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()