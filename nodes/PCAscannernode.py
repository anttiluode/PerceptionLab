"""
PC Scanner Node - Automatically scans through all principal components
Creates a contact sheet showing what each PC controls
"""

import numpy as np
from PyQt6 import QtGui
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class PCScannerNode(BaseNode):
    """
    Systematically scans through all PCs to visualize their effects.
    Creates a grid showing: [PC0-, PC0+, PC1-, PC1+, ...]
    """
    NODE_CATEGORY = "AI / Physics"
    NODE_COLOR = QtGui.QColor(220, 180, 100)
    
    def __init__(self, scan_amplitude=2.0, grid_cols=4):
        super().__init__()
        self.node_title = "PC Scanner"
        
        self.inputs = {
            'latent_in': 'spectrum',
            'reconstructed_image': 'image',  # Feedback from iFFT
            'scan_speed': 'signal',  # How fast to scan
            'trigger': 'signal',  # Start scan
            'amplitude': 'signal'  # How much to modify each PC
        }
        self.outputs = {
            'latent_out': 'spectrum',  # Modified latent for current scan
            'contact_sheet': 'image',  # The full grid
            'current_pc': 'signal',  # Which PC we're scanning
            'progress': 'signal'  # 0-1 scan progress
        }
        
        self.scan_amplitude = float(scan_amplitude)
        self.grid_cols = int(grid_cols)
        
        # Scanning state
        self.is_scanning = False
        self.scan_index = 0  # Which PC we're currently scanning
        self.scan_direction = 1  # 1 for +, -1 for -
        self.frame_counter = 0
        self.frames_per_scan = 30  # How many frames to wait per PC
        
        # Storage
        self.base_latent = None
        self.current_latent = None
        self.captured_images = {}  # {(pc_idx, direction): image}
        self.contact_sheet = None
        
        # Dimensions
        self.cell_size = 64
        
    def step(self):
        # Get inputs
        latent_in = self.get_blended_input('latent_in', 'first')
        reconstructed = self.get_blended_input('reconstructed_image', 'mean')
        trigger = self.get_blended_input('trigger', 'sum') or 0.0
        scan_speed = self.get_blended_input('scan_speed', 'sum')
        amplitude_signal = self.get_blended_input('amplitude', 'sum')
        
        if amplitude_signal is not None:
            amplitude = amplitude_signal * 5.0
        else:
            amplitude = self.scan_amplitude
            
        if scan_speed is not None:
            self.frames_per_scan = max(5, int(30 / (scan_speed + 0.1)))
        
        # Store base latent
        if latent_in is not None and self.base_latent is None:
            self.base_latent = latent_in.copy()
            self.current_latent = latent_in.copy()
            
        # Trigger scan
        if trigger > 0.5 and not self.is_scanning:
            self.start_scan()
            
        # Scanning logic
        if self.is_scanning and self.base_latent is not None:
            self.frame_counter += 1
            
            # Capture reconstructed image
            if reconstructed is not None and self.frame_counter > 5:  # Wait a few frames for stabilization
                key = (self.scan_index, self.scan_direction)
                if key not in self.captured_images:
                    # Resize and store
                    img_resized = cv2.resize(reconstructed, (self.cell_size, self.cell_size))
                    self.captured_images[key] = img_resized.copy()
                    
            # Time to move to next scan?
            if self.frame_counter >= self.frames_per_scan:
                self.advance_scan()
                
            # Generate current modified latent
            self.current_latent = self.base_latent.copy()
            if self.scan_index < len(self.base_latent):
                self.current_latent[self.scan_index] += amplitude * self.scan_direction
                
        # Build contact sheet
        if len(self.captured_images) > 0:
            self.build_contact_sheet()
            
    def start_scan(self):
        """Start a new scan"""
        self.is_scanning = True
        self.scan_index = 0
        self.scan_direction = -1  # Start with negative
        self.frame_counter = 0
        self.captured_images = {}
        print("PC Scanner: Starting scan...")
        
    def advance_scan(self):
        """Move to next PC/direction"""
        self.frame_counter = 0
        
        if self.scan_direction == -1:
            # Switch to positive
            self.scan_direction = 1
        else:
            # Move to next PC
            self.scan_direction = -1
            self.scan_index += 1
            
            # Check if scan complete
            if self.scan_index >= len(self.base_latent):
                self.is_scanning = False
                self.current_latent = self.base_latent.copy()
                print(f"PC Scanner: Scan complete! Captured {len(self.captured_images)} images.")
                
    def build_contact_sheet(self):
        """Build the grid visualization"""
        num_pcs = len(self.base_latent) if self.base_latent is not None else 8
        
        # Calculate grid dimensions
        # Each PC gets 2 cells (- and +)
        total_cells = num_pcs * 2
        rows = (total_cells + self.grid_cols - 1) // self.grid_cols
        
        # Create canvas
        canvas_width = self.grid_cols * self.cell_size
        canvas_height = rows * self.cell_size
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
        
        # Fill grid
        cell_idx = 0
        for pc_idx in range(num_pcs):
            for direction in [-1, 1]:
                key = (pc_idx, direction)
                
                row = cell_idx // self.grid_cols
                col = cell_idx % self.grid_cols
                
                y_start = row * self.cell_size
                x_start = col * self.cell_size
                
                if key in self.captured_images:
                    img = self.captured_images[key]
                    
                    # Ensure correct shape
                    if img.ndim == 2:
                        img = np.stack([img, img, img], axis=-1)
                    elif img.shape[2] == 1:
                        img = np.repeat(img, 3, axis=2)
                        
                    canvas[y_start:y_start+self.cell_size, 
                           x_start:x_start+self.cell_size] = img
                else:
                    # Empty cell - draw placeholder
                    cv2.rectangle(canvas, (x_start, y_start), 
                                (x_start+self.cell_size-1, y_start+self.cell_size-1),
                                (0.2, 0.2, 0.2), 1)
                
                # Label
                label = f"PC{pc_idx}" + ("-" if direction == -1 else "+")
                cv2.putText(canvas, label, (x_start+2, y_start+12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (1, 1, 1), 1)
                
                # Highlight current scan position
                if self.is_scanning and pc_idx == self.scan_index and direction == self.scan_direction:
                    cv2.rectangle(canvas, (x_start, y_start),
                                (x_start+self.cell_size-1, y_start+self.cell_size-1),
                                (0, 1, 0), 2)
                
                cell_idx += 1
                
        self.contact_sheet = canvas
        
    def get_output(self, port_name):
        if port_name == 'latent_out':
            return self.current_latent
        elif port_name == 'contact_sheet':
            return self.contact_sheet
        elif port_name == 'current_pc':
            return float(self.scan_index) if self.is_scanning else -1.0
        elif port_name == 'progress':
            if self.base_latent is not None and self.is_scanning:
                total = len(self.base_latent) * 2
                current = self.scan_index * 2 + (0 if self.scan_direction == -1 else 1)
                return current / total
            return 0.0
        return None
        
    def get_display_image(self):
        if self.contact_sheet is not None:
            # Display the contact sheet
            img = (np.clip(self.contact_sheet, 0, 1) * 255).astype(np.uint8)
            h, w = img.shape[:2]
            return QtGui.QImage(img.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)
        else:
            # Show status
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            if self.is_scanning:
                status = f"Scanning PC{self.scan_index}{'-' if self.scan_direction == -1 else '+'}"
                progress = int(self.get_output('progress') * 100)
                cv2.putText(img, status, (10, 128), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0,255,0), 2)
                cv2.putText(img, f"{progress}%", (10, 160), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (255,255,255), 1)
                
                # Progress bar
                bar_width = int(256 * self.get_output('progress'))
                cv2.rectangle(img, (0, 240), (bar_width, 256), (0,255,0), -1)
            else:
                cv2.putText(img, "Ready to scan", (10, 128), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (255,255,255), 1)
                cv2.putText(img, "Send trigger signal", (10, 160), cv2.FONT_HERSHEY_SIMPLEX,
                           0.4, (200,200,200), 1)
                
            return QtGui.QImage(img.data, 256, 256, 256*3, QtGui.QImage.Format.Format_RGB888)
            
    def get_config_options(self):
        return [
            ("Scan Amplitude", "scan_amplitude", self.scan_amplitude, None),
            ("Grid Columns", "grid_cols", self.grid_cols, None)
        ]