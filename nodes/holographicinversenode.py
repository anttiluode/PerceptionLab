import numpy as np
import cv2

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode: 
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui

class HolographicInverseNode(BaseNode):
    """
    The Holographic Reconstructor (Inverse Scattering).
    
    Attempts to recover the "Ghost Image" stored in a resonance field
    by interacting the Phase (Wave) with the Scars (Hologram).
    
    Logic: Image ≈ Phase_Angle * Scar_Density
    """
    NODE_CATEGORY = "Analysis"
    NODE_TITLE = "Holographic Inverse"
    NODE_COLOR = QtGui.QColor(200, 200, 255) # Ghostly White/Blue
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'structure_in': 'image',    # The Complex Field (Real+Imag)
            'scars_in': 'image',        # The Transfer Function (Memory)
            'focus': 'signal',          # Gamma/Contrast control
            'phase_shift': 'signal'     # Rotate phase to find the image
        }
        
        self.outputs = {
            'reconstructed_image': 'image',
            'phase_map': 'image'
        }
        
        self.last_recon = None

    def step(self):
        # 1. Get Inputs (Raw Data)
        # Note: The host passes the raw numpy array, even if it's complex.
        structure = self.get_blended_input('structure_in', 'first')
        scars = self.get_blended_input('scars_in', 'first')
        focus = self.get_blended_input('focus', 'sum')
        shift = self.get_blended_input('phase_shift', 'sum') or 0.0
        
        if structure is None or scars is None:
            return

        # 2. Extract Phase (The Wavefront)
        if np.iscomplexobj(structure):
            # Rotate phase if requested (Scanning through the hologram)
            structure_shifted = structure * np.exp(1j * shift * np.pi * 2)
            phase = np.angle(structure_shifted)
        else:
            # Fallback if magnitude was passed (Lossy, but tries)
            phase = structure # Treat brightness as phase proxy?
            
        # Normalize Phase to 0.0 - 1.0
        # Map -pi..pi to 0..1
        phase_norm = (phase + np.pi) / (2 * np.pi)
        
        # 3. The Reconstruction (Interference)
        # We modulate the Wavefront (Phase) by the Medium Density (Scars)
        recon = phase_norm * scars
        
        # 4. Optical Focus (Contrast Enhancement)
        # Helps pull the weak ghost signal out of the background
        gamma = 1.0
        if focus is not None:
            gamma = 0.5 + (focus * 2.0) # Range 0.5 to 2.5
            
        if gamma != 1.0 and gamma > 0:
            recon = np.power(recon, gamma)
            
        # Normalize
        if recon.max() > 0:
            recon /= recon.max()
            
        self.last_recon = recon

    def get_output(self, port_name):
        if port_name == 'reconstructed_image':
            return self.last_recon
        elif port_name == 'phase_map':
            # Just output the phase for debugging
            return self.last_recon # Placeholder
        return None

    def get_display_image(self):
        if self.last_recon is None: return None
        
        # Visualize
        img = (np.clip(self.last_recon, 0, 1) * 255).astype(np.uint8)
        
        # Use BONE colormap (X-Ray style) as it looks best for "Ghosts"
        color_img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
        
        # Add Label
        cv2.putText(color_img, "RECONSTRUCTION", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return QtGui.QImage(color_img.data, color_img.shape[1], color_img.shape[0], 
                           color_img.shape[1]*3, QtGui.QImage.Format.Format_RGB888)