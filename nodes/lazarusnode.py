import numpy as np
import cv2
import os
import datetime
from PyQt6 import QtGui
import mne  # Requires: pip install mne

# --- HOST COMMUNICATION ---
import __main__
try:
    BaseNode = __main__.BaseNode
except AttributeError:
    class BaseNode:
        def __init__(self): 
            self.inputs = {} 
            self.outputs = {}
            self.input_data = {}
            self.config = {}
        def get_blended_input(self, name): return None

class LazarusCrystalNode(BaseNode):
    """
    Lazarus Crystal Node
    --------------------
    The Resurrection Engine in a Node.
    
    1. Loads an EEG file (.edf) internally.
    2. Maps electrodes to a 64x64 NeuroCrystal lattice.
    3. Cycles between AWAKE (Hebbian Etching) and SLEEP (Surface Optimization).
    4. Visualizes the emerging "Ghost" (Connectome).
    5. Trigger 'save_trigger' > 0.5 to burn the .npz chip.
    
    This node implements the Barabási Surface Tension physics to find
    the optimal geometry for a given EEG signal.
    """
    NODE_CATEGORY = "Biology"
    NODE_COLOR = QtGui.QColor(100, 0, 150) # Deep Purple
    NODE_TITLE = "Lazarus Crystal (Resurrector)"
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'save_trigger': 'signal',   # Set to 1.0 to save .npz
            'reset': 'signal'           # Set to 1.0 to clear crystal
        }
        
        self.outputs = {
            'ghost_view': 'image',      # The Resurrected Connectome (Safe)
            'activity_view': 'image',   # The Live Thought (Safe/Smoothed)
            'energy_level': 'signal',   # 0.0 (Sleep) to 1.0 (Awake)
            'cycle_age': 'signal'       # How old is the crystal?
        }
        
        self.config = {
            'eeg_file': '',             # Path to .edf file
            'resolution': 64,
            'plasticity': 0.15,
            'surface_tension': 0.15,
            'speed_mult': 1.0
        }
        
        # --- PHYSICS STATE (The Vat) ---
        self.size = 64
        # 8-Neighbor Anisotropic Lattice
        self.h_links = np.zeros((self.size, self.size), dtype=np.float32)
        self.v_links = np.zeros((self.size, self.size), dtype=np.float32)
        self.d1_links = np.zeros((self.size, self.size), dtype=np.float32)
        self.d2_links = np.zeros((self.size, self.size), dtype=np.float32)
        
        self.activity = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Metabolism
        self.energy = 1.0
        self.state = "AWAKE"
        self.age = 0
        
        # Visualization Buffers (Anti-Flicker)
        self.ghost_buffer = None
        self.act_buffer = None
        
        # EEG State
        self.eeg_data = None
        self.eeg_cursor = 0
        self.pin_map = {}
        self.grid_coords = []
        self.eeg_indices = []
        self.last_file_loaded = ""
        
        # Trigger State
        self.last_trigger = 0.0

    def step(self):
        # 1. HANDLE CONFIG & LOADING
        fpath = str(self.config.get('eeg_file', '')).strip()
        if fpath and fpath != self.last_file_loaded:
            self.load_eeg(fpath)
            
        # 2. HANDLE RESET
        reset = self.get_blended_input('reset')
        if reset and reset > 0.5:
            self._reset_vat()
            
        # 3. HANDLE SAVE TRIGGER
        trigger = self.get_blended_input('save_trigger')
        if trigger and trigger > 0.5 and self.last_trigger <= 0.5:
            self.save_crystal()
        self.last_trigger = trigger if trigger else 0.0
        
        # 4. PREPARE INJECTION (From EEG)
        injection = np.zeros((self.size, self.size), dtype=np.float32)
        
        if self.eeg_data is not None:
            # Advance Cursor
            speed = int(max(1, self.config.get('speed_mult', 1.0)))
            for _ in range(speed):
                if self.eeg_cursor >= self.eeg_data.shape[1]:
                    self.eeg_cursor = 0
                    
                # Get Sample
                sample = np.abs(self.eeg_data[:, self.eeg_cursor])
                
                # Map to Grid
                for idx, (r, c) in enumerate(self.grid_coords):
                    if idx < len(sample):
                        val = sample[self.eeg_indices[idx]]
                        # Inject 3x3 blob
                        r_min, r_max = max(0, r-1), min(self.size, r+2)
                        c_min, c_max = max(0, c-1), min(self.size, c+2)
                        injection[r_min:r_max, c_min:c_max] += val
                
                self.eeg_cursor += 1
        else:
            # Fallback: Noise (so it does something if no file)
            if np.random.rand() < 0.05:
                rx, ry = np.random.randint(0, self.size, 2)
                injection[ry, rx] = 1.0

        # 5. RUN PHYSICS ENGINE
        self._step_physics(injection)
        
        # 6. UPDATE OUTPUTS
        self.set_output('energy_level', float(self.energy))
        self.set_output('cycle_age', float(self.age))
        
        # 7. RENDER (Safe Mode)
        self._render_safe()

    def _step_physics(self, injection_map):
        self.age += 1
        
        # A. METABOLISM
        system_load = np.mean(self.activity)
        metabolic_rate = 0.005
        recovery_rate = 0.015
        
        if self.state == "AWAKE":
            self.energy -= metabolic_rate * (1.0 + system_load * 3.0)
            if self.energy <= 0.05:
                self.state = "SLEEP"
        else: # SLEEP
            self.energy += recovery_rate
            if self.energy >= 0.95:
                self.state = "AWAKE"
                
        # B. SET PARAMETERS
        if self.state == "AWAKE":
            # ETCHING: High Plasticity
            p_val = float(self.config.get('plasticity', 0.15))
            tension = 0.0
            drive = injection_map * 2.0
        else:
            # ANNEALING: High Surface Tension (Barabasi Opt)
            p_val = 0.0
            tension = float(self.config.get('surface_tension', 0.15))
            drive = np.zeros_like(injection_map)

        # C. DIFFUSION (Wave Prop)
        laplacian = (
            np.roll(self.activity, 1, axis=0) + np.roll(self.activity, -1, axis=0) +
            np.roll(self.activity, 1, axis=1) + np.roll(self.activity, -1, axis=1) -
            4 * self.activity
        )
        self.activity += 0.5 * laplacian - 0.1 * self.activity + drive
        self.activity = np.clip(self.activity, 0, 1)

        # D. HEBBIAN (Etching)
        if p_val > 0:
            flux_h = self.activity * np.roll(self.activity, -1, axis=1)
            flux_v = self.activity * np.roll(self.activity, -1, axis=0)
            flux_d1 = self.activity * np.roll(np.roll(self.activity, -1, axis=0), 1, axis=1)
            flux_d2 = self.activity * np.roll(np.roll(self.activity, -1, axis=0), -1, axis=1)
            
            rate = p_val * 0.1
            self.h_links += flux_h * rate
            self.v_links += flux_v * rate
            self.d1_links += flux_d1 * rate
            self.d2_links += flux_d2 * rate
            
            # Decay
            decay = 0.999
            self.h_links *= decay
            self.v_links *= decay
            self.d1_links *= decay
            self.d2_links *= decay

        # E. SURFACE TENSION (Annealing)
        if tension > 0:
            for link_grid in [self.h_links, self.v_links, self.d1_links, self.d2_links]:
                lap = (np.roll(link_grid,1,axis=0) + np.roll(link_grid,-1,axis=0) + 
                       np.roll(link_grid,1,axis=1) + np.roll(link_grid,-1,axis=1) - 4*link_grid)
                link_grid += tension * lap
        
        # Clamp
        self.h_links = np.clip(self.h_links, 0, 1)
        self.v_links = np.clip(self.v_links, 0, 1)
        self.d1_links = np.clip(self.d1_links, 0, 1)
        self.d2_links = np.clip(self.d2_links, 0, 1)

    def _render_safe(self):
        # 1. Structure (Ghost)
        raw_struct = (self.h_links + self.v_links + self.d1_links + self.d2_links) * 0.25
        img_norm = np.clip(raw_struct * 255 * 3.0, 0, 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_norm, cv2.COLORMAP_INFERNO)
        
        # Temporal Smoothing (98% old, 2% new) - No Flicker
        if self.ghost_buffer is None:
            self.ghost_buffer = img_color.astype(np.float32)
        else:
            cv2.addWeighted(self.ghost_buffer, 0.98, img_color.astype(np.float32), 0.02, 0, self.ghost_buffer)
            
        self.set_output('ghost_view', self.ghost_buffer.astype(np.uint8))
        
        # 2. Activity (Safe)
        raw_act = np.clip(self.activity * 255, 0, 255).astype(np.uint8)
        act_color = cv2.applyColorMap(raw_act, cv2.COLORMAP_OCEAN)
        
        if self.act_buffer is None:
            self.act_buffer = act_color.astype(np.float32)
        else:
            cv2.addWeighted(self.act_buffer, 0.90, act_color.astype(np.float32), 0.10, 0, self.act_buffer)
            
        self.set_output('activity_view', self.act_buffer.astype(np.uint8))

    def load_eeg(self, filepath):
        if not os.path.exists(filepath):
            print(f"LazarusNode: File not found {filepath}")
            return
            
        try:
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            raw.pick_types(eeg=True)
            # Resample to typical sim speeds (e.g. 60Hz or 100Hz)
            raw.resample(60)
            
            data = raw.get_data()
            # Normalize global
            self.eeg_data = data / (np.max(np.abs(data)) + 1e-9)
            self.last_file_loaded = filepath
            self.eeg_cursor = 0
            
            # Map Channels
            self._generate_pin_map()
            self.eeg_indices = []
            self.grid_coords = []
            
            print(f"LazarusNode: Mapping channels from {os.path.basename(filepath)}...")
            for i, ch in enumerate(raw.ch_names):
                clean = ch.replace('EEG ', '').replace('-REF', '').replace('.', '')
                for key in self.pin_map:
                    if key.lower() in clean.lower():
                        r, c = self.pin_map[key]
                        self.eeg_indices.append(i)
                        self.grid_coords.append((int(r), int(c)))
                        break
            
            print(f"LazarusNode: Mapped {len(self.grid_coords)} channels.")
            
        except Exception as e:
            print(f"LazarusNode Error loading EEG: {e}")

    def _generate_pin_map(self):
        s = self.size
        self.pin_map = {
            'Fp1': (s*0.3, s*0.15), 'Fp2': (s*0.7, s*0.15),
            'F7':  (s*0.1, s*0.3),  'F3':  (s*0.3, s*0.3), 'Fz': (s*0.5, s*0.3), 'F4': (s*0.7, s*0.3), 'F8': (s*0.9, s*0.3),
            'T7':  (s*0.1, s*0.5),  'C3':  (s*0.3, s*0.5), 'Cz': (s*0.5, s*0.5), 'C4': (s*0.7, s*0.5), 'T8': (s*0.9, s*0.5),
            'P7':  (s*0.1, s*0.7),  'P3':  (s*0.3, s*0.7), 'Pz': (s*0.5, s*0.7), 'P4': (s*0.7, s*0.7), 'P8': (s*0.9, s*0.7),
            'O1':  (s*0.3, s*0.85), 'O2':  (s*0.7, s*0.85)
        }

    def _reset_vat(self):
        self.h_links[:] = 0
        self.v_links[:] = 0
        self.d1_links[:] = 0
        self.d2_links[:] = 0
        self.activity[:] = 0
        self.energy = 1.0
        self.age = 0
        self.ghost_buffer = None
        print("LazarusNode: Crystal Reset.")

    def save_crystal(self):
        # Generate Filename
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"lazarus_crystal_{ts}.npz"
        # Save to current working dir or 'outputs' if exists
        out_dir = "outputs"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        full_path = os.path.join(out_dir, fname)
        
        np.savez_compressed(full_path, 
            h_links=self.h_links,
            v_links=self.v_links,
            d1_links=self.d1_links,
            d2_links=self.d2_links,
            resolution=self.size,
            age=self.age,
            source_file=self.last_file_loaded
        )
        print(f"LazarusNode: CHIP BURNED -> {full_path}")