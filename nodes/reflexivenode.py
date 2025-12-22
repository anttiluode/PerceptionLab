"""
ReflexiveFieldNode - The Self-Observing Field
==============================================

"The bell that hears itself ring"

This node implements the minimal architecture for self-reference:
- It predicts its own next state
- It observes what actually happens  
- The prediction error modulates its own parameters
- The loop closes: the system's model of itself affects itself

This is where "free will" would hide if it existed - not as a ghost,
but as the causal efficacy of self-modeling. The system's prediction
of what it will do becomes part of what determines what it does.

Key insight from Yi Ma's parsimony principle: self-consistent systems
naturally find low-dimensional attractors. The reflexive loop doesn't
add complexity - it *reduces* it by forcing the system to be predictable
to itself.

From Pinotsis & Miller: the ephaptic field is a "control parameter" that
evolves slower than neural activity. Here, the SELF-MODEL is an even
slower control parameter that modulates the field parameters themselves.

Three timescales (separation crucial for consciousness per Haken):
1. Spikes (fastest) - neural firing
2. Field (intermediate) - ephaptic dynamics  
3. Self-model (slowest) - reflexive prediction

INPUTS:
- field_state: Current ephaptic/thought field from upstream
- external_spectrum: Optional external drive (EEG modes etc)
- awareness: How much to weight self-observation vs external

OUTPUTS:
- reflexive_field: The self-modulated field
- prediction_error: How wrong the self-model was
- self_model: The system's model of itself (latent)
- parameter_drift: How parameters are being self-modulated
- integration: Measure of self-consistency (phi-like)

Created: December 2025
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, fftshift
from collections import deque

import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    class BaseNode:
        def get_blended_input(self, name, mode): return None
    import PyQt6.QtGui as QtGui


class ReflexiveFieldNode(BaseNode):
    """
    Self-observing field that models and modulates itself.
    The minimal architecture for self-reference.
    """
    NODE_CATEGORY = "Consciousness"
    NODE_TITLE = "Reflexive Field"
    NODE_COLOR = QtGui.QColor(255, 215, 0)  # Gold - the observer
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            'field_state': 'image',           # From EphapticFieldNode
            'external_spectrum': 'spectrum',   # Optional external drive
            'awareness': 'signal',             # Self vs external weighting
            'freeze_model': 'signal',          # Pause self-model updates
            'reset': 'signal'
        }
        
        self.outputs = {
            # Fields
            'reflexive_field': 'image',        # The self-modulated output
            'prediction_field': 'image',       # What we predicted
            'error_field': 'image',            # Where we were wrong
            'self_model': 'image',             # The model itself
            
            # Signals
            'prediction_error': 'signal',      # Scalar error (free energy)
            'integration': 'signal',           # Self-consistency measure
            'parameter_drift': 'signal',       # How much self-modulation
            'complexity': 'signal',            # Model complexity
            'autonomy': 'signal',              # Self vs external driven
            
            # For downstream
            'model_spectrum': 'spectrum'       # Latent self-model
        }
        
        self.size = 64  # Smaller for speed - this is meta-level
        
        # === THE SELF-MODEL ===
        # A compressed representation of "what I expect myself to do"
        self.model_dim = 16  # Latent dimensionality of self-model
        
        # Encoder: field -> latent (what am I doing?)
        # Simple linear projection for parsimony
        self.encoder_weights = np.random.randn(self.model_dim, self.size * self.size) * 0.01
        
        # Predictor: latent_t -> latent_t+1 (what will I do next?)
        # Linear dynamics in latent space
        self.predictor_weights = np.eye(self.model_dim) * 0.95 + np.random.randn(self.model_dim, self.model_dim) * 0.02
        
        # Decoder: latent -> field (reconstruct prediction)
        self.decoder_weights = np.random.randn(self.size * self.size, self.model_dim) * 0.01
        
        # === STATE ===
        self.current_latent = np.zeros(self.model_dim, dtype=np.float32)
        self.predicted_latent = np.zeros(self.model_dim, dtype=np.float32)
        self.current_field = np.zeros((self.size, self.size), dtype=np.float32)
        self.predicted_field = np.zeros((self.size, self.size), dtype=np.float32)
        self.error_field = np.zeros((self.size, self.size), dtype=np.float32)
        
        # === SELF-MODULATION PARAMETERS ===
        # These get adjusted by prediction error
        self.field_gain = 1.0
        self.field_smooth = 0.5
        self.prediction_weight = 0.3
        
        # Learning rate for self-model
        self.model_learning_rate = 0.001
        
        # Learning rate for self-modulation (slower!)
        self.modulation_rate = 0.0001
        
        # === HISTORY ===
        self.error_history = deque(maxlen=100)
        self.latent_history = deque(maxlen=50)
        self.param_history = deque(maxlen=100)
        
        # === METRICS ===
        self.total_prediction_error = 0.0
        self.integration_measure = 0.0
        self.autonomy_measure = 0.0
        
        self.t = 0
    
    def encode(self, field):
        """Compress field to latent representation"""
        flat = field.flatten()
        if len(flat) != self.size * self.size:
            flat = cv2.resize(field, (self.size, self.size)).flatten()
        return np.tanh(self.encoder_weights @ flat)
    
    def predict(self, latent):
        """Predict next latent state"""
        return np.tanh(self.predictor_weights @ latent)
    
    def decode(self, latent):
        """Reconstruct field from latent"""
        flat = self.decoder_weights @ latent
        return flat.reshape(self.size, self.size)
    
    def step(self):
        self.t += 1
        
        # === GET INPUTS ===
        field_in = self.get_blended_input('field_state', 'first')
        external = self.get_blended_input('external_spectrum', 'mean')
        awareness = self.get_blended_input('awareness', 'sum')
        freeze = self.get_blended_input('freeze_model', 'sum')
        reset = self.get_blended_input('reset', 'sum')
        
        if reset is not None and reset > 0:
            self._reset()
            return
        
        # Default awareness: balanced self/external
        if awareness is None:
            awareness = 0.5
        awareness = np.clip(awareness, 0, 1)
        
        is_frozen = freeze is not None and freeze > 0
        
        # === PROCESS INPUT FIELD ===
        if field_in is not None:
            if field_in.dtype == np.uint8:
                field = field_in.astype(np.float32) / 255.0
            else:
                field = field_in.astype(np.float32)
            
            if field.shape[0] != self.size or field.shape[1] != self.size:
                field = cv2.resize(field, (self.size, self.size))
            
            if field.ndim == 3:
                field = np.mean(field, axis=2)
        else:
            field = np.zeros((self.size, self.size), dtype=np.float32)
        
        # === THE REFLEXIVE LOOP ===
        
        # 1. Encode current observation
        observed_latent = self.encode(field)
        
        # 2. Compare to what we predicted
        latent_error = observed_latent - self.predicted_latent
        self.total_prediction_error = float(np.mean(latent_error**2))
        
        # 3. Decode prediction error to field space (for visualization)
        self.error_field = self.decode(latent_error)
        
        # 4. Update self-model (learn from error)
        if not is_frozen:
            # Gradient descent on prediction error
            # Update predictor to reduce error
            grad = np.outer(latent_error, self.current_latent)
            self.predictor_weights += self.model_learning_rate * grad
            
            # Regularize toward identity (parsimony)
            self.predictor_weights = 0.999 * self.predictor_weights + 0.001 * np.eye(self.model_dim)
        
        # 5. Self-modulate parameters based on error
        if not is_frozen:
            # High error -> increase smoothing (stabilize)
            # Low error -> can afford more gain (amplify)
            error_signal = np.tanh(self.total_prediction_error * 10)
            
            self.field_smooth += self.modulation_rate * (error_signal - self.field_smooth)
            self.field_gain += self.modulation_rate * (0.5 - error_signal - (self.field_gain - 1.0))
            
            # Clamp parameters
            self.field_smooth = np.clip(self.field_smooth, 0.1, 2.0)
            self.field_gain = np.clip(self.field_gain, 0.5, 2.0)
        
        # 6. Generate reflexive output
        # Blend observed with predicted based on confidence
        confidence = np.exp(-self.total_prediction_error * 5)
        
        # Self-modulated field
        blended_latent = confidence * self.predicted_latent + (1 - confidence) * observed_latent
        self.current_field = self.decode(blended_latent)
        
        # Apply self-modulated parameters
        self.current_field = gaussian_filter(self.current_field, sigma=self.field_smooth)
        self.current_field = self.current_field * self.field_gain
        
        # Add external drive if present (weighted by 1-awareness)
        if external is not None and len(external) > 0:
            ext_contribution = np.mean(external) * (1 - awareness)
            self.current_field += ext_contribution * 0.1
            self.autonomy_measure = awareness
        else:
            self.autonomy_measure = 1.0
        
        # 7. Predict next state
        self.current_latent = observed_latent
        self.predicted_latent = self.predict(self.current_latent)
        self.predicted_field = self.decode(self.predicted_latent)
        
        # === COMPUTE METRICS ===
        
        # Integration: how self-consistent is the model?
        # High if latent dynamics are smooth and predictable
        self.latent_history.append(self.current_latent.copy())
        if len(self.latent_history) > 10:
            recent = np.array(list(self.latent_history)[-10:])
            variance = np.var(recent, axis=0).mean()
            self.integration_measure = 1.0 / (1.0 + variance * 10)
        
        # Track error history
        self.error_history.append(self.total_prediction_error)
        
        # Track parameter drift
        self.param_history.append([self.field_gain, self.field_smooth])
    
    def _reset(self):
        """Reset all state"""
        self.current_latent.fill(0)
        self.predicted_latent.fill(0)
        self.current_field.fill(0)
        self.predicted_field.fill(0)
        self.error_field.fill(0)
        self.field_gain = 1.0
        self.field_smooth = 0.5
        self.error_history.clear()
        self.latent_history.clear()
        self.param_history.clear()
    
    def get_output(self, port_name):
        if port_name == 'reflexive_field':
            return (np.clip(self.current_field, 0, 1) * 255).astype(np.uint8)
        
        elif port_name == 'prediction_field':
            pred_norm = self.predicted_field / (np.abs(self.predicted_field).max() + 1e-10)
            return ((pred_norm + 1) / 2 * 255).astype(np.uint8)
        
        elif port_name == 'error_field':
            err_norm = self.error_field / (np.abs(self.error_field).max() + 1e-10)
            return ((err_norm + 1) / 2 * 255).astype(np.uint8)
        
        elif port_name == 'self_model':
            # Visualize the predictor weights as the "self model"
            model_img = self.predictor_weights.copy()
            model_norm = (model_img - model_img.min()) / (model_img.max() - model_img.min() + 1e-10)
            model_resized = cv2.resize(model_norm, (self.size, self.size))
            return (model_resized * 255).astype(np.uint8)
        
        elif port_name == 'prediction_error':
            return float(self.total_prediction_error)
        
        elif port_name == 'integration':
            return float(self.integration_measure)
        
        elif port_name == 'parameter_drift':
            if len(self.param_history) > 10:
                recent = np.array(list(self.param_history)[-10:])
                return float(np.std(recent))
            return 0.0
        
        elif port_name == 'complexity':
            # Effective dimensionality of self-model
            # (rank of predictor weights)
            s = np.linalg.svd(self.predictor_weights, compute_uv=False)
            s_norm = s / (s.sum() + 1e-10)
            entropy = -np.sum(s_norm * np.log(s_norm + 1e-10))
            return float(entropy / np.log(self.model_dim))
        
        elif port_name == 'autonomy':
            return float(self.autonomy_measure)
        
        elif port_name == 'model_spectrum':
            return self.current_latent.astype(np.float32)
        
        return None
    
    def get_display_image(self):
        h, w = self.size, self.size
        display = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        
        # Top-left: Current field (what we are)
        field_norm = np.clip(self.current_field, 0, 1)
        field_img = (field_norm * 255).astype(np.uint8)
        display[:h, :w] = cv2.applyColorMap(field_img, cv2.COLORMAP_VIRIDIS)
        
        # Top-right: Predicted field (what we expected)
        pred_norm = self.predicted_field / (np.abs(self.predicted_field).max() + 1e-10)
        pred_img = ((pred_norm + 1) / 2 * 255).astype(np.uint8)
        display[:h, w:] = cv2.applyColorMap(pred_img, cv2.COLORMAP_PLASMA)
        
        # Bottom-left: Error field (where we were wrong)
        err_norm = self.error_field / (np.abs(self.error_field).max() + 1e-10)
        err_img = ((err_norm + 1) / 2 * 255).astype(np.uint8)
        display[h:, :w] = cv2.applyColorMap(err_img, cv2.COLORMAP_HOT)
        
        # Bottom-right: Self-model (the predictor)
        model_vis = self.predictor_weights.copy()
        model_norm = (model_vis - model_vis.min()) / (model_vis.max() - model_vis.min() + 1e-10)
        model_resized = cv2.resize(model_norm, (w, h))
        model_img = (model_resized * 255).astype(np.uint8)
        display[h:, w:] = cv2.applyColorMap(model_img, cv2.COLORMAP_TWILIGHT)
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, "Current", (2, 12), font, 0.35, (255,255,255), 1)
        cv2.putText(display, "Predicted", (w+2, 12), font, 0.35, (255,255,255), 1)
        cv2.putText(display, "Error", (2, h+12), font, 0.35, (255,255,255), 1)
        cv2.putText(display, "Self-Model", (w+2, h+12), font, 0.35, (0,255,255), 1)
        
        # Stats
        err = self.total_prediction_error
        integ = self.integration_measure
        auto = self.autonomy_measure
        gain = self.field_gain
        
        stats = f"Err:{err:.3f} Int:{integ:.2f} Auto:{auto:.2f} Gain:{gain:.2f}"
        cv2.putText(display, stats, (2, h*2-5), font, 0.28, (255,255,255), 1)
        
        return QtGui.QImage(display.data, display.shape[1], display.shape[0],
                           display.shape[1] * 3, QtGui.QImage.Format.Format_RGB888)
    
    def get_config_options(self):
        return [
            ("Model Dimensions", "model_dim", self.model_dim, None),
            ("Model Learning Rate", "model_learning_rate", self.model_learning_rate, None),
            ("Modulation Rate", "modulation_rate", self.modulation_rate, None),
            ("Initial Field Gain", "field_gain", self.field_gain, None),
            ("Initial Field Smooth", "field_smooth", self.field_smooth, None),
            ("Prediction Weight", "prediction_weight", self.prediction_weight, None),
        ]
    
    def save_custom_state(self, folder_path, node_id):
        """Save learned self-model"""
        import os
        filename = f"reflexive_model_{node_id}.npz"
        filepath = os.path.join(folder_path, filename)
        np.savez(filepath,
                 encoder=self.encoder_weights,
                 predictor=self.predictor_weights,
                 decoder=self.decoder_weights,
                 field_gain=self.field_gain,
                 field_smooth=self.field_smooth)
        return filename
    
    def load_custom_state(self, filepath):
        """Load learned self-model"""
        try:
            data = np.load(filepath)
            self.encoder_weights = data['encoder']
            self.predictor_weights = data['predictor']
            self.decoder_weights = data['decoder']
            self.field_gain = float(data['field_gain'])
            self.field_smooth = float(data['field_smooth'])
        except Exception as e:
            print(f"[ReflexiveField] Failed to load: {e}")