"""
Token QKV Attention Node (Biological Transformer)
=================================================
Performs Dot-Product Attention on Neural Tokens.

PIPELINE:
1. Inputs sparse tokens from WaveletEngine (Frontal=Q, Temporal=K, Occipital=V).
2. Embeds them into 64-dim dense vectors (based on their KeyID).
3. Computes the Attention Matrix (Interference Pattern).
4. Outputs the weighted Context Vector.

INPUTS:
- query_tokens: (Frontal) The "Seeker"
- key_tokens: (Temporal) The "Map"
- value_tokens: (Sensory) The "Payload"
- temperature: Sharpness of the attention mechanism

OUTPUTS:
- attention_map: Image (The Matrix)
- context_vector: Spectrum (The Resulting Thought)
- display: Dashboard
"""

import numpy as np
import cv2

# --- COMPATIBILITY ---
import __main__
try:
    BaseNode = __main__.BaseNode
    QtGui = __main__.QtGui
except AttributeError:
    from PyQt6 import QtGui
    class BaseNode:
        def __init__(self): self.inputs = {}; self.outputs = {}
        def get_blended_input(self, name, mode): return 0.0

class TokenQKVNode(BaseNode):
    NODE_CATEGORY = "Synthesis"
    NODE_TITLE = "Token QKV Attention"
    NODE_COLOR = QtGui.QColor(180, 50, 255) # Transformer Purple
    
    def __init__(self):
        super().__init__()
        
        self.inputs = {
            "query_tokens": "spectrum", # From Frontal
            "key_tokens": "spectrum",   # From Temporal
            "value_tokens": "spectrum", # From Occipital/Parietal
            "temperature": "float",
        }
        
        self.outputs = {
            "display": "image",
            "attention_map": "image",
            "context_vector": "spectrum" # The 64-dim Result
        }
        
        # --- EMBEDDING STATE ---
        # We need a stable vector space for our 20 possible tokens
        # (4 regions * 5 bands = 20 keys)
        self.vocab_size = 20 
        self.embed_dim = 64
        
        # Fixed random embeddings (Orthogonal-ish basis)
        np.random.seed(42)
        self.embedding_matrix = np.random.randn(self.vocab_size, self.embed_dim)
        # Normalize
        self.embedding_matrix /= np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)
        
        self._display = np.zeros((600, 800, 3), dtype=np.uint8)

    def _sanitize(self, data):
        """Ensure input is (N, 3) array, robust against strings/None"""
        # 1. Null check
        if data is None: 
            return np.zeros((0, 3), dtype=np.float32)
            
        # 2. String check (THE FIX)
        if isinstance(data, str):
            return np.zeros((0, 3), dtype=np.float32)
            
        # 3. List conversion
        if isinstance(data, (list, tuple)): 
            try:
                data = np.array(data)
            except:
                return np.zeros((0, 3), dtype=np.float32)
        
        # 4. Attribute Check (Prevent .ndim crash)
        if not hasattr(data, 'ndim'):
            return np.zeros((0, 3), dtype=np.float32)

        # 5. Dimension Fixes
        if data.ndim == 1:
            if len(data) == 3: 
                return data.reshape(1, 3)
            else:
                return np.zeros((0, 3), dtype=np.float32)
                
        if data.ndim != 2 or data.shape[1] < 3: 
            return np.zeros((0, 3), dtype=np.float32)
            
        return data

    def _tokens_to_dense(self, tokens):
        """
        Convert sparse [Key, Amp, Phase] list into a Dense Vector Sum.
        Result shape: (64,)
        """
        dense_accum = np.zeros(self.embed_dim, dtype=np.float32)
        
        if len(tokens) == 0:
            return dense_accum
            
        for t in tokens:
            key_id = int(t[0]) % self.vocab_size
            amp = t[1]
            phase = t[2] # Unused for simple QKV, but could rotate vector
            
            # Get base vector
            base_vec = self.embedding_matrix[key_id]
            
            # Add to accumulator (weighted by amplitude)
            dense_accum += base_vec * amp
            
        return dense_accum

    def step(self):
        # 1. Gather Inputs
        raw_q = self.inputs.get("query_tokens", None)
        raw_k = self.inputs.get("key_tokens", None)
        raw_v = self.inputs.get("value_tokens", None)
        temp_val = self.inputs.get("temperature", 1.0)
        
        # Safety
        temp = 1.0
        if isinstance(temp_val, (int, float)): temp = temp_val
        elif hasattr(temp_val, 'item'): temp = temp_val.item()
        if temp < 0.1: temp = 0.1

        # 2. Sanitize Data (Robust)
        q_toks = self._sanitize(raw_q)
        k_toks = self._sanitize(raw_k)
        v_toks = self._sanitize(raw_v)
        
        # 3. Embedding (Sparse -> Dense)
        def stack_vectors(tokens):
            vecs = []
            amps = []
            if len(tokens) == 0: return np.zeros((1, self.embed_dim)), [0]
            for t in tokens:
                key_id = int(t[0]) % self.vocab_size
                vec = self.embedding_matrix[key_id]
                vecs.append(vec)
                amps.append(t[1])
            return np.array(vecs), amps

        Q_mat, Q_amps = stack_vectors(q_toks) # (N_q, 64)
        K_mat, K_amps = stack_vectors(k_toks) # (N_k, 64)
        V_mat, V_amps = stack_vectors(v_toks) # (N_v, 64)
        
        # Apply amplitudes
        Q_mat = Q_mat * np.array(Q_amps)[:, None]
        K_mat = K_mat * np.array(K_amps)[:, None]
        V_mat = V_mat * np.array(V_amps)[:, None]
        
        # 4. Attention Mechanism: Softmax(Q * K.T / sqrt(d))
        # Result shape: (N_q, N_k)
        
        scale = np.sqrt(self.embed_dim)
        scores = np.matmul(Q_mat, K_mat.T) / scale
        
        # Softmax (row-wise)
        scores = scores / temp
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True)) # Stability
        attn_weights = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-9)
        
        # 5. Context Output: Weights * V
        # Hack for mismatched dimensions: Use global max attention
        global_match = np.mean(np.max(attn_weights, axis=1)) # Best match for each query
        context_vec = np.sum(V_mat, axis=0) * global_match
        
        # 6. Outputs
        self.outputs['context_vector'] = context_vec.astype(np.float32)
        
        # Visual Matrix (Resize for display)
        attn_vis = cv2.resize(attn_weights, (256, 256), interpolation=cv2.INTER_NEAREST)
        attn_vis = (attn_vis * 255).astype(np.uint8)
        self.outputs['attention_map'] = cv2.applyColorMap(attn_vis, cv2.COLORMAP_VIRIDIS)
        
        # 7. Render
        self._render_dashboard(Q_amps, K_amps, V_amps, attn_weights)

    def _render_dashboard(self, q_a, k_a, v_a, attn):
        img = self._display
        img[:] = (20, 20, 30)
        h, w = img.shape[:2]
        
        # Draw Matrices
        
        # Top Left: Q (Rows)
        cv2.putText(img, "QUERY (Frontal)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        start_y = 50
        for amp in q_a:
            val = int(min(amp, 2.0) * 100)
            cv2.rectangle(img, (20, start_y), (60, start_y + 10), (255, 100, 100), -1)
            cv2.rectangle(img, (20, start_y), (20 + val, start_y + 10), (255, 200, 200), -1)
            start_y += 15
            if start_y > 150: break
            
        # Top Middle: K (Cols)
        cv2.putText(img, "KEY (Temporal)", (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
        start_x = 150
        for amp in k_a:
            val = int(min(amp, 2.0) * 100)
            cv2.rectangle(img, (start_x, 50), (start_x + 10, 90), (100, 255, 100), -1)
            cv2.rectangle(img, (start_x, 50 + (40-val//3)), (start_x + 10, 90), (200, 255, 200), -1)
            start_x += 15
            if start_x > 400: break

        # Center: Attention Matrix
        mat_size = 250
        mat_img = cv2.resize(attn, (mat_size, mat_size), interpolation=cv2.INTER_NEAREST)
        mat_img = (mat_img * 255).astype(np.uint8)
        mat_col = cv2.applyColorMap(mat_img, cv2.COLORMAP_INFERNO)
        
        img[120:120+mat_size, 150:150+mat_size] = mat_col
        cv2.rectangle(img, (150, 120), (150+mat_size, 120+mat_size), (100, 100, 100), 1)
        
        # Right: Value Gating
        cv2.putText(img, "OUTPUT (Gated)", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        
        # Simple viz of context vector energy
        ctx = self.outputs['context_vector']
        ctx_norm = np.linalg.norm(ctx)
        
        # Draw "Neuron" activity based on result
        cv2.circle(img, (500, 150), int(10 + ctx_norm*10), (255, 150, 50), -1)
        cv2.putText(img, f"Focus: {ctx_norm:.2f}", (450, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        self._display = img

    def get_display_image(self): return self._display
    def get_output(self, name): return self.outputs.get(name)