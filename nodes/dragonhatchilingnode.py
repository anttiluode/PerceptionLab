# dragon_hatchling_fixed.py
# PerceptionLab-compatible Dragon Hatchling node
#
# Key fix vs Gemini version:
# - PerceptionLab BaseNode has input_data but does NOT have output_data.
# - Outputs must be returned via get_output(port_name) (host pulls outputs).
# - Use get_blended_input(...) to read ports (supports multiple edges).

import numpy as np
import cv2
from PyQt6 import QtGui
import __main__

# --- BaseNode injection (PerceptionLab sets this in __main__) ---
try:
    BaseNode = __main__.BaseNode
except AttributeError:
    class BaseNode:
        def __init__(self):
            self.inputs = {}
            self.outputs = {}
            self.input_data = {}
            self.node_title = "Base Node"
        def pre_step(self):
            self.input_data = {name: [] for name in self.inputs}
        def set_input(self, port_name, value, port_type='signal', coupling=1.0):
            if port_name not in self.input_data: return
            if port_type == 'signal':
                if isinstance(value, (list, np.ndarray)):
                    value = value[0] if len(value) else 0.0
                try:
                    value = float(value)
                except Exception:
                    value = 0.0
                self.input_data[port_name].append(value * coupling)
            else:
                self.input_data[port_name].append(value)
        def get_blended_input(self, port_name, blend_mode='sum'):
            vals = self.input_data.get(port_name, [])
            if not vals: return None
            if blend_mode == 'sum' and isinstance(vals[0], (int, float)):
                return float(np.sum(vals))
            if blend_mode == 'mean' and isinstance(vals[0], np.ndarray):
                return np.mean(vals, axis=0)
            return vals[0]
        def step(self): ...
        def get_output(self, port_name): return None
        def get_display_image(self): return None


def _resample_1d(x: np.ndarray, n: int) -> np.ndarray:
    """Fast 1D resample (no cv2 needed)."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if x.size == n:
        return x
    if x.size < 2:
        return np.zeros(n, dtype=np.float32)
    src = np.linspace(0.0, 1.0, x.size, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.interp(dst, src, x).astype(np.float32)


class DragonHatchlingNode(BaseNode):
    """
    Dragon Hatchling Node (Bio-Plausible Learner)
    ---------------------------------------------
    Scale-free recurrent net + simple Hebbian plasticity.
    PerceptionLab compatible: outputs are served via get_output().
    """
    NODE_CATEGORY = "Biology"
    NODE_COLOR = QtGui.QColor(200, 50, 50)  # Dragon Red

    def __init__(self):
        super().__init__()
        self.node_title = "Dragon Hatchling"

        self.inputs = {
            "stimulus_in": "spectrum",
            "learning_rate": "signal",
        }
        self.outputs = {
            "network_view": "image",
            "concept_lock": "signal",
        }

        self.config = {
            "neurons": 128,
            "sparsity": 0.10,
            "plasticity": 0.05,
            "decay": 0.01,
            "stim_gain": 0.50,
            "recur_gain": 0.10,
            "activity_decay": 0.90,
            "edge_thresh": 0.20,
            "render_size": 256,
            "inhib_gain": 0.20,          # global inhibition strength
            "max_total_weight": 2.0,     # per-neuron incoming weight budget
            "prune_thresh": 0.002,       # prune tiny synapses to keep sparsity
            "topk_incoming": 0,          # keep only top-k incoming weights per neuron (0=off)
            "oja": 1.0,                  # 1.0=use Oja stabilization term, 0.0=plain Hebb
        }

        # Output caches (host pulls via get_output)
        self._network_view = None
        self._concept_lock = 0.0

        self._init_network()

    # --- PerceptionLab-friendly input helper ---
    def get_input(self, name: str, blend_mode: str | None = None):
        """
        Wrapper that uses BaseNode.get_blended_input if available.
        Default: mean for arrays, sum for scalars.
        """
        if not hasattr(self, "get_blended_input"):
            vals = getattr(self, "input_data", {}).get(name, [])
            return vals[0] if vals else None

        vals = getattr(self, "input_data", {}).get(name, [])
        if not vals:
            return None

        if blend_mode is None:
            blend_mode = "mean" if isinstance(vals[0], np.ndarray) else "sum"

        return self.get_blended_input(name, blend_mode)

    def _init_network(self):
        N = int(self.config["neurons"])

        # Scale-free-ish topology: pick some hubs, boost their in-degree probability
        self.weights = np.zeros((N, N), dtype=np.float32)
        hubs = np.random.choice(N, size=max(1, int(N * 0.1)), replace=False)

        base_p = float(self.config["sparsity"])
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                p = base_p * (3.0 if j in hubs else 1.0)
                if np.random.rand() < p:
                    self.weights[i, j] = np.random.rand() * 0.1

        self.activity = np.zeros(N, dtype=np.float32)

        # Layout for visualization
        self.positions = np.zeros((N, 2), dtype=np.float32)
        for i in range(N):
            angle = (i / N) * 2 * np.pi
            radius = 0.8 if i not in hubs else np.random.rand() * 0.5
            self.positions[i] = [np.cos(angle) * radius, np.sin(angle) * radius]

    def step(self):
        # --- Inputs ---
        stim = self.get_input("stimulus_in")  # spectrum (array) → mean blend by default
        lr_in = self.get_input("learning_rate", "sum")  # signal (scalar)

        lr_scale = float(lr_in) if lr_in is not None else 1.0
        plasticity = float(self.config.get("plasticity", 0.05)) * max(0.0, min(5.0, lr_scale))

        # --- Activity update ---
        inhib_gain = float(self.config.get("inhib_gain", 0.20))

        # Start from previous activity (decay if no stim)
        if stim is None:
            self.activity *= float(self.config.get("activity_decay", 0.90))
        else:
            stim_vec = _resample_1d(np.asarray(stim, dtype=np.float32), int(self.config.get("neurons", 128)))
            self.activity += stim_vec * float(self.config.get("stim_gain", 0.50))

        recurrent_drive = self.weights @ self.activity

        # Integrate excitation + global inhibition (pre-nonlinearity)
        pre = self.activity + recurrent_drive * float(self.config.get("recur_gain", 0.10))

        # Global inhibitory current proportional to mean positive activation
        inh = inhib_gain * float(np.mean(np.maximum(pre, 0.0)))
        pre = pre - inh

        self.activity = np.tanh(pre).astype(np.float32)

        # --- Hebbian learning (gas) + Oja stabilization (brake) ---
        if float(np.max(np.abs(self.activity))) > 0.1:
            co = np.outer(self.activity, self.activity).astype(np.float32)  # a_i a_j
            oja = float(self.config.get("oja", 1.0))
            if oja > 0.0:
                # Oja term: - a_i^2 * w_ij (prevents runaway growth per postsynaptic neuron i)
                post_sq = (self.activity ** 2).astype(np.float32).reshape(-1, 1)
                delta = co - oja * (post_sq * self.weights)
            else:
                delta = co
            self.weights += delta * plasticity

        # --- Weight decay / homeostasis (the brake pedal) ---
        self.weights *= (1.0 - float(self.config.get("decay", 0.01)))

        # 1) Hard normalization: conserve a per-neuron incoming weight budget
        max_total = float(self.config.get("max_total_weight", 2.0))
        row_sum = np.sum(self.weights, axis=1, keepdims=True) + 1e-6  # incoming to neuron i
        scale = np.minimum(1.0, max_total / row_sum).astype(np.float32)
        self.weights *= scale

        # 2) Optional top-k sparsification per neuron (keeps 'lightning bolts' crisp)
        k = int(self.config.get("topk_incoming", 0))
        if k > 0 and k < self.weights.shape[1]:
            idx = np.argpartition(self.weights, -k, axis=1)[:, :-k]
            self.weights[np.arange(self.weights.shape[0])[:, None], idx] = 0.0

        # 3) Prune tiny synapses (keeps the graph from 'fogging in')
        prune = float(self.config.get("prune_thresh", 0.002))
        if prune > 0.0:
            self.weights[self.weights < prune] = 0.0

        # No self-loops
        np.fill_diagonal(self.weights, 0.0)

        np.clip(self.weights, 0.0, 1.0, out=self.weights)

        # --- Outputs ---
        self._concept_lock = float(np.mean(np.abs(self.activity)))
        self._network_view = self._render_network()

    def _render_network(self):
        h = w = int(self.config.get("render_size", 256))
        img = np.zeros((h, w, 3), dtype=np.float32)

        screen_pos = ((self.positions + 1.0) * 0.5 * np.array([w - 1, h - 1], dtype=np.float32)).astype(np.int32)
        rows, cols = np.where(self.weights > float(self.config.get("edge_thresh", 0.20)))

        # Connections
        for i, j in zip(rows.tolist(), cols.tolist()):
            br = float(max(0.0, self.activity[i]))
            if br <= 0.05:
                continue
            pt1 = (int(screen_pos[i, 0]), int(screen_pos[i, 1]))
            pt2 = (int(screen_pos[j, 0]), int(screen_pos[j, 1]))
            weight = float(self.weights[i, j])
            color = (0.0, weight, weight * 0.5)  # BGR-ish float tuple
            cv2.line(img, pt1, pt2, color, 1)

        # Neurons
        for i in range(int(self.config["neurons"])):
            act = float(max(0.0, self.activity[i]))
            if act <= 0.05:
                continue
            pt = (int(screen_pos[i, 0]), int(screen_pos[i, 1]))
            radius = 3 if (i % 10 == 0) else 1
            if act < 0.5:
                color = (0.0, 0.0, act * 2.0)
            else:
                color = (0.0, act, act)
            cv2.circle(img, pt, radius, (float(color[0]), float(color[1]), float(color[2])), -1)

        return img

    # --- PerceptionLab output contract ---
    def get_output(self, port_name):
        if port_name == "network_view":
            return self._network_view
        if port_name == "concept_lock":
            return self._concept_lock
        return None

    def get_display_image(self):
        return self._network_view
