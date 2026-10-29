"""Minimal auditable recurrent temporal-memory machine (not a biophysical neuron)."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LoopConfig:
    dim: int = 16
    leak: float = 0.92
    gate: float = 1.2
    geometry: str = 'signed_shift'  # signed_shift, cyclic, orthogonal, identity
    nonlinearity: str = 'tanh'  # tanh or linear
    output_normalize: bool = True


def recurrence_matrix(dim: int, geometry: str, rng: np.random.Generator) -> np.ndarray:
    if geometry == 'identity':
        return np.eye(dim)
    if geometry in ('signed_shift', 'cyclic'):
        R = np.roll(np.eye(dim), 1, axis=0)
        if geometry == 'signed_shift':
            R[0, -1] = -1.0
        return R
    if geometry == 'orthogonal':
        Q, T = np.linalg.qr(rng.normal(size=(dim, dim)))
        Q *= np.where(np.diag(T) < 0, -1.0, 1.0)
        return Q
    raise ValueError(f'unknown geometry: {geometry}')


class LoopMemory:
    """Fixed random input channels and apical projection; only readout is trained.

    s[t+1] = f(leak R s[t] + B u[t] + gate * (B u[t]) * tanh(A R s[t]))
    For gate=0 and nonlinearity='linear', this is an exactly linear reservoir.
    """
    def __init__(self, config: LoopConfig = LoopConfig(), seed: int = 99):
        if config.dim < 2 or not (0 <= config.leak <= 1.0):
            raise ValueError('dim must be >=2 and leak between 0 and 1')
        if config.nonlinearity not in ('tanh', 'linear'):
            raise ValueError('nonlinearity must be tanh or linear')
        rng = np.random.default_rng(seed)
        d = config.dim
        self.config = config
        self.R = recurrence_matrix(d, config.geometry, rng)
        self.A = rng.normal(size=(d, d)) / np.sqrt(d)
        self.B = rng.normal(size=(d, d)) / np.sqrt(d)

    def advance(self, state: np.ndarray, stimulus: np.ndarray) -> np.ndarray:
        cfg = self.config
        history = self.R @ state
        basal = self.B @ stimulus
        context = np.tanh(self.A @ history)
        drive = basal + cfg.gate * (basal * context)
        z = cfg.leak * history + drive
        return np.tanh(z) if cfg.nonlinearity == 'tanh' else z

    def encode(self, inputs: np.ndarray, noise: np.ndarray | None = None,
               erase_before: int | None = None, trace: bool = False):
        """inputs and optional noise: (timesteps, dim). erase_before clears state
        immediately before the input at that index (causal history knockout).
        """
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim != 2 or inputs.shape[1] != self.config.dim:
            raise ValueError('inputs must be (T,dim)')
        if noise is not None and np.shape(noise) != np.shape(inputs):
            raise ValueError('noise must have same shape as inputs')
        s = np.zeros(self.config.dim)
        states = [s.copy()] if trace else None
        for t, u in enumerate(inputs):
            if erase_before is not None and t == erase_before:
                s.fill(0.0)
            s = self.advance(s, u if noise is None else u + noise[t])
            if trace:
                states.append(s.copy())
        if trace:
            return np.asarray(states)
        if self.config.output_normalize:
            s = s / (np.linalg.norm(s) + 1e-12)
        return s

    def encode_batch(self, inputs: np.ndarray, noise: np.ndarray | None = None,
                     erase_before: int | None = None) -> np.ndarray:
        if inputs.ndim != 3 or inputs.shape[2] != self.config.dim:
            raise ValueError('batch must be (N,T,dim)')
        if noise is not None and noise.shape != inputs.shape:
            raise ValueError('batch noise shape mismatch')
        # Batch update, identical equations to encode(), much faster for sweeps.
        n, steps, dim = inputs.shape
        s = np.zeros((n, dim), dtype=np.float64)
        for t in range(steps):
            if erase_before is not None and t == erase_before:
                s.fill(0)
            history = s @ self.R.T
            basal = (inputs[:, t, :] if noise is None else inputs[:, t, :] + noise[:, t, :]) @ self.B.T
            drive = basal + self.config.gate * basal * np.tanh(history @ self.A.T)
            z = self.config.leak * history + drive
            s = np.tanh(z) if self.config.nonlinearity == 'tanh' else z
        if self.config.output_normalize:
            s = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-12)
        return s


def fit_linear_readout(states: np.ndarray, labels: np.ndarray, ridge: float = 1e-2) -> np.ndarray:
    """Linear ridge only: NO nonlinear feature maps or products in the decoder."""
    X = np.column_stack([states, np.ones(len(states))])
    reg = ridge * np.eye(X.shape[1]); reg[-1, -1] = 0
    return np.linalg.solve(X.T @ X + reg, X.T @ labels)


def predict_linear(states: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.where(np.column_stack([states, np.ones(len(states))]) @ weights >= 0, 1, -1)


def accuracy(states: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    return float(np.mean(predict_linear(states, weights) == labels))
