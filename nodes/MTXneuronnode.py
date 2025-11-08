"""
MTX Neuron Node - A realistic spiking neuron with H-S-L token emission.
Combines Izhikevich spiking, synaptic dynamics, and dendritic plateaus.
Outputs H/S/L tokens as signal pulses.

Ported from mtxneuron.py
Requires: pip install numpy
"""

import numpy as np
from PyQt6 import QtGui
import cv2
from collections import deque
import sys
import os

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

rng = np.random.default_rng(42)

# --- Core Simulation Classes (from mtxneuron.py) ---

class MtxPort:
    def __init__(self, win_ms=300.0, step_ms=0.1):
        self.win_ms = float(win_ms)
        self.step_ms = float(step_ms)
        self.spike_times = deque(maxlen=4000)
        self.voltage_buf = deque(maxlen=int(win_ms/step_ms))
        self.prev_plateau = False
        self.persist_l = 0

    def update(self, voltage, spike, plateau_active, t_ms):
        self.voltage_buf.append(voltage)
        if spike:
            self.spike_times.append(t_ms)

        if len(self.voltage_buf) < 20:
            return None, 0.0, 0.0

        W = self.win_ms
        recent = [s for s in self.spike_times if t_ms - s <= W]
        rate_hz = len(recent) / (W/1000.0)

        if len(recent) >= 4 and np.mean(np.diff(recent)) > 0:
            isis = np.diff(recent)
            cv = np.std(isis) / np.mean(isis)
            coherence = float(np.clip(1.0 - cv, 0.0, 1.0))
        else:
            coherence = 0.0

        v = np.array(self.voltage_buf)
        dv = np.abs(np.diff(v[-20:])).mean()
        novelty = float(np.clip(dv/20.0 + (rate_hz/50.0), 0.0, 1.0))

        token = None
        burst = len(recent) >= 3 and (recent[-1] - recent[-3]) <= 50.0
        plateau_onset = plateau_active and not self.prev_plateau
        if burst or plateau_onset:
            token = 'h'
            self.persist_l = 0
        elif 5.0 <= rate_hz <= 25.0 and coherence > 0.5:
            self.persist_l += 1
            if self.persist_l * self.step_ms >= 200.0:
                token = 'l'
        else:
            self.persist_l = 0

        if token is None and (novelty > 0.25 or spike):
            token = 's'

        self.prev_plateau = plateau_active
        return token, novelty, coherence

class Synapse:
    def __init__(self, syn_type='AMPA', weight=1.0):
        self.type = syn_type
        self.weight = weight
        self.g = 0.0
        self.x = 1.0
        self.u = 0.3 if syn_type == 'AMPA' else 0.1
        if syn_type == 'AMPA': self.tau, self.E_rev = 2.0, 0.0
        elif syn_type == 'NMDA': self.tau, self.E_rev = 50.0, 0.0
        elif syn_type == 'GABAA': self.tau, self.E_rev = 10.0, -70.0
        elif syn_type == 'GABAB': self.tau, self.E_rev = 100.0, -90.0

    def update(self, dt, voltage=0.0):
        self.g *= np.exp(-dt / self.tau)
        if self.type == 'NMDA':
            mg_block = 1.0 / (1.0 + 0.28 * np.exp(-0.062 * voltage))
            return self.g * mg_block
        return self.g

    def receive_spike(self):
        release = self.u * self.x
        self.x = min(1.0, self.x - release + 0.02)
        self.g += self.weight * release

class Dendrite:
    def __init__(self):
        self.voltage = -65.0
        self.calcium = 0.0
        self.plateau_active = False
        self.synapses = []

    def add_synapse(self, syn): self.synapses.append(syn)
    def update(self, dt, soma_v):
        total_I, nmda_I = 0.0, 0.0
        for syn in self.synapses:
            g = syn.update(dt, self.voltage)
            I = g * (syn.E_rev - self.voltage)
            total_I += I
            if syn.type == 'NMDA': nmda_I += I
        self.voltage += dt * (-(self.voltage - soma_v) / 10.0 + total_I / 50.0)
        ca_influx = max(0.0, nmda_I * 0.1)
        self.calcium += dt * (ca_influx - self.calcium / 20.0)
        self.plateau_active = (self.calcium > 0.25 and self.voltage > -55.0)
        return self.plateau_active

class BioNeuron:
    def __init__(self, step_ms=0.1):
        self.a, self.b, self.c, self.d = 0.02, 0.2, -65.0, 8.0
        self.v, self.u = -65.0, self.b * -65.0
        self.spike = False
        self.m_current = 0.0
        self.adaptation = 0.0
        self.atp = 1.0
        self.ampa = [Synapse('AMPA', 0.5) for _ in range(10)]
        self.nmda = [Synapse('NMDA', 0.3) for _ in range(5)]
        self.gabaa = [Synapse('GABAA', 0.7) for _ in range(3)]
        self.gabab = [Synapse('GABAB', 0.4) for _ in range(2)]
        self.dend = Dendrite()
        for s in self.nmda: self.dend.add_synapse(s)
        self.pre, self.post = 0.0, 0.0
        self.DA, self.ACh, self.NE = 0.5, 0.3, 0.2
        self.mtx = MtxPort(win_ms=300.0, step_ms=step_ms)
        self.v_history = deque(maxlen=128) # For display

    def receive_input(self, typ='AMPA'):
        syn_list = {'AMPA': self.ampa, 'NMDA': self.nmda, 'GABAA': self.gabaa, 'GABAB': self.gabab}.get(typ)
        if syn_list: rng.choice(syn_list).receive_spike()

    def _neuromods(self, novelty, coherence):
        if hasattr(self, "_last_nov"):
            if self._last_nov > 0.5 and novelty < 0.3: self.DA = min(1.0, self.DA + 0.05)
            else: self.DA *= 0.99
        self._last_nov = novelty
        self.ACh = 0.8 * (1 - coherence) + 0.2 * self.ACh
        self.NE = 0.7 * novelty + 0.3 * self.NE

    def _stdp(self, dt):
        self.pre *= np.exp(-dt/20.0); self.post *= np.exp(-dt/20.0)
        if self.spike:
            self.post += 1.0
            if self.DA > 0.4:
                for syn in (self.ampa + self.nmda):
                    if syn.x < 0.8: syn.weight = min(2.0, syn.weight + 0.001 * self.pre * self.DA)

    def step(self, dt, t_ms, ext_I=0.0):
        plateau = self.dend.update(dt, self.v)
        I_syn = 0.0
        for s in self.ampa: I_syn += s.update(dt, self.v) * (s.E_rev - self.v)
        nmda_I = 0.0
        for s in self.nmda:
            g = s.update(dt, self.v); I = g * (s.E_rev - self.v)
            I_syn += 0.3 * I; nmda_I += I
        for s in self.gabaa + self.gabab: I_syn += s.update(dt, self.v) * (s.E_rev - self.v)
        
        self.m_current += dt * ((self.v + 35.0)/10.0 - self.m_current) / 100.0
        I_adapt = -5.0 * self.m_current
        noise_gain = 1.0 + 2.0 * self.ACh; gain = 1.0 + 1.5 * self.NE
        I_total = I_syn + I_adapt + ext_I * gain + rng.normal(0.0, 2.0*noise_gain)

        if abs(I_total) > 10: self.atp -= 0.001
        self.atp = min(1.0, self.atp + 0.0005)
        if self.atp < 0.5: I_total *= 0.7

        self.spike = False
        if self.v >= 30.0:
            self.spike = True; self.v = self.c; self.u += self.d; self.adaptation += 0.2
        else:
            dv = 0.04*self.v**2 + 5*self.v + 140 - self.u + I_total
            du = self.a*(self.b*self.v - self.u)
            self.v += dt * dv; self.u += dt * du
        
        self.adaptation *= np.exp(-dt/50.0)
        self.v -= 2.0 * self.adaptation
        self.v_history.append(self.v)
        
        token, novelty, coherence = self.mtx.update(self.v, self.spike, plateau, t_ms)
        self._neuromods(novelty, coherence)
        self._stdp(dt)
        return token, novelty, coherence, plateau

# --- The Main Node Class ---

class MTXNeuronNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(220, 120, 40) # Neural orange
    
    def __init__(self, step_ms=1.0, steps_per_frame=10):
        super().__init__()
        self.node_title = "BioNeuron (MTX)"
        
        # H=Hub/Burst, S=State/Novelty, L=Loop/Rhythm
        self.outputs = {
            'H_out': 'signal',
            'S_out': 'signal',
            'L_out': 'signal',
            'voltage': 'signal',
            'novelty': 'signal',
            'coherence': 'signal'
        }
        
        self.dt = float(step_ms)
        self.steps_per_frame = int(steps_per_frame)
        self.neuron = BioNeuron(step_ms=self.dt)
        self.time_ms = 0.0
        
        # Internal state for pulses
        self.h_pulse = 0.0
        self.s_pulse = 0.0
        self.l_pulse = 0.0
        self.novelty = 0.0
        self.coherence = 0.0

    def step(self):
        # Reset pulses
        self.h_pulse, self.s_pulse, self.l_pulse = 0.0, 0.0, 0.0
        
        for _ in range(self.steps_per_frame):
            self.time_ms += self.dt
            
            # --- Internal Stimulation (from mtxneuron.py) ---
            ext_I = 0.0
            if rng.random() < 0.05: self.neuron.receive_input('AMPA')
            if rng.random() < 0.02: self.neuron.receive_input('NMDA')
            if rng.random() < 0.03: self.neuron.receive_input('GABAA')
            if rng.random() < 0.002: # Plateau trigger
                for _ in range(6): self.neuron.receive_input('NMDA')
            # ------------------------------------------------
            
            token, nov, coh, plat = self.neuron.step(self.dt, self.time_ms, ext_I)
            
            if token == 'h': self.h_pulse = 1.0
            if token == 's': self.s_pulse = 1.0
            if token == 'l': self.l_pulse = 1.0
            
            self.novelty = nov
            self.coherence = coh

    def get_output(self, port_name):
        if port_name == 'H_out': return self.h_pulse
        if port_name == 'S_out': return self.s_pulse
        if port_name == 'L_out': return self.l_pulse
        if port_name == 'voltage': return (self.neuron.v + 65.0) / 95.0 # Normalize
        if port_name == 'novelty': return self.novelty
        if port_name == 'coherence': return self.coherence
        return None
        
    def get_display_image(self):
        w, h = 128, 64
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw voltage trace
        v_hist = np.array(list(self.neuron.v_history))
        if len(v_hist) > 1:
            v_norm = (v_hist - v_hist.min()) / (v_hist.max() - v_hist.min() + 1e-9)
            v_scaled = (v_norm * (h - 10) + 5).astype(int)
            
            for i in range(len(v_scaled) - 1):
                x1 = int(i / len(v_scaled) * w)
                x2 = int((i + 1) / len(v_scaled) * w)
                y1 = h - v_scaled[i]
                y2 = h - v_scaled[i+1]
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        
        # Draw token indicators
        if self.h_pulse: cv2.circle(img, (w-10, 10), 5, (0, 0, 255), -1) # H = Red
        if self.s_pulse: cv2.circle(img, (w-10, 25), 5, (0, 255, 0), -1) # S = Green
        if self.l_pulse: cv2.circle(img, (w-10, 40), 5, (255, 0, 0), -1) # L = Blue
            
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Time Step (ms)", "dt", self.dt, None),
            ("Steps / Frame", "steps_per_frame", self.steps_per_frame, None),
        ]