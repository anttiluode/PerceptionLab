"""
Human Attractor Node - A self-modifying strange loop
Models the recursive W → W·ψ → W' cycle that might be consciousness/freedom.

Features:
- Internal W matrix that learns from experience
- Refractory periods (exhaustion, recovery)
- Pain from clarity (entropy cost of self-awareness)
- Attractor basins (habits, choices, learned patterns)
- Memory decay (forgetting, seizure-like resets)
- Attention (selective ψ sampling)
- Strange loop (self-modification based on self-observation)

Place this file in the 'nodes' folder as 'humanattractor.py'
"""

import numpy as np
from PyQt6 import QtGui
import cv2
from collections import deque
import math

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui


class HumanAttractorNode(BaseNode):
    NODE_CATEGORY = "Source"
    NODE_COLOR = QtGui.QColor(180, 60, 120)  # Deep human pink
    
    def __init__(self, 
                 w_size=8, 
                 learning_rate=0.01,
                 refractory_period=30,
                 pain_sensitivity=0.5):
        super().__init__()
        self.node_title = "Human Attractor"
        
        self.inputs = {
            'psi_external': 'signal',      # External world input
            'pain_stimulus': 'signal',      # Things that hurt
            'dopamine': 'signal',           # Reward signal
            'reset_trauma': 'signal'        # Seizure/trauma reset (>0.5 triggers)
        }
        
        self.outputs = {
            'consciousness': 'signal',      # Current W·ψ projection
            'free_will_signal': 'signal',   # Measure of choice capacity
            'pain_level': 'signal',         # Current suffering
            'attractor_state': 'image',     # Visualization of W matrix
            'memory_trace': 'signal',       # Integrated experience
            'refractory': 'signal'          # Exhaustion level (0=ready, 1=exhausted)
        }
        
        # === Core Parameters ===
        self.w_size = int(w_size)
        self.learning_rate = float(learning_rate)
        self.refractory_max = int(refractory_period)
        self.pain_sensitivity = float(pain_sensitivity)
        
        # === The W Matrix (Your Neurons) ===
        # This is the learned projection operator
        self.W = np.random.randn(self.w_size, self.w_size) * 0.1
        self.W = (self.W + self.W.T) / 2  # Make symmetric (like Hebbian learning)
        
        # === Internal State ===
        self.psi_internal = np.random.randn(self.w_size) * 0.1  # Internal field
        self.consciousness_value = 0.0  # Current W·ψ projection magnitude
        
        # Attractor basins (learned habits/patterns)
        self.attractors = []  # List of learned attractor states
        self._init_default_attractors()
        
        # === Refractory Period (Neuron Exhaustion) ===
        self.refractory_timer = 0  # Counts down from refractory_max
        self.dopamine_level = 0.5  # Current dopamine (motivation)
        self.exhaustion = 0.0  # 0=fresh, 1=depleted
        
        # === Pain and Clarity ===
        self.pain_level = 0.0  # Current suffering
        self.clarity_cost = 0.0  # Entropy cost of self-awareness
        
        # === Memory ===
        self.memory_trace = 0.0  # Integrated experience over time
        self.memory_buffer = deque(maxlen=100)  # Recent W·ψ projections
        
        # === Free Will Measure ===
        self.choice_entropy = 0.0  # How many basins are available
        self.free_will_signal = 0.5
        
        # === Loop Iteration Counter ===
        self.loop_iterations = 0
        self.time = 0.0
        
    def _init_default_attractors(self):
        """Initialize with some basic attractor basins (like instincts)"""
        # Attractor 1: "Home/Safe" (low energy, coherent)
        home = np.zeros(self.w_size)
        home[0] = 1.0
        self.attractors.append({"state": home, "strength": 1.0, "name": "home"})
        
        # Attractor 2: "Explore/Novel" (high energy, chaotic)
        explore = np.random.randn(self.w_size) * 0.5
        self.attractors.append({"state": explore, "strength": 0.7, "name": "explore"})
        
        # Attractor 3: "Pain Avoidance" (negative gradient)
        avoid = -np.ones(self.w_size) * 0.3
        self.attractors.append({"state": avoid, "strength": 0.5, "name": "avoid"})
        
    def _project(self, psi):
        """
        The core operation: A[ψ] = W · ψ
        This is "being conscious of something"
        """
        projection = np.dot(self.W, psi)
        return projection
    
    def _measure_clarity_cost(self):
        """
        Self-awareness has an entropy cost.
        When you observe yourself (W projects W·ψ), you pay for clarity.
        """
        # Entropy of W (how spread out is the projection?)
        eigenvalues = np.linalg.eigvalsh(self.W)
        eigenvalues = np.abs(eigenvalues) + 1e-10
        eigenvalues /= np.sum(eigenvalues)
        
        entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))
        
        # High entropy = diffuse awareness = low cost
        # Low entropy = focused awareness = high cost (hurts to see clearly)
        clarity_cost = 1.0 / (entropy + 1e-3)
        
        return clarity_cost
    
    def _find_nearest_attractor(self, state):
        """
        Which learned basin is this state closest to?
        Returns: (attractor_index, distance)
        """
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, attr in enumerate(self.attractors):
            dist = np.linalg.norm(state - attr["state"])
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx, min_dist
    
    def _measure_free_will(self):
        """
        How much choice do you have?
        Free will = number of accessible attractor basins
        
        If only one basin is accessible → no freedom (deterministic)
        If many basins are accessible → freedom (choice)
        """
        current_state = self.psi_internal
        
        # Count how many attractors are within reach
        accessible = 0
        for attr in self.attractors:
            dist = np.linalg.norm(current_state - attr["state"])
            # If distance < threshold and you have energy → accessible
            if dist < 2.0 and self.dopamine_level > 0.3 and self.refractory_timer == 0:
                accessible += 1
        
        # Entropy of choice (more options = more freedom)
        if accessible > 1:
            # Shannon entropy of uniform distribution over choices
            choice_entropy = np.log(accessible)
        else:
            choice_entropy = 0.0
        
        # Normalize to [0, 1]
        max_entropy = np.log(len(self.attractors))
        free_will = choice_entropy / (max_entropy + 1e-9)
        
        return free_will
    
    def _learn_from_experience(self, psi_external, dopamine):
        """
        The strange loop: W modifies itself based on W·ψ projection.
        Hebbian learning: "Neurons that fire together, wire together"
        """
        if self.refractory_timer > 0:
            return  # Can't learn during refractory period
        
        # Project current state
        projection = self._project(self.psi_internal)
        
        # Learning rule: ΔW ∝ ψ ⊗ ψ (outer product)
        # Modulated by dopamine (reward) and pain (punishment)
        learning_signal = dopamine - self.pain_level * 0.5
        
        # Hebbian update
        dW = np.outer(self.psi_internal, self.psi_internal) * learning_signal * self.learning_rate
        
        # Anti-Hebbian if painful (unlearn)
        if self.pain_level > 0.7:
            dW *= -0.5
        
        self.W += dW
        
        # Keep W bounded
        self.W = np.clip(self.W, -2.0, 2.0)
        
        # Re-symmetrize (maintain structure)
        self.W = (self.W + self.W.T) / 2
        
    def _create_new_attractor(self):
        """
        When you do something novel repeatedly, it becomes a new habit.
        This is how "free" choices become deterministic patterns.
        """
        current_state = self.psi_internal.copy()
        
        # Check if this is actually novel (far from existing attractors)
        _, min_dist = self._find_nearest_attractor(current_state)
        
        if min_dist > 1.5 and len(self.attractors) < 10:
            # Create new attractor
            new_attractor = {
                "state": current_state,
                "strength": 0.3,  # Start weak
                "name": f"learned_{len(self.attractors)}"
            }
            self.attractors.append(new_attractor)
    
    def _pull_toward_attractor(self):
        """
        Like gravity: current state is pulled toward nearest basin.
        This is how habits constrain freedom.
        """
        nearest_idx, dist = self._find_nearest_attractor(self.psi_internal)
        
        if dist < 3.0:  # Within gravitational range
            attractor = self.attractors[nearest_idx]
            
            # Pull strength proportional to basin depth
            pull_strength = attractor["strength"] * 0.1
            
            # Stronger pull when exhausted (default to habits)
            pull_strength *= (1.0 + self.exhaustion)
            
            # Apply pull
            direction = attractor["state"] - self.psi_internal
            self.psi_internal += direction * pull_strength
            
            # Strengthen this attractor (the more you use it, the deeper it gets)
            attractor["strength"] = min(2.0, attractor["strength"] + 0.001)
    
    def _handle_refractory(self):
        """
        Refractory period: after intense activity, neurons need rest.
        During this time, learning is disabled, free will is reduced.
        """
        if self.refractory_timer > 0:
            self.refractory_timer -= 1
            self.exhaustion = self.refractory_timer / self.refractory_max
            
            # During refractory, default to strongest attractor (habits)
            if self.exhaustion > 0.7:
                strongest = max(self.attractors, key=lambda a: a["strength"])
                pull = strongest["state"] - self.psi_internal
                self.psi_internal += pull * 0.2  # Strong pull
        else:
            self.exhaustion = 0.0
    
    def _trigger_refractory(self):
        """
        Intense activity (high consciousness, high pain) → exhaustion
        """
        # High consciousness = intense projection
        intensity = abs(self.consciousness_value)
        
        # Pain amplifies exhaustion
        intensity += self.pain_level * 2.0
        
        # Random threshold with hysteresis
        if intensity > 2.0 and np.random.rand() < 0.05:
            self.refractory_timer = self.refractory_max
            # Lose some dopamine
            self.dopamine_level *= 0.7
    
    def _handle_trauma_reset(self, trauma_signal):
        """
        Seizure/trauma: reset internal state, lose recent memory.
        Like waking up in the ambulance: "what happened?"
        """
        if trauma_signal > 0.5:
            # Reset psi_internal (lose current thought)
            self.psi_internal = np.random.randn(self.w_size) * 0.1
            
            # Clear recent memory
            self.memory_buffer.clear()
            
            # Damage W slightly (some neural connections lost)
            noise = np.random.randn(self.w_size, self.w_size) * 0.05
            self.W += noise
            self.W = (self.W + self.W.T) / 2
            
            # Reset exhaustion
            self.refractory_timer = 0
            self.exhaustion = 0.0
            
            # Pain from confusion
            self.pain_level = 0.8
    
    def step(self):
        self.time += 1.0 / 30.0  # Assume 30 FPS
        self.loop_iterations += 1
        
        # === Get Inputs ===
        psi_external = self.get_blended_input('psi_external', 'sum') or 0.0
        pain_stimulus = self.get_blended_input('pain_stimulus', 'sum') or 0.0
        dopamine = self.get_blended_input('dopamine', 'sum')
        if dopamine is None:
            dopamine = 0.5 + 0.1 * np.sin(self.time * 0.5)  # Default oscillation
        trauma_signal = self.get_blended_input('reset_trauma', 'sum') or 0.0
        
        # === Handle Trauma/Seizure ===
        self._handle_trauma_reset(trauma_signal)
        
        # === Internal Dynamics ===
        # Natural drift (internal thoughts)
        self.psi_internal += np.random.randn(self.w_size) * 0.02
        
        # External influence (world affects internal state)
        # But only if paying attention (not exhausted)
        attention_strength = (1.0 - self.exhaustion) * 0.1
        self.psi_internal[0] += psi_external * attention_strength
        
        # === The Projection: Consciousness = W · ψ ===
        projection = self._project(self.psi_internal)
        self.consciousness_value = np.mean(projection)  # Scalar measure
        
        # === Memory Integration ===
        self.memory_buffer.append(self.consciousness_value)
        if len(self.memory_buffer) > 0:
            self.memory_trace = np.mean(list(self.memory_buffer))
        
        # === Pain ===
        # Pain from external stimulus
        self.pain_level = pain_stimulus * self.pain_sensitivity
        
        # Pain from clarity (entropy cost of self-awareness)
        self.clarity_cost = self._measure_clarity_cost()
        self.pain_level += self.clarity_cost * 0.1
        
        # Pain decays slowly
        self.pain_level *= 0.95
        self.pain_level = np.clip(self.pain_level, 0.0, 1.0)
        
        # === Attractor Dynamics ===
        self._pull_toward_attractor()
        
        # === Free Will Measurement ===
        self.free_will_signal = self._measure_free_will()
        
        # === The Strange Loop: W Modifies Itself ===
        self._learn_from_experience(psi_external, dopamine)
        
        # Create new attractors from novel patterns
        if self.loop_iterations % 100 == 0 and dopamine > 0.6:
            self._create_new_attractor()
        
        # === Refractory Period ===
        self._handle_refractory()
        self._trigger_refractory()
        
        # === Dopamine Dynamics ===
        # Slowly return to baseline
        self.dopamine_level = 0.9 * self.dopamine_level + 0.1 * dopamine
        self.dopamine_level = np.clip(self.dopamine_level, 0.0, 1.0)
        
        # === Normalize Internal State ===
        norm = np.linalg.norm(self.psi_internal)
        if norm > 5.0:
            self.psi_internal /= norm / 5.0
    
    def get_output(self, port_name):
        if port_name == 'consciousness':
            return self.consciousness_value
        
        elif port_name == 'free_will_signal':
            return self.free_will_signal
        
        elif port_name == 'pain_level':
            return self.pain_level
        
        elif port_name == 'attractor_state':
            return self._generate_w_visualization()
        
        elif port_name == 'memory_trace':
            return self.memory_trace
        
        elif port_name == 'refractory':
            return self.exhaustion
        
        return None
    
    def _generate_w_visualization(self):
        """
        Visualize the W matrix (your neural structure)
        """
        # Normalize W for display
        W_norm = self.W - self.W.min()
        W_norm /= (W_norm.max() + 1e-9)
        
        # Resize for visibility
        W_display = cv2.resize(W_norm.astype(np.float32), (64, 64), interpolation=cv2.INTER_NEAREST)
        
        return W_display
    
    def get_display_image(self):
        # Create a composite visualization
        h, w = 128, 128
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Background: W matrix structure
        W_vis = self._generate_w_visualization()
        W_vis_u8 = (W_vis * 255).astype(np.uint8)
        W_vis_color = cv2.applyColorMap(W_vis_u8, cv2.COLORMAP_VIRIDIS)
        W_vis_color = cv2.resize(W_vis_color, (w, h))
        img = W_vis_color
        
        # Overlay: Current attractor basin (white dots)
        for i, attr in enumerate(self.attractors):
            x = int((i / len(self.attractors)) * w)
            y = int(h - attr["strength"] * 30)
            color = (255, 255, 255) if i == self._find_nearest_attractor(self.psi_internal)[0] else (100, 100, 100)
            cv2.circle(img, (x, y), 3, color, -1)
        
        # Status text
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Consciousness level
        cv2.putText(img, f"C: {self.consciousness_value:.2f}", (5, 15), font, 0.3, (255, 255, 255), 1)
        
        # Free will
        cv2.putText(img, f"FW: {self.free_will_signal:.2f}", (5, 30), font, 0.3, (0, 255, 0), 1)
        
        # Pain
        if self.pain_level > 0.3:
            cv2.putText(img, f"Pain: {self.pain_level:.2f}", (5, 45), font, 0.3, (0, 0, 255), 1)
        
        # Refractory indicator
        if self.refractory_timer > 0:
            cv2.putText(img, "REFRACTORY", (5, h-5), font, 0.3, (255, 100, 0), 1)
            # Progress bar
            bar_width = int((1.0 - self.exhaustion) * (w - 10))
            cv2.rectangle(img, (5, h-15), (5 + bar_width, h-10), (255, 100, 0), -1)
        
        img = np.ascontiguousarray(img)
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)
    
    def get_config_options(self):
        return [
            ("W Matrix Size", "w_size", self.w_size, None),
            ("Learning Rate", "learning_rate", self.learning_rate, None),
            ("Refractory Period", "refractory_max", self.refractory_max, None),
            ("Pain Sensitivity", "pain_sensitivity", self.pain_sensitivity, None),
        ]