"""
Conscious Galaxy Node - Audio-reactive consciousness field with agent dynamics
Creates galaxy-like memory patterns from audio and internal agent activity
Requires: pip install torch scipy
Place this file in the 'nodes' folder
"""

import numpy as np
from PyQt6 import QtGui
import cv2
import torch
from scipy.fft import fft, fftfreq
from collections import deque

import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

try:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TORCH_AVAILABLE = True
    from scipy.fft import fft, fftfreq
    SCIPY_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    SCIPY_AVAILABLE = False
    print("Warning: ConsciousGalaxyNode requires 'torch' and 'scipy'.")


class ConsciousAgent:
    """A field processing agent with emotional resonance"""
    def __init__(self, pos, frequency_range, sensitivity):
        self.pos = np.array(pos, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.activation = 0.0
        self.frequency_range = frequency_range
        self.sensitivity = sensitivity
        self.audio_resonance = 0.0
        self.emotion_state = 0.0  # Current emotional activation


class ConsciousGalaxyNode(BaseNode):
    NODE_CATEGORY = "Transform"
    NODE_COLOR = QtGui.QColor(180, 100, 220)  # Purple consciousness
    
    def __init__(self, grid_size=96, num_agents=8):
        super().__init__()
        self.node_title = "Conscious Galaxy"
        
        self.inputs = {
            'audio_signal': 'signal',    # Audio drives emotion/activation
            'emotion_modulator': 'signal',  # External emotion control
            'awareness': 'signal'        # Awareness level (affects memory)
        }
        self.outputs = {
            'consciousness_field': 'image',  # The living field
            'memory_trace': 'image',         # Persistent memories
            'awareness_level': 'signal',     # Current awareness
            'dominant_emotion': 'signal'     # Strongest emotion
        }
        
        if not (TORCH_AVAILABLE and SCIPY_AVAILABLE):
            self.node_title = "Conscious (Missing Libs!)"
            return
            
        self.grid_size = int(grid_size)
        self.num_agents = int(num_agents)
        self.dt = 0.03
        self.time = 0.0
        
        # Field state
        self.psi = torch.zeros((self.grid_size, self.grid_size), 
                               dtype=torch.cfloat, device=DEVICE)
        self.psi_prev = torch.zeros_like(self.psi)
        self.memory = torch.zeros((self.grid_size, self.grid_size), 
                                  dtype=torch.float32, device=DEVICE)
        
        # Laplacian kernel
        self.laplace_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
            dtype=torch.float32, device=DEVICE
        ).unsqueeze(0).unsqueeze(0)
        
        # Create conscious agents
        self.agents = self._create_agents()
        
        # Audio processing
        self.audio_buffer = deque(maxlen=512)
        self.frequency_memory = deque(maxlen=50)
        
        # Emotion system
        self.emotions = {
            'joy': 0.0,
            'sadness': 0.0,
            'anger': 0.0,
            'fear': 0.0,
            'surprise': 0.0,
            'calm': 0.0
        }
        self.awareness_level = 0.12
        
        # Parameters
        self.wave_speed = 1.8
        self.field_damping = 0.05
        self.memory_persistence = 0.995
        
    def _create_agents(self):
        """Create field processing agents positioned around the space"""
        agents = []
        positions = [
            (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8),
            (0.5, 0.3), (0.3, 0.7), (0.7, 0.5), (0.5, 0.5)
        ]
        
        for i in range(min(self.num_agents, len(positions))):
            x, y = positions[i]
            agent = ConsciousAgent(
                pos=[x * self.grid_size, y * self.grid_size],
                frequency_range=(50 + i*200, 250 + i*200),
                sensitivity=0.3 + i * 0.1
            )
            agents.append(agent)
        
        return agents
    
    def _process_audio_spectrum(self, audio_signal):
        """Analyze audio and update agent activations"""
        if audio_signal is None or abs(audio_signal) < 0.01:
            # Decay activations
            for agent in self.agents:
                agent.activation *= 0.95
                agent.audio_resonance *= 0.9
            return
        
        # Add to buffer
        self.audio_buffer.append(audio_signal)
        
        if len(self.audio_buffer) < 256:
            return
        
        # FFT analysis
        recent_audio = np.array(list(self.audio_buffer)[-256:])
        spectrum = fft(recent_audio)
        freqs = fftfreq(len(recent_audio), 1.0/44100)
        power = np.abs(spectrum[:128])
        
        volume = np.sqrt(np.mean(recent_audio**2))
        
        # Store frequency memory
        self.frequency_memory.append({
            'spectrum': power[:50].copy(),
            'volume': volume
        })
        
        # Update agents based on their frequency ranges
        for agent in self.agents:
            f_min, f_max = agent.frequency_range
            freq_mask = (np.abs(freqs[:128]) >= f_min) & (np.abs(freqs[:128]) <= f_max)
            
            if np.any(freq_mask):
                emotional_power = np.mean(power[freq_mask])
                activation_strength = emotional_power * volume * 1000
                
                # Update with momentum
                agent.activation = 0.85 * agent.activation + 0.15 * activation_strength
                agent.activation = np.clip(agent.activation, 0, 2.0)
                
                # Audio resonance
                if self.frequency_memory:
                    recent_spectrum = self.frequency_memory[-1]['spectrum']
                    freq_response = np.mean(recent_spectrum) * agent.sensitivity
                    agent.audio_resonance = 0.8 * agent.audio_resonance + 0.2 * freq_response
    
    def _update_emotions(self):
        """Update emotional state based on agent activations"""
        # Map agent activations to emotions
        if len(self.agents) >= 6:
            self.emotions['joy'] = self.agents[0].activation / 2.0
            self.emotions['sadness'] = self.agents[1].activation / 2.0
            self.emotions['anger'] = self.agents[2].activation / 2.0
            self.emotions['fear'] = self.agents[3].activation / 2.0
            self.emotions['surprise'] = self.agents[4].activation / 2.0
            self.emotions['calm'] = self.agents[5].activation / 2.0
        
        # Decay emotions
        for key in self.emotions:
            self.emotions[key] *= 0.98
            self.emotions[key] = np.clip(self.emotions[key], 0, 1)
    
    def _create_agent_patterns(self):
        """Agents create field patterns based on their activation"""
        Y, X = torch.meshgrid(
            torch.arange(self.grid_size, device=DEVICE), 
            torch.arange(self.grid_size, device=DEVICE), 
            indexing='ij'
        )
        
        field_additions = torch.zeros_like(self.psi)
        
        for i, agent in enumerate(self.agents):
            if agent.activation > 0.1:
                ax, ay = agent.pos
                
                # Distance from agent
                r = torch.sqrt((X - ax)**2 + (Y - ay)**2)
                theta = torch.atan2(Y - ay, X - ax)
                
                # Different pattern types
                if i % 3 == 0:  # Expanding circles
                    pattern = agent.activation * torch.sin(3 * r * 0.1 - self.time * 5)
                    phase = self.time
                    phase_cplx = torch.cos(torch.tensor(phase, device=DEVICE)) + \
                                1j * torch.sin(torch.tensor(phase, device=DEVICE))
                    field_additions += 0.5 * pattern * phase_cplx
                    
                elif i % 3 == 1:  # Spirals
                    pattern = agent.activation * torch.sin(r * 0.1 - theta * 3 - self.time * 2)
                    phase_cplx = torch.cos(theta) + 1j * torch.sin(theta)
                    field_additions += 0.3 * pattern * phase_cplx
                    
                else:  # Ripples
                    pattern = agent.activation * torch.exp(-r / 20) * torch.sin(r * 0.3 - self.time * 4)
                    phase = self.time * 3
                    phase_cplx = torch.cos(torch.tensor(phase, device=DEVICE)) + \
                                1j * torch.sin(torch.tensor(phase, device=DEVICE))
                    field_additions += 0.4 * pattern * phase_cplx
        
        return field_additions
    
    def _laplacian(self, field):
        """Compute Laplacian"""
        real_part = torch.nn.functional.conv2d(
            field.real.unsqueeze(0).unsqueeze(0), 
            self.laplace_kernel, 
            padding=1
        ).squeeze()
        
        imag_part = torch.nn.functional.conv2d(
            field.imag.unsqueeze(0).unsqueeze(0), 
            self.laplace_kernel, 
            padding=1
        ).squeeze()
        
        return real_part + 1j * imag_part
    
    def _update_agents(self):
        """Move agents based on field gradients"""
        field_intensity = torch.abs(self.psi)**2
        field_np = field_intensity.cpu().numpy()
        
        for agent in self.agents:
            x, y = int(agent.pos[0]), int(agent.pos[1])
            x = np.clip(x, 1, self.grid_size - 2)
            y = np.clip(y, 1, self.grid_size - 2)
            
            if agent.activation > 0.2:
                # Follow field gradients
                grad_x = field_np[y, min(x+1, self.grid_size-1)] - \
                        field_np[y, max(x-1, 0)]
                grad_y = field_np[min(y+1, self.grid_size-1), x] - \
                        field_np[max(y-1, 0), x]
                
                agent.vel += np.array([grad_x, grad_y]) * 0.1 * agent.activation
                
                # Add exploration
                agent.vel += np.random.randn(2) * 0.3
            
            # Damping
            agent.vel *= 0.85
            agent.vel = np.clip(agent.vel, -3, 3)
            
            # Update position
            agent.pos += agent.vel * self.dt
            agent.pos = np.clip(agent.pos, 5, self.grid_size - 5)

    def step(self):
        if not (TORCH_AVAILABLE and SCIPY_AVAILABLE):
            return
            
        # Get inputs
        audio = self.get_blended_input('audio_signal', 'sum') or 0.0
        emotion_mod = self.get_blended_input('emotion_modulator', 'sum')
        awareness_in = self.get_blended_input('awareness', 'sum')
        
        # Update awareness
        if awareness_in is not None:
            self.awareness_level = 0.9 * self.awareness_level + 0.1 * abs(awareness_in)
        else:
            self.awareness_level = 0.9 * self.awareness_level + 0.1 * 0.12
        
        # Process audio
        self._process_audio_spectrum(audio)
        
        # Update emotions
        self._update_emotions()
        
        # Apply emotion modulator
        if emotion_mod is not None:
            for agent in self.agents:
                agent.activation *= (1.0 + emotion_mod * 0.2)
        
        # Create agent patterns
        agent_patterns = self._create_agent_patterns()
        self.psi += agent_patterns
        
        # Evolve field
        laplacian = self._laplacian(self.psi)
        psi_new = (2 * self.psi - self.psi_prev + 
                   self.dt**2 * (self.wave_speed * laplacian - 
                                 self.field_damping * self.psi))
        
        # Limit amplitude
        amp = torch.abs(psi_new)
        max_amp = 5.0
        mask = amp > max_amp
        psi_new[mask] = psi_new[mask] / amp[mask] * max_amp
        
        # Update memory with awareness modulation
        field_intensity = torch.abs(self.psi)**2
        memory_rate = self.memory_persistence + (1 - self.memory_persistence) * self.awareness_level
        self.memory = memory_rate * self.memory + (1 - memory_rate) * field_intensity
        
        # Update
        self.psi_prev = self.psi.clone()
        self.psi = psi_new
        
        # Update agents
        self._update_agents()
        
        self.time += self.dt

    def get_output(self, port_name):
        if port_name == 'consciousness_field':
            field_cpu = torch.abs(self.psi).cpu().numpy().astype(np.float32)
            max_val = field_cpu.max()
            if max_val > 1e-9:
                return field_cpu / max_val
            return field_cpu
            
        elif port_name == 'memory_trace':
            memory_cpu = self.memory.cpu().numpy().astype(np.float32)
            max_val = memory_cpu.max()
            if max_val > 1e-9:
                return memory_cpu / max_val
            return memory_cpu
            
        elif port_name == 'awareness_level':
            return float(self.awareness_level)
            
        elif port_name == 'dominant_emotion':
            if self.emotions:
                return float(max(self.emotions.values()))
            return 0.0
            
        return None
        
    def get_display_image(self):
        # Show memory trace with magma colormap
        memory_np = self.memory.cpu().numpy()
        
        max_val = memory_np.max()
        if max_val > 1e-9:
            memory_norm = memory_np / max_val
        else:
            memory_norm = memory_np
            
        img_u8 = (memory_norm * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_MAGMA)
        
        # Draw agents as dots
        for agent in self.agents:
            x, y = int(agent.pos[0]), int(agent.pos[1])
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                brightness = int(agent.activation * 127 + 128)
                color = (brightness, brightness, 255)
                cv2.circle(img_color, (x, y), 2, color, -1)
        
        img_color = np.ascontiguousarray(img_color)
        h, w = img_color.shape[:2]
        return QtGui.QImage(img_color.data, w, h, 3*w, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        return [
            ("Grid Size (NxN)", "grid_size", self.grid_size, None),
            ("Num Agents", "num_agents", self.num_agents, None),
        ]
    
    def randomize(self):
        """Reset the consciousness"""
        if TORCH_AVAILABLE:
            self.psi.zero_()
            self.psi_prev.zero_()
            self.memory.zero_()
            for agent in self.agents:
                agent.activation = 0.0
                agent.vel[:] = 0.0