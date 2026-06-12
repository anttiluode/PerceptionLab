"""
Associative Cortex Node (Modern Hopfield / Attention)
=====================================================
Receives a noisy query and uses Softmax attention over stored memories to denoise it.
The sharpness of the attention (Beta) is directly modulated by available ATP.
High ATP = sharp recall. Low ATP = collapsed attention (noise).
"""
import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class AssociativeCortexNode(BaseNode):
    NODE_CATEGORY = "Cognition"
    NODE_COLOR = QtGui.QColor(140, 60, 180) # Cortical Purple

    def __init__(self, max_memories=5):
        super().__init__()
        self.node_title = "Associative Cortex"
        self.inputs = {
            'query_vec': 'spectrum',
            'teach_vec': 'spectrum',
            'atp_level': 'signal'
        }
        self.outputs = {
            'recall_vec': 'spectrum',
            'recall_img': 'image',
            'confidence': 'signal',
            'demand': 'signal'
        }
        self.max_memories = int(max_memories)
        self.memories = [] # Stores ground-truth vectors
        
        self.recall_vec = None
        self.recall_img = np.zeros((20, 20, 3), dtype=np.float32)
        self.confidence = 0.0
        self.demand = 0.0
        self.display_img = np.zeros((128, 128, 3), dtype=np.uint8)

    def step(self):
        query = self.get_blended_input('query_vec', 'first')
        teach = self.get_blended_input('teach_vec', 'first')
        atp = self.get_blended_input('atp_level', 'sum')
        if atp is None: atp = 1.0

        # 1. Store novel memories (One-shot learning)
        if teach is not None and len(self.memories) < self.max_memories:
            # Simple novelty check
            is_novel = True
            for m in self.memories:
                if np.linalg.norm(m - teach) < 1e-3:
                    is_novel = False
                    break
            if is_novel:
                self.memories.append(np.array(teach, dtype=np.float32))

        if query is None or len(self.memories) == 0:
            return

        q = np.array(query, dtype=np.float32)
        M = np.stack(self.memories) # Shape: (Num_Memories, Dim)

        # 2. Modern Hopfield / Attention (Q * K^T)
        scores = np.dot(M, q)
        
        # ATP modulates the Softmax Temperature (Beta)
        # If ATP is low, Beta -> 0 (attention collapses to uniform blur)
        beta = max(0.01, (atp - 0.2) * 15.0) 
        
        # Numerically stable softmax
        exp_scores = np.exp(beta * (scores - np.max(scores)))
        weights = exp_scores / np.sum(exp_scores)

        # 3. Readout (Weights * V)
        self.recall_vec = np.dot(weights, M)
        self.confidence = float(np.max(weights))

        # Demand is proportional to confidence. 
        # Sustaining a sharp, focused attractor burns massive energy.
        # A collapsed/resting attractor burns near zero.
        self.demand = self.confidence if self.confidence > 0.5 else 0.05

        # Format output image
        side = int(np.sqrt(len(self.recall_vec)))
        img_2d = self.recall_vec.reshape((side, side))
        self.recall_img = np.stack([img_2d]*3, axis=-1)

        # UI Visualization
        disp = (np.clip(self.recall_img, 0, 1) * 255).astype(np.uint8)
        disp = cv2.resize(disp, (128, 128), interpolation=cv2.INTER_NEAREST)
        cv2.putText(disp, f"Conf: {self.confidence:.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        self.display_img = disp

    def get_output(self, port_name):
        if port_name == 'recall_vec': return self.recall_vec
        if port_name == 'recall_img': return self.recall_img
        if port_name == 'confidence': return self.confidence
        if port_name == 'demand': return self.demand
        return None

    def get_display_image(self):
        return QtGui.QImage(self.display_img.data, 128, 128, 128*3, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [("Max Memories", "max_memories", self.max_memories, None)]
    
    def set_config_options(self, options):
        if "max_memories" in options: self.max_memories = int(options["max_memories"])