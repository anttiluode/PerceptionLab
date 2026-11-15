"""Minimal scalar field-readout -> binary gate adapter for PerceptionLab.

Always threshold the *absolute* scalar signal; sign-reversing wave lobes
are detected as arrivals. Calibration is explicit; not learned by this node.
No changes to the wave medium. MIT; PerceptionLabAI experiment 002.
"""
import math
import __main__
BaseNode=__main__.BaseNode
QtGui=__main__.QtGui

class WaveArrivalGateNode(BaseNode):
    NODE_CATEGORY="Analysis"
    NODE_COLOR=QtGui.QColor(110,155,90)
    def __init__(self, threshold=0.01):
        super().__init__()
        self.node_title="Wave Arrival Gate"
        self.inputs={"signal_in":"signal"}
        self.outputs={"gate":"signal","amplitude":"signal"}
        self.threshold=float(threshold)
        self.gate=0.0
        self.amplitude=0.0
    def step(self):
        val=self.get_blended_input("signal_in","sum")
        if val is None or not math.isfinite(float(val)):
            self.gate=self.amplitude=0.0
        else:
            self.amplitude=abs(float(val))
            self.gate=1.0 if self.amplitude>=max(0.0,self.threshold) else 0.0
    def get_output(self,port_name):
        if port_name in self.outputs:return float(getattr(self,port_name))
        return None
    def get_config_options(self):
        return [("Absolute threshold","threshold",self.threshold,"float")]
    def set_config_options(self,opts):
        if "threshold" in opts:self.threshold=max(0.0,float(opts["threshold"]))
