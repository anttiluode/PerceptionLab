"""Temporal transit prediction from two synchronous PerceptionLab scalar taps.

Use near[t-delay] to predict far[t], versus a same-time near[t] baseline.
Both linear fits use strictly *past* pairs; the present far[t] is held out.
This is a read-only instrument, not training of the physical medium.
2026-11-15, PerceptionLabAI, MIT.
"""
import __main__
import math
from collections import deque

BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


def _fit(xs, ys):
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mx = sum(xs)/n
    my = sum(ys)/n
    den = sum((x-mx)**2 for x in xs)
    if den < 1e-14:
        return None
    slope = sum((x-mx)*(y-my) for x,y in zip(xs,ys))/den
    return slope, my - slope*mx


class TemporalTransitPredictorNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(60, 130, 160)

    def __init__(self, delay=13, window=200, min_samples=50):
        super().__init__()
        self.node_title = "Temporal Transit Predictor"
        self.inputs = {"near":"signal", "far":"signal"}
        self.outputs = {k:"signal" for k in
                        ("prediction","innovation","mse_delayed",
                         "mse_simultaneous","advantage_db","valid")}
        self.delay = max(0,int(delay))
        self.window = max(16,int(window))
        self.min_samples = max(8,int(min_samples))
        self._reset()

    def _reset(self):
        self._near = deque(maxlen=self.window+self.delay+2)
        self._far = deque(maxlen=self.window+self.delay+2)
        self._errors_d = deque(maxlen=self.window)
        self._errors_0 = deque(maxlen=self.window)
        self.prediction = self.innovation = 0.0
        self.mse_delayed = self.mse_simultaneous = 0.0
        self.advantage_db = self.valid = 0.0

    def step(self):
        a = self.get_blended_input("near", "sum")
        b = self.get_blended_input("far", "sum")
        if a is None or b is None:
            self.valid = 0.0
            return
        a,b = float(a),float(b)
        if not math.isfinite(a) or not math.isfinite(b):
            self.valid = 0.0
            return

        # Data from previous ticks only; train a new fit before appending b.
        xs,ys = list(self._near),list(self._far)
        n = min(self.window,len(ys),len(xs)-self.delay)
        self.valid = 0.0
        if n >= self.min_samples:
            lagged = xs[-n-self.delay:-self.delay] if self.delay else xs[-n:]
            aligned = xs[-n:]
            target = ys[-n:]
            fd,f0 = _fit(lagged,target),_fit(aligned,target)
            if fd is not None and f0 is not None:
                past_a = xs[-self.delay] if self.delay else a
                pred_d = fd[0]*past_a+fd[1]
                pred_0 = f0[0]*a+f0[1]
                self.prediction = pred_d
                self.innovation = b-pred_d
                self._errors_d.append((b-pred_d)**2)
                self._errors_0.append((b-pred_0)**2)
                self.mse_delayed = sum(self._errors_d)/len(self._errors_d)
                self.mse_simultaneous = sum(self._errors_0)/len(self._errors_0)
                self.advantage_db = 10*math.log10(
                    (self.mse_simultaneous+1e-16)/(self.mse_delayed+1e-16))
                self.valid = 1.0
        self._near.append(a)
        self._far.append(b)

    def get_output(self, port_name):
        return float(getattr(self,port_name)) if port_name in self.outputs else None

    def get_config_options(self):
        return [("Delay frames","delay",self.delay,None),
                ("Training window","window",self.window,None),
                ("Min samples","min_samples",self.min_samples,None)]

    def set_config_options(self, options):
        changed=False
        for k,lo in (("delay",0),("window",16),("min_samples",8)):
            if k in options:
                v=max(lo,int(float(options[k])))
                changed |= getattr(self,k)!=v
                setattr(self,k,v)
        if changed:
            self._reset()
