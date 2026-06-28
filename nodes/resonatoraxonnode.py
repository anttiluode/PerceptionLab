"""
ResonatorAxonNode (v18) — The Directional Excitable Axon
---------------------------------------------------------
The directional half of the v18 neuron, as a PerceptionLab node. When its
`fire_in` crosses threshold, it ORIGINATES a spike at its input (AIS) end and
the spike PROPAGATES down a FitzHugh-Nagumo excitable line toward the output
end. The spike travels one way — the inactivation wake behind it blocks
reversal (v14). An excitability gradient (dendrite-side hard to excite) makes
back-propagation attenuate, the Leterrier AIS mechanism.

  outputs:
    spike_out : the field value at the OUTPUT end (transient — a moving pulse)
    arrived   : 1.0 once a spike has reached the output end this run (latched
                by the SynapseLatchNode downstream, NOT here — here it is the
                raw transient, honestly)
    field     : the whole membrane line as a spectrum (for the FieldView node)

This is the v14/v15 mechanism: origination + inactivation + (optional)
gradient. Feed `fire_in` from a ResonatorSomaNode. Read `spike_out` into a
SynapseLatchNode to capture the transient into a held bit (v18's key finding:
a propagating spike is transient, logic needs a capture — the synapse).

PerceptionLab / Antti Luode, with Claude. Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import numpy as np
import cv2

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class ResonatorAxonNode(BaseNode):
    NODE_CATEGORY = "Resonator"
    NODE_COLOR = QtGui.QColor(210, 110, 90)  # axon orange-red

    def __init__(self, n_cells=20, kick=12.0, gradient=1.0):
        super().__init__()
        self.node_title = "Resonator Axon"

        self.inputs = {
            'fire_in':  'signal',   # from a ResonatorSomaNode 'fire'
            'trigger':  'signal',   # paced origination (e.g. Septum 'peak') -
                                    # originates ONE spike per rising edge,
                                    # then lets it conduct undisturbed
            'reset':    'signal',   # >0.5 hard-clears the line to rest
        }
        self.outputs = {
            'spike_out': 'signal',   # transient field value at the output end
            'arrived':   'signal',   # raw threshold cross at output (transient)
            'field':     'spectrum', # the whole membrane line (for FieldView)
        }

        # FitzHugh-Nagumo params (same as the repo's propagating_spike.py)
        self.n = int(n_cells)
        self.D, self.eps, self.a, self.b, self.dt = 0.5, 0.08, 0.7, 0.8, 0.3
        self.kick = float(kick)
        self.gradient = float(gradient)   # 1.0 = AIS gradient on; 0.0 = uniform
        self.ais_cell = 3
        self.out_cell = self.n - 3

        self.v = np.full(self.n, -1.2)
        self.w = np.full(self.n, -0.6)
        self._kick_left = 0
        self._fired_latch = False
        self._prev_trig = 0.0
        self._armed = True
        self.plot_img = np.zeros((90, 220, 3), np.uint8)

    def _a_profile(self):
        a = np.full(self.n, self.a)
        if self.gradient > 0.5:
            a[:self.ais_cell] = 1.3   # dendrite side: hard to excite (attenuates back-prop)
        return a

    def _laplacian(self):
        x = self.v
        lap = np.zeros_like(x)
        lap[1:-1] = x[2:] + x[:-2] - 2 * x[1:-1]
        lap[0] = x[1] - x[0]
        lap[-1] = x[-2] - x[-1]
        return lap

    def step(self):
        # 'reset' HARD-clears the line to rest. Use sparingly: a theta peak
        # should NOT drive this directly, because if theta is faster than the
        # spike's traversal time it will wipe the spike before it arrives
        # (the v16 lesson: the clock period must exceed the conduction
        # latency). Leave 'reset' for an explicit "start a new computation".
        reset = self.get_blended_input('reset', 'sum')
        if reset is not None and float(reset) > 0.5:
            self.v[:] = -1.2
            self.w[:] = -0.6
            self._fired_latch = False
            self._kick_left = 0
            self._armed = True            # ready to originate again

        fire = self.get_blended_input('fire_in', 'sum')
        fire = 0.0 if fire is None else float(fire)

        # 'trigger' ORIGINATES one spike (rising edge), without wiping the
        # line. This is the correct pacing input: a theta peak triggers a
        # fresh spike, which then conducts UNDISTURBED to the output. The
        # neuron fires only if its soma says fire (fire_in > 0.5) AND it is
        # armed (hasn't already fired this cycle).
        trig_raw = self.get_blended_input('trigger', 'sum')
        trig = 0.0 if trig_raw is None else float(trig_raw)
        trig_edge = (trig > 0.5 and self._prev_trig <= 0.5)
        self._prev_trig = trig

        if trig_edge and fire > 0.5 and self._armed:
            self._kick_left = 6           # paced origination: one spike per trigger
            self._armed = False
        elif trig < 0.5:
            self._armed = True            # re-arm between triggers

        # free-running mode: if no trigger wired, originate on a fire edge
        if trig_raw is None:
            if fire > 0.5 and not self._fired_latch:
                self._kick_left = 6
                self._fired_latch = True
            if fire < 0.5:
                self._fired_latch = False

        I = np.zeros(self.n)
        if self._kick_left > 0:
            I[self.ais_cell] = self.kick
            self._kick_left -= 1

        a_prof = self._a_profile()
        lap = self._laplacian()
        dv = self.v - self.v**3 / 3 - self.w + self.D * lap + I
        dw = self.eps * (self.v + a_prof - self.b * self.w)
        self.v = np.clip(self.v + self.dt * dv, -3.0, 3.0)
        self.w = self.w + self.dt * dw
        self._render()

    def _render(self):
        h, w = 90, 220
        img = np.zeros((h, w, 3), np.uint8)
        # draw the membrane line v(x) as a heat strip
        vv = np.clip((self.v + 1.3) / 3.3, 0, 1)   # -1.3..2.0 -> 0..1
        for i in range(self.n):
            x0 = int(i / self.n * w)
            x1 = int((i + 1) / self.n * w)
            c = int(vv[i] * 255)
            col = (c // 2, c, 255 - c // 2)
            cv2.rectangle(img, (x0, 30), (x1, h), col, -1)
        # mark input (cyan) and output (lime) ends
        cv2.line(img, (int(self.ais_cell/self.n*w), 30),
                 (int(self.ais_cell/self.n*w), h), (255, 255, 0), 1)
        cv2.line(img, (int(self.out_cell/self.n*w), 30),
                 (int(self.out_cell/self.n*w), h), (0, 255, 0), 1)
        spk = self.v[self.out_cell]
        cv2.putText(img, f"out v={spk:+.2f}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 255, 0) if spk > 0.5 else (180, 180, 180), 1)
        self.plot_img = img

    def get_output(self, port_name):
        if port_name == 'spike_out':
            return float(self.v[self.out_cell])
        if port_name == 'arrived':
            return 1.0 if self.v[self.out_cell] > 0.5 else 0.0
        if port_name == 'field':
            return self.v.astype(np.float32)
        return None

    def get_display_image(self):
        return QtGui.QImage(self.plot_img.data, 220, 90, 220*3,
                            QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Cells", "n", self.n, None),
            ("Kick strength", "kick", self.kick, "float"),
            ("AIS gradient (1=on,0=off)", "gradient", self.gradient, "float"),
        ]

    def set_config_options(self, options):
        if "n" in options:
            self.n = int(options["n"])
            self.v = np.full(self.n, -1.2)
            self.w = np.full(self.n, -0.6)
            self.out_cell = self.n - 3
        if "kick" in options:
            self.kick = float(options["kick"])
        if "gradient" in options:
            self.gradient = float(options["gradient"])
