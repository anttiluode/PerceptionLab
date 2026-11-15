"""Regression: PerceptionLab graph-loader size must change the actual wave.

Run: python -m unittest discover -s tests -p test_wavefield_size_load.py -v
Requires numpy and opencv-python-headless. No Qt display required.
"""
import __main__
import importlib.util
import pathlib
import unittest

import numpy as np

class DummyColor:
    def __init__(self,*args): pass

class DummyQtGui:
    QColor = DummyColor

class DummyBaseNode:
    def __init__(self):
        self.input_data={}
    def pre_step(self):
        self.input_data={name: [] for name in self.inputs}
    def set_input(self,port,value,port_type="signal",coupling=1.0):
        if port in self.input_data:
            self.input_data[port].append(float(value)*coupling)
    def get_blended_input(self,port,mode="sum"):
        values=self.input_data.get(port,[])
        return sum(values) if values else None

__main__.BaseNode=DummyBaseNode
__main__.QtGui=DummyQtGui

root=pathlib.Path(__file__).resolve().parents[1]
path=root/"nodes"/"wavefieldnode.py"
spec=importlib.util.spec_from_file_location("wavefield_for_size_test",path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Wave=module.WaveFieldNode

class SavedGraphResize(unittest.TestCase):
    def test_host_style_config_reallocates_arrays_and_stamps(self):
        wave=Wave()                      # what the host instantiates
        self.assertEqual(wave.field.shape,(256,256))
        config={"size":96,"speed":0.5,"damping":0.006,
                "read_aperture":6,"line_y":0.5,"first_x":0.2,
                "last_x":0.92,"source_x":0.08}
        wave.set_config_options(config) # what the loader does
        # The host's fallback property setting must be a no-op for size.
        for key,value in config.items():
            if hasattr(wave,key) and getattr(wave,key)!=value:
                setattr(wave,key,value)
        self.assertEqual(wave.size,96)
        self.assertEqual(wave.field.shape,(96,96))
        self.assertEqual(wave.vel.shape,(96,96))
        self.assertEqual(wave.display_image.shape,(96,96,3))
        self.assertTrue(all(0<=x<96 and 0<=y<96
                            for x,y in wave.aperture_xy))
        self.assertAlmostEqual(wave.spacing_px,96*(0.92-0.20)/7)
        self.assertEqual(len(wave.write_stamps),8)
        wave.pre_step()
        wave.set_input("write_0",1.0)
        wave.step()
        self.assertGreater(wave.energy_value,0.0)
        self.assertEqual(wave.get_output("image").shape,(96,96,3))

    def test_resizing_resets_old_wave_state(self):
        wave=Wave(size=96)
        wave.pre_step();wave.set_input("write_0",1.0);wave.step()
        self.assertGreater(wave.energy_value,0)
        wave.set_config_options({"size":128})
        self.assertEqual(wave.field.shape,(128,128))
        self.assertAlmostEqual(float(np.max(np.abs(wave.field))),0)
        self.assertEqual(wave.energy_value,0)
        self.assertTrue(np.all(wave.read_values==0))
        wave.pre_step();wave.set_input("write_0",1.0);wave.step()
        self.assertGreater(wave.energy_value,0)

    def test_same_size_config_does_not_erase_wave(self):
        wave=Wave(size=96)
        wave.pre_step();wave.set_input("write_0",1.0);wave.step()
        before=wave.field.copy()
        wave.set_config_options({"size":96,"damping":0.035})
        self.assertTrue(np.array_equal(before,wave.field))

if __name__=="__main__":
    unittest.main()
