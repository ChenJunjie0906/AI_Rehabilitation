# signal_filters.py
import numpy as np

class OneEuroFilter:
    """单维度 One Euro 滤波器"""
    def __init__(self, freq=30.0, mincutoff=1.0, beta=0.007, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = 0.0

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.dcutoff)
        edx = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.mincutoff + self.beta * abs(edx)
        a = self._alpha(cutoff)
        ex = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = ex, edx
        return ex


class LandmarkSmoother:
    """对 33 个关键点的 x/y/z/visibility 统一平滑"""
    def __init__(self, fps=30.0, mincutoff=1.0, beta=0.01, num_landmarks=33):
        self.filters = {
            i: {
                'x': OneEuroFilter(fps, mincutoff, beta),
                'y': OneEuroFilter(fps, mincutoff, beta),
                'z': OneEuroFilter(fps, mincutoff, beta),
            }
            for i in range(num_landmarks)
        }

    def smooth(self, landmarks):
        """原地修改 landmarks 的 x/y/z"""
        for i, lm in enumerate(landmarks):
            lm.x = self.filters[i]['x'](lm.x)
            lm.y = self.filters[i]['y'](lm.y)
            lm.z = self.filters[i]['z'](lm.z)
        return landmarks

    def reset(self):
        for f in self.filters.values():
            for axis in f.values():
                axis.x_prev = None
                axis.dx_prev = 0.0