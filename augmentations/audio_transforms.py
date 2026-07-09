import numpy as np

class AudioTransform:

    def add_noise(self, waveform):
        noise = np.random.randn(len(waveform))
        return waveform + 0.005 * noise

    def random_gain(self, waveform):
        gain = np.random.uniform(0.8, 1.2)
        return waveform * gain

    def __call__(self, waveform):

        if np.random.rand() > 0.5:
            waveform = self.add_noise(waveform)

        if np.random.rand() > 0.5:
            waveform = self.random_gain(waveform)

        return waveform
