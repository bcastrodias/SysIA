import numpy as np
import torchaudio


wave,sr=torchaudio.load("LibriSpeech/test-clean/61/70968/61-70968-0000.flac")
print(wave.numpy())
train_audio_transforms=torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)
spec=train_audio_transforms(wave.squeeze(0))
print(spec.numpy().transpose())
print(spec)
def plot_waveform(waveform, sample_rate):
    waveform=waveform.numpy()
    num_channels, num_frames = waveform.shape