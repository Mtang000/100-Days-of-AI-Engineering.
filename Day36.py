import torch
import torch.nn as nn


raw_audio = torch.randn(1, 16000)
print(f"Raw Audio Shape: {raw_audio.shape} -> (1 Batch, 16,000 samples)")
# Feeding 16,000 numbers per second into a Transformer would crash it instantly.


class WaveformToSpectrogram(nn.Module):
    def __init__(self, n_fft=400, hop_length=160, n_mels=80):
        super().__init__()

        self.spectrogram_converter = nn.Conv1d(
            in_channels=1,

            out_channels=n_mels,
            kernel_size=n_fft,
            stride=hop_length
        )

    def forward(self, x):

        x = x.unsqueeze(1)

        spectrogram = self.spectrogram_converter(x)

        spectrogram = torch.abs(spectrogram)
        return spectrogram


audio_processor = WaveformToSpectrogram()


print("Converting 1D sound wave into a 2D frequency image...")

spectrogram = audio_processor(raw_audio)

print(f"\nFinal Spectrogram Shape: {spectrogram.shape}")
print("-> (1 Batch, 80 Frequency Bins (Height), 98 Time Steps (Width))")
