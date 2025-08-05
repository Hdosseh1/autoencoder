import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa as lb
import os
import glob

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
# from tensorflow.keras import layers, losses
# from tensorflow.keras.datasets import fashion_mnist
# from tensorflow.keras.models import Model

AUDIO_DIR = "/Volumes/aid_data/buowset/audio"
file_pattern = os.path.join(AUDIO_DIR, "**", "*.wav")
audio_path = glob.glob(file_pattern, recursive = True)

if not audio_path:
    raise RuntimeError(f"No .wav files found in {AUDIO_DIR}")
path0 = audio_path[0]
y, sr = lb.load(path0,sr = 22050)

print(f"Loaded{path0!r}: {y.shape[0]} samples at {sr} Hz")
chunk_dur = 1.0
def audio_slice (y, chunk_dur, overlap = 0.0, pad = True):
    chunk_size = int(chunk_dur * sr)
    step = int((chunk_dur - overlap) * sr)
    chunks = []
    for i in range(0,len(y), step):
        end = i + chunk_size
        chunk = y[i:end]
        if len(chunk) < chunk_size:
            if pad:
                chunk = np.pad(chunk(0,chunk_size - len(chunk)), 'constant')
        elif len(chunk) < chunk_size * 0.5:
            break
        chunks.append(chunk)
    return np.array(chunks)

chunks = audio_slice(y, chunk_dur=1.0)
print(f"Created {len(chunks)} chunks of shape {chunks.shape[0]}")