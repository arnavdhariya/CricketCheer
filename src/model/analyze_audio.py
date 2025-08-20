import numpy as np
import librosa
from pathlib import Path

def analyze_audio(audio_path: str):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis = 1)
    return mfcc_mean
