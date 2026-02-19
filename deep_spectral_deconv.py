"""
Deep Spectral Deconvolution (DSD) API
Author: Sharon Melhi
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import sparse
from scipy.sparse.linalg import spsolve

def baseline_arpls(y, lam=1e5, itermax=100):
    """Applies physics-guided Asymmetric Least Squares Baseline Correction."""
    N = len(y)
    D = sparse.eye(N, format='csc')
    D = D[1:] - D[:-1]
    D = D[1:] - D[:-1]
    H = lam * D.T * D
    w = np.ones(N)
    
    for i in range(itermax):
        W = sparse.diags(w, 0, shape=(N, N))
        Z = W + H
        z = spsolve(Z, w * y)
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn) if len(dn) > 0 else 0
        s = np.std(dn) if len(dn) > 0 else 1
        # Safe soft-thresholding
        exponent = 2 * (d - (2*s - m)) / (s + 1e-9)
        exponent = np.clip(exponent, -500, 500) # Clip to avoid overflow
        w_new = 1 / (1 + np.exp(exponent))
        if np.linalg.norm(w - w_new) < 1e-3: break
        w = w_new
    return z

def load_spectrum(filepath):
    """Loads a raw spectrum from a CSV file."""
    df = pd.read_csv(filepath)
    raw_intensities = df.iloc[:, 1].values 
    return raw_intensities

class DSD_Model:
    def __init__(self, weights_path):
        print(f"Loading Deep Spectral Deconvolution Model from {weights_path}...")
        self.model = tf.keras.models.load_model(weights_path)
        
    def predict_composition(self, spectrum, apply_arpls=True):
        processed_spectrum = spectrum.copy()
        
        if apply_arpls:
            print("Applying ArPLS baseline correction...")
            baseline = baseline_arpls(processed_spectrum)
            processed_spectrum = processed_spectrum - baseline
            
        # Min-Max Normalization 
        y_min, y_max = np.min(processed_spectrum), np.max(processed_spectrum)
        processed_spectrum = (processed_spectrum - y_min) / (y_max - y_min + 1e-9)
        
        # Reshape for CNN
        cnn_input = processed_spectrum.reshape(1, 1800, 1)
        probabilities = self.model.predict(cnn_input, verbose=0)[0]
        
        return {f"Analyte_{i}": round(prob, 4) for i, prob in enumerate(probabilities)}
