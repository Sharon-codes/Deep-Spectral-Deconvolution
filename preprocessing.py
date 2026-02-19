import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d

class SpectralPreprocessor:
    def __init__(self, target_wavenumbers=1800, min_wn=400, max_wn=4000):
        self.target_x = np.linspace(min_wn, max_wn, target_wavenumbers)

    def baseline_arpls(self, y, lam=1e5, itermax=100):
        """Asymmetric Least Squares (ArPLS) smoothing for baseline correction."""
        N = len(y)
        D = sparse.eye(N, format='csc')
        D = D[1:] - D[:-1]
        D = D[1:] - D[:-1]
        H = lam * D.T * D
        w = np.ones(N)
        
        for _ in range(itermax):
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

    def full_pipeline(self, x, y):
        """Interpolates, removes physics baseline, and normalizes."""
        # 1. Resample to 1800 points
        f = interp1d(x, y, kind='cubic', bounds_error=False, fill_value=0.0)
        y_interp = f(self.target_x)
        
        # 2. Physics Baseline Removal (ArPLS)
        baseline = self.baseline_arpls(y_interp)
        y_corrected = np.maximum(y_interp - baseline, 0)
        
        # 3. Min-Max Normalization
        y_min, y_max = np.min(y_corrected), np.max(y_corrected)
        y_norm = (y_corrected - y_min) / (y_max - y_min + 1e-9)
        
        return self.target_x, y_norm
