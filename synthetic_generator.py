import numpy as np

class DigitalTwinningGenerator:
    def __init__(self, target_len=1800):
        self.target_len = target_len
        self.x = np.linspace(0, 1, target_len)

    def generate_chebyshev_baseline(self, max_order=5):
        coeffs = np.random.randn(max_order + 1) * [1.0, 0.5, 0.2, 0.1, 0.05, 0.01][:max_order+1]
        return np.polynomial.chebyshev.chebval(self.x, coeffs)

    def add_composite_noise(self, spectrum, snr_target=50):
        gaussian_noise = np.random.normal(0, np.std(spectrum) * 0.05, self.target_len)
        poisson_noise = np.random.poisson(np.abs(spectrum) * 100) / 100.0 - np.abs(spectrum)
        noisy_spectrum = spectrum + gaussian_noise + poisson_noise * 0.1
        return noisy_spectrum

    def create_mixture(self, pure_spectra_dict, snr_target=50):
        mixture = np.zeros(self.target_len)
        labels = np.zeros(len(pure_spectra_dict))
        
        # Add target analytes with Log-Uniform concentrations
        for idx, (class_id, pure_spec) in enumerate(pure_spectra_dict.items()):
            if np.random.rand() > 0.5: # 50% chance of presence
                conc = 10 ** np.random.uniform(-5, -1)
                mixture += pure_spec * conc
                labels[idx] = 1
                
        # Add Non-linear Baseline Drift and Noise
        mixture += self.generate_chebyshev_baseline()
        mixture = self.add_composite_noise(mixture, snr_target=snr_target)
        
        # Normalize final mixture
        mixture = (mixture - np.min(mixture)) / (np.max(mixture) - np.min(mixture) + 1e-9)
        return mixture, labels
