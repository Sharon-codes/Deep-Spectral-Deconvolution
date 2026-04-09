import argparse
import csv
import json
import math
import os
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from model import build_dsd_model
from preprocessing import SpectralPreprocessor


PAH_ASCII_URLS = {
    "anthracene": "https://www.astrochemistry.org/pahdata/nanthracene.txt",
    "phenanthrene": "https://www.astrochemistry.org/pahdata/nphenanthrene.txt",
    "pyrene": "https://www.astrochemistry.org/pahdata/npyrene.txt",
}


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_if_missing(url: str, output_path: Path) -> Path:
    if output_path.exists():
        return output_path

    with urllib.request.urlopen(url, timeout=60) as response:
        output_path.write_bytes(response.read())
    return output_path


def parse_ascii_band_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    frequencies: List[float] = []
    intensities: List[float] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) < 3 or not row[0].strip():
                continue

            raw_freqs = [piece.strip() for piece in row[0].split(",") if piece.strip()]
            try:
                absolute_intensity = float(row[2].strip())
            except ValueError:
                continue

            split_intensity = absolute_intensity / max(len(raw_freqs), 1)
            for raw_freq in raw_freqs:
                try:
                    frequencies.append(float(raw_freq))
                    intensities.append(split_intensity)
                except ValueError:
                    continue

    if not frequencies:
        raise ValueError(f"No band data found in {path}")

    return np.asarray(frequencies, dtype=np.float32), np.asarray(intensities, dtype=np.float32)


def gaussian_broaden(
    freqs: np.ndarray,
    intensities: np.ndarray,
    grid: np.ndarray,
    fwhm: float,
) -> np.ndarray:
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    diff = grid[None, :] - freqs[:, None]
    profile = np.exp(-0.5 * (diff / sigma) ** 2)
    spectrum = (intensities[:, None] * profile).sum(axis=0)
    spectrum = spectrum.astype(np.float32)
    max_value = float(np.max(spectrum))
    if max_value > 0.0:
        spectrum /= max_value
    return spectrum


def shift_profile(profile: np.ndarray, shift_cm: float, step_cm: float) -> np.ndarray:
    sample_shift = shift_cm / step_cm
    x = np.arange(profile.size, dtype=np.float32)
    shifted = np.interp(
        x,
        x - sample_shift,
        profile,
        left=0.0,
        right=0.0,
    )
    shifted = shifted.astype(np.float32)
    max_value = float(np.max(shifted))
    if max_value > 0.0:
        shifted /= max_value
    return shifted


def load_reference_library(cache_dir: Path, grid: np.ndarray) -> Dict[str, np.ndarray]:
    ensure_directory(cache_dir)
    library: Dict[str, np.ndarray] = {}
    for analyte, url in PAH_ASCII_URLS.items():
        txt_path = download_if_missing(url, cache_dir / f"{analyte}.txt")
        freqs, intensities = parse_ascii_band_file(txt_path)
        spectrum = gaussian_broaden(freqs, intensities, grid, fwhm=14.0)
        library[analyte] = spectrum.astype(np.float32)
    return library


def load_background_spectra(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with csv_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)

    wavenumbers = np.asarray([float(value) for value in header[1:]], dtype=np.float32)
    ids = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=[0], dtype=str)
    spectra = np.loadtxt(
        csv_path,
        delimiter=",",
        skiprows=1,
        usecols=range(1, len(header)),
        dtype=np.float32,
    )
    return ids, wavenumbers, spectra


def restrict_wavenumber_range(
    wavenumbers: np.ndarray,
    spectra: np.ndarray,
    min_wavenumber: float,
    max_wavenumber: float,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = (wavenumbers >= min_wavenumber) & (wavenumbers <= max_wavenumber)
    filtered_wavenumbers = wavenumbers[mask].astype(np.float32)
    filtered_spectra = spectra[:, mask].astype(np.float32)
    return filtered_wavenumbers, filtered_spectra


def split_backgrounds(
    spectra: np.ndarray,
    holdout_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(spectra.shape[0])
    rng.shuffle(indices)
    split = int(round(spectra.shape[0] * (1.0 - holdout_fraction)))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return spectra[train_idx], spectra[test_idx]


class EmpiricalDigitalTwin:
    def __init__(
        self,
        wavenumbers: np.ndarray,
        reference_library: Dict[str, np.ndarray],
        preprocessor: SpectralPreprocessor,
        gain_per_ppm: float,
        extra_baseline_scale: float,
        noise_floor: float,
        seed: int,
    ):
        self.wavenumbers = wavenumbers.astype(np.float32)
        self.reference_library = reference_library
        self.preprocessor = preprocessor
        self.gain_per_ppm = float(gain_per_ppm)
        self.extra_baseline_scale = float(extra_baseline_scale)
        self.noise_floor = float(noise_floor)
        self.rng = np.random.default_rng(seed)
        self.grid_unit = np.linspace(-1.0, 1.0, self.wavenumbers.size, dtype=np.float32)
        self.step_cm = float(np.mean(np.diff(self.wavenumbers)))
        self.class_names = list(reference_library.keys())

    def _random_chebyshev(self) -> np.ndarray:
        coeffs = self.rng.normal(
            loc=0.0,
            scale=np.array([0.02, 0.01, 0.006, 0.004, 0.002, 0.001], dtype=np.float32),
        )
        baseline = np.polynomial.chebyshev.chebval(self.grid_unit, coeffs)
        return (self.extra_baseline_scale * baseline).astype(np.float32)

    def _noise_sigma(self, analyte_signal: np.ndarray, snr_target: float) -> float:
        analyte_peak = float(np.max(np.abs(analyte_signal))) + 1e-8
        return max(self.noise_floor, analyte_peak / max(snr_target, 1e-3))

    def _draw_targets(self) -> Dict[str, float]:
        chosen: Dict[str, float] = {}
        n_targets = int(self.rng.choice([0, 1, 2, 3], p=[0.12, 0.58, 0.22, 0.08]))
        if n_targets == 0:
            return chosen

        selected = self.rng.choice(self.class_names, size=n_targets, replace=False)
        for analyte in selected.tolist():
            ppm = 10 ** self.rng.uniform(math.log10(6.0), math.log10(220.0))
            chosen[analyte] = float(ppm)
        return chosen

    def create_sample(
        self,
        background: np.ndarray,
        forced_targets: Dict[str, float] | None = None,
        snr_range: Tuple[float, float] = (3.0, 18.0),
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        chosen = forced_targets or self._draw_targets()
        labels = np.zeros(len(self.class_names), dtype=np.float32)

        mixture = background.astype(np.float32).copy()
        mixture *= self.rng.uniform(0.98, 1.02)
        mixture += self.rng.normal(0.0, self.noise_floor * 0.5, size=mixture.shape).astype(np.float32)

        analyte_signal = np.zeros_like(mixture)
        for class_index, analyte in enumerate(self.class_names):
            ppm = chosen.get(analyte)
            if ppm is None:
                continue

            profile = self.reference_library[analyte]
            shift_cm = self.rng.normal(0.0, 3.0)
            shifted_profile = shift_profile(profile, shift_cm=shift_cm, step_cm=self.step_cm)
            signal = shifted_profile * (self.gain_per_ppm * ppm)
            analyte_signal += signal.astype(np.float32)
            labels[class_index] = 1.0

        snr_target = float(self.rng.uniform(*snr_range))
        noise_sigma = self._noise_sigma(analyte_signal, snr_target=snr_target)
        mixture += analyte_signal
        mixture += self._random_chebyshev()
        mixture += self.rng.normal(0.0, noise_sigma, size=mixture.shape).astype(np.float32)

        _, processed = self.preprocessor.full_pipeline(self.wavenumbers, mixture)
        return processed.astype(np.float32), labels, chosen


def build_dataset(
    twin: EmpiricalDigitalTwin,
    backgrounds: np.ndarray,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.zeros((n_samples, twin.wavenumbers.size), dtype=np.float32)
    y = np.zeros((n_samples, len(twin.class_names)), dtype=np.float32)

    for index in range(n_samples):
        background = backgrounds[twin.rng.integers(0, backgrounds.shape[0])]
        spectrum, labels, _ = twin.create_sample(background)
        X[index] = spectrum
        y[index] = labels

    return X, y


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    thresholds = np.zeros(y_true.shape[1], dtype=np.float32)
    grid = np.linspace(0.1, 0.9, 33)

    for class_index in range(y_true.shape[1]):
        best_threshold = 0.5
        best_score = -1.0
        y_class_true = y_true[:, class_index]
        y_class_prob = y_prob[:, class_index]
        for threshold in grid:
            preds = (y_class_prob >= threshold).astype(int)
            tp = float(np.sum((preds == 1) & (y_class_true == 1)))
            tn = float(np.sum((preds == 0) & (y_class_true == 0)))
            fp = float(np.sum((preds == 1) & (y_class_true == 0)))
            fn = float(np.sum((preds == 0) & (y_class_true == 1)))
            tpr = tp / max(tp + fn, 1.0)
            tnr = tn / max(tn + fp, 1.0)
            score = 0.65 * tpr + 0.35 * tnr
            if score > best_score or (abs(score - best_score) < 1e-8 and threshold < best_threshold):
                best_threshold = float(threshold)
                best_score = float(score)
        thresholds[class_index] = best_threshold

    return thresholds


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    class_names: Iterable[str],
) -> Dict[str, object]:
    y_pred = (y_prob >= thresholds[None, :]).astype(int)
    macro_auc = float(roc_auc_score(y_true, y_prob, average="macro"))
    per_class_auc = {
        class_name: float(roc_auc_score(y_true[:, idx], y_prob[:, idx]))
        for idx, class_name in enumerate(class_names)
    }
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    metrics: Dict[str, object] = {
        "macro_auc": macro_auc,
        "macro_f1": macro_f1,
        "thresholds": {class_name: float(thresholds[idx]) for idx, class_name in enumerate(class_names)},
        "per_class_auc": per_class_auc,
    }
    return metrics


def fit_pca_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=15, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    classifier = OneVsRestClassifier(
        SVC(
            kernel="rbf",
            C=10.0,
            gamma=0.1,
            probability=True,
            random_state=42,
        )
    )
    classifier.fit(X_train_pca, y_train)
    val_prob = np.column_stack(
        [estimator.predict_proba(X_val_pca)[:, 1] for estimator in classifier.estimators_]
    ).astype(np.float32)
    test_prob = np.column_stack(
        [estimator.predict_proba(X_test_pca)[:, 1] for estimator in classifier.estimators_]
    ).astype(np.float32)
    return val_prob, test_prob


def compute_lod_table(
    twin: EmpiricalDigitalTwin,
    model_predictor,
    thresholds: np.ndarray,
    holdout_backgrounds: np.ndarray,
    low_snr: float,
    ppm_grid: Iterable[float],
    n_samples_per_level: int,
) -> Dict[str, object]:
    ppm_list = [float(ppm) for ppm in ppm_grid]
    lods: Dict[str, float | None] = {}
    traces: Dict[str, List[Dict[str, float]]] = {}

    for class_index, analyte in enumerate(twin.class_names):
        analyte_trace: List[Dict[str, float]] = []
        lod_value = None

        for ppm in ppm_list:
            X_level = np.zeros((n_samples_per_level, twin.wavenumbers.size), dtype=np.float32)
            y_level = np.zeros((n_samples_per_level,), dtype=np.int32)
            forced_targets = {analyte: ppm}

            for sample_index in range(n_samples_per_level):
                background = holdout_backgrounds[twin.rng.integers(0, holdout_backgrounds.shape[0])]
                spectrum, labels, _ = twin.create_sample(
                    background,
                    forced_targets=forced_targets,
                    snr_range=(low_snr, low_snr + 0.25),
                )
                X_level[sample_index] = spectrum
                y_level[sample_index] = int(labels[class_index])

            probs = model_predictor(X_level)[:, class_index]
            preds = (probs >= thresholds[class_index]).astype(int)
            tpr = float(np.mean(preds == y_level))
            analyte_trace.append({"ppm": ppm, "tpr": tpr})
            if lod_value is None and tpr >= 0.95:
                lod_value = ppm

        lods[analyte] = lod_value
        traces[analyte] = analyte_trace

    return {"lod_ppm": lods, "traces": traces}


def compute_isomer_confusion(
    twin: EmpiricalDigitalTwin,
    model_predictor,
    holdout_backgrounds: np.ndarray,
    anthracene_threshold: float,
    phenanthrene_threshold: float,
    n_samples_per_class: int,
    ppm: float,
) -> Dict[str, object]:
    target_classes = ["anthracene", "phenanthrene"]
    true_labels: List[str] = []
    pred_labels: List[str] = []

    for analyte in target_classes:
        forced = {analyte: ppm}
        for _ in range(n_samples_per_class):
            background = holdout_backgrounds[twin.rng.integers(0, holdout_backgrounds.shape[0])]
            spectrum, _, _ = twin.create_sample(background, forced_targets=forced, snr_range=(4.0, 8.0))
            probs = model_predictor(spectrum[None, :])[0]
            anth_prob = probs[twin.class_names.index("anthracene")]
            phen_prob = probs[twin.class_names.index("phenanthrene")]

            if anth_prob >= anthracene_threshold and anth_prob >= phen_prob:
                pred = "anthracene"
            elif phen_prob >= phenanthrene_threshold and phen_prob > anth_prob:
                pred = "phenanthrene"
            else:
                pred = "unresolved"

            true_labels.append(analyte)
            pred_labels.append(pred)

    labels = ["anthracene", "phenanthrene", "unresolved"]
    matrix = confusion_matrix(true_labels, pred_labels, labels=labels)
    row_normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)

    return {
        "labels": labels,
        "counts": matrix.tolist(),
        "row_normalized": row_normalized.tolist(),
    }


def train_dsd(
    dataset: DatasetBundle,
    input_length: int,
    num_classes: int,
) -> Tuple[tf.keras.Model, np.ndarray, np.ndarray]:
    model = build_dsd_model(input_length=input_length, num_classes=num_classes)
    callbacks = [
        ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.5,
            patience=3,
            verbose=1,
            mode="max",
        ),
        EarlyStopping(
            monitor="val_auc",
            patience=6,
            restore_best_weights=True,
            mode="max",
        ),
    ]

    model.fit(
        dataset.X_train[..., None],
        dataset.y_train,
        validation_data=(dataset.X_val[..., None], dataset.y_val),
        epochs=25,
        batch_size=64,
        callbacks=callbacks,
        verbose=2,
    )

    val_prob = model.predict(dataset.X_val[..., None], verbose=0).astype(np.float32)
    test_prob = model.predict(dataset.X_test[..., None], verbose=0).astype(np.float32)
    return model, val_prob, test_prob


def make_bundle(
    twin: EmpiricalDigitalTwin,
    train_backgrounds: np.ndarray,
    test_backgrounds: np.ndarray,
    train_samples: int,
    val_samples: int,
    test_samples: int,
) -> DatasetBundle:
    X_train, y_train = build_dataset(twin, train_backgrounds, train_samples)
    X_val, y_val = build_dataset(twin, train_backgrounds, val_samples)
    X_test, y_test = build_dataset(twin, test_backgrounds, test_samples)

    return DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"
    ensure_directory(output_dir)
    ensure_directory(cache_dir)

    csv_path = Path(args.background_csv)
    _, wavenumbers, spectra = load_background_spectra(csv_path)
    wavenumbers, spectra = restrict_wavenumber_range(
        wavenumbers=wavenumbers,
        spectra=spectra,
        min_wavenumber=args.min_wavenumber,
        max_wavenumber=args.max_wavenumber,
    )
    train_backgrounds, holdout_backgrounds = split_backgrounds(
        spectra=spectra,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )

    preprocessor = SpectralPreprocessor(
        target_wavenumbers=wavenumbers.size,
        min_wn=float(wavenumbers.min()),
        max_wn=float(wavenumbers.max()),
    )
    reference_library = load_reference_library(cache_dir=cache_dir, grid=wavenumbers)
    twin = EmpiricalDigitalTwin(
        wavenumbers=wavenumbers,
        reference_library=reference_library,
        preprocessor=preprocessor,
        gain_per_ppm=args.gain_per_ppm,
        extra_baseline_scale=args.extra_baseline_scale,
        noise_floor=args.noise_floor,
        seed=args.seed,
    )

    dataset = make_bundle(
        twin=twin,
        train_backgrounds=train_backgrounds,
        test_backgrounds=holdout_backgrounds,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
    )

    dsd_model, dsd_val_prob, dsd_test_prob = train_dsd(
        dataset=dataset,
        input_length=wavenumbers.size,
        num_classes=len(twin.class_names),
    )
    dsd_thresholds = tune_thresholds(dataset.y_val, dsd_val_prob)
    dsd_metrics = compute_metrics(dataset.y_test, dsd_test_prob, dsd_thresholds, twin.class_names)

    svm_val_prob, svm_test_prob = fit_pca_svm(
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_val=dataset.X_val,
        y_val=dataset.y_val,
        X_test=dataset.X_test,
    )
    svm_thresholds = tune_thresholds(dataset.y_val, svm_val_prob)
    svm_metrics = compute_metrics(dataset.y_test, svm_test_prob, svm_thresholds, twin.class_names)

    dsd_lod = compute_lod_table(
        twin=twin,
        model_predictor=lambda X: dsd_model.predict(X[..., None], verbose=0).astype(np.float32),
        thresholds=dsd_thresholds,
        holdout_backgrounds=holdout_backgrounds,
        low_snr=args.lod_snr,
        ppm_grid=args.lod_grid,
        n_samples_per_level=args.lod_samples,
    )

    pca_model = PCA(n_components=15, random_state=args.seed)
    X_train_pca = pca_model.fit_transform(dataset.X_train)
    X_holdout_for_lod = holdout_backgrounds
    svm_lod_classifier = OneVsRestClassifier(
        SVC(kernel="rbf", C=10.0, gamma=0.1, probability=True, random_state=args.seed)
    )
    svm_lod_classifier.fit(X_train_pca, dataset.y_train)
    svm_lod = compute_lod_table(
        twin=twin,
        model_predictor=lambda X: np.column_stack(
            [est.predict_proba(pca_model.transform(X))[:, 1] for est in svm_lod_classifier.estimators_]
        ).astype(np.float32),
        thresholds=svm_thresholds,
        holdout_backgrounds=X_holdout_for_lod,
        low_snr=args.lod_snr,
        ppm_grid=args.lod_grid,
        n_samples_per_level=args.lod_samples,
    )

    isomer_confusion = compute_isomer_confusion(
        twin=twin,
        model_predictor=lambda X: dsd_model.predict(X[..., None], verbose=0).astype(np.float32),
        holdout_backgrounds=holdout_backgrounds,
        anthracene_threshold=float(dsd_thresholds[twin.class_names.index("anthracene")]),
        phenanthrene_threshold=float(dsd_thresholds[twin.class_names.index("phenanthrene")]),
        n_samples_per_class=args.isomer_samples,
        ppm=args.isomer_ppm,
    )

    model_path = output_dir / "empirical_dsd_model.keras"
    dsd_model.save(model_path)

    results = {
        "paper_anchor": {
            "validation_type": "empirical-background domain adaptation using real NeoSpectra/KSSL MIR soil spectra with archived experimental PAH standards",
            "spectral_range_cm-1": [float(wavenumbers.min()), float(wavenumbers.max())],
            "analytes": twin.class_names,
        },
        "dataset": {
            "background_csv": str(csv_path),
            "real_background_count": int(spectra.shape[0]),
            "train_background_count": int(train_backgrounds.shape[0]),
            "holdout_background_count": int(holdout_backgrounds.shape[0]),
            "train_samples": int(dataset.X_train.shape[0]),
            "val_samples": int(dataset.X_val.shape[0]),
            "test_samples": int(dataset.X_test.shape[0]),
        },
        "parameters": {
            "gain_per_ppm": args.gain_per_ppm,
            "extra_baseline_scale": args.extra_baseline_scale,
            "noise_floor": args.noise_floor,
            "holdout_fraction": args.holdout_fraction,
            "lod_snr": args.lod_snr,
            "lod_grid": list(args.lod_grid),
        },
        "metrics": {
            "dsd": dsd_metrics,
            "pca_svm": svm_metrics,
        },
        "lod": {
            "dsd": dsd_lod,
            "pca_svm": svm_lod,
        },
        "isomer_confusion": isomer_confusion,
        "artifacts": {
            "model_path": str(model_path),
        },
    }

    results_path = output_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Empirical-background validation for the DSD manuscript using real MIR soil spectra.",
    )
    parser.add_argument("--background-csv", required=True, help="Path to the real MIR background CSV.")
    parser.add_argument("--output-dir", default="outputs/empirical_validation")
    parser.add_argument("--train-samples", type=int, default=6000)
    parser.add_argument("--val-samples", type=int, default=1500)
    parser.add_argument("--test-samples", type=int, default=2000)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--gain-per-ppm", type=float, default=0.002)
    parser.add_argument("--extra-baseline-scale", type=float, default=0.30)
    parser.add_argument("--noise-floor", type=float, default=0.0022)
    parser.add_argument("--min-wavenumber", type=float, default=600.0)
    parser.add_argument("--max-wavenumber", type=float, default=1800.0)
    parser.add_argument("--lod-snr", type=float, default=3.0)
    parser.add_argument("--lod-samples", type=int, default=256)
    parser.add_argument("--isomer-samples", type=int, default=400)
    parser.add_argument("--isomer-ppm", type=float, default=80.0)
    parser.add_argument(
        "--lod-grid",
        type=float,
        nargs="+",
        default=[8, 10, 12, 15, 18, 20, 22, 25, 30, 40],
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    summary = run_experiment(arguments)
    print(json.dumps(summary["metrics"], indent=2))
    print(json.dumps(summary["lod"], indent=2))
