import numpy as np
from tqdm import tqdm
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from data_ingestion import UniversalParser
from preprocessing import SpectralPreprocessor
from synthetic_generator import DigitalTwinningGenerator
from model import build_dsd_model

def run_pipeline():
    # 1. LOAD REAL DATA
    parser = UniversalParser()
    preprocessor = SpectralPreprocessor(target_wavenumbers=1800)
    
    # CHANGE THIS to your XML filename if different
    xml_path = "pahdb-complete-theoretical-v4.00.xml" 
    
    # Load the complete database (set limit=None for all species)
    raw_spectra = parser.parse_pahdb_xml(xml_path, limit=None) 
    
    if not raw_spectra:
        print("Failed to load PAHdb. Please check the file path.")
        return

    # Preprocess real standards
    pure_standards = {}
    class_names = []
    print("Preprocessing spectra...")
    for idx, spec in enumerate(tqdm(raw_spectra)):
        _, processed_y = preprocessor.full_pipeline(spec.x, spec.y)
        pure_standards[idx] = processed_y
        class_names.append(spec.id)
    
    num_classes = len(pure_standards)

    # 2. GENERATE SYNTHETIC MIXTURES (50,000 samples)
    print("\nGenerating 50,000 Synthetic Mixtures (Digital Twinning)...")
    generator = DigitalTwinningGenerator(target_len=1800)
    
    N_samples = 50000
    X_train = np.zeros((N_samples, 1800, 1))
    y_train = np.zeros((N_samples, num_classes))
    
    for i in range(N_samples):
        mixture, labels = generator.create_mixture(pure_standards, snr_target=np.random.uniform(2, 50))
        X_train[i, :, 0] = mixture
        y_train[i, :] = labels

    print("Data generation complete!")

    # 3. TRAIN THE MODEL
    print("\nInitializing 1D-CNN Model...")
    model = build_dsd_model(input_length=1800, num_classes=num_classes)
    
    callbacks = [
        ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, verbose=1, mode='max'),
        EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, mode='max')
    ]
    
    print("Starting Training...")
    model.fit(
        X_train, y_train,
        validation_split=0.2, 
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    # 4. SAVE MODEL WEIGHTS
    output_name = "meteorite_stress_test.keras"
    model.save(output_name)
    print(f"\nSUCCESS! Model saved as '{output_name}'")

if __name__ == "__main__":
    run_pipeline()
