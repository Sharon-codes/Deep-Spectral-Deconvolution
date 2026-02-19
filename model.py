import tensorflow as tf
from tensorflow.keras import layers, models

def build_dsd_model(input_length=1800, num_classes=10):
    inputs = layers.Input(shape=(input_length, 1))
    
    # Encoder 1 (Broad features)
    x = layers.Conv1D(32, kernel_size=15, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Encoder 2 (Intermediate features)
    x = layers.Conv1D(64, kernel_size=10, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Encoder 3 (Fine fingerprint features)
    x = layers.Conv1D(128, kernel_size=5, padding='same', activation='relu')(x)
    
    # Translation Invariance
    x = layers.GlobalAveragePooling1D()(x)
    
    # Latent Dense
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    
    # Classifier Head
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
