import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
IMG_HEIGHT = 64
IMG_WIDTH = 64
CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 25
NUM_CLASSES = 1  # 1 for binary (e.g., open/closed eyes, smiling/neutral mouth)
LEARNING_RATE = 0.001

# ==========================================
# 2. DATA AUGMENTATION & PIPELINE
# ==========================================
def get_augmentation_model():
    """Data augmentation pipeline to prevent overfitting on small face crops."""
    return models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1)
    ], name="data_augmentation")

def create_dummy_dataset(num_samples=1000):
    """Generates synthetic dataset for demonstration purposes."""
    X = np.random.rand(num_samples, IMG_HEIGHT, IMG_WIDTH, CHANNELS).astype(np.float32)
    y = np.random.randint(0, NUM_CLASSES + 1 if NUM_CLASSES > 1 else 2, size=(num_samples, 1)).astype(np.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

# ==========================================
# 3. ADVANCED CUSTOM CNN ARCHITECTURE
# ==========================================
def build_eye_mouth_cnn(input_shape=(64, 64, 3), num_classes=1):
    """
    Deep Conv2D-MaxPool architecture specifically tuned for spatial 
    feature extraction in localized facial regions (eyes/mouth).
    """
    inputs = layers.Input(shape=input_shape)
    
    # Apply data augmentation only during training
    x = get_augmentation_model()(inputs)
    
    # Block 1: Low-level edge and corner detection
    x = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 2: Mid-level texture and shape patterns
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 3: High-level semantic features (open/closed, contours)
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Classifier Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer
    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid', name='output_layer')(x)
        loss_fn = 'binary_crossentropy'
        metric_list = ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name='output_layer')(x)
        loss_fn = 'sparse_categorical_crossentropy'
        metric_list = ['accuracy']
        
    model = models.Model(inputs=inputs, outputs=outputs, name="EyeMouth_Custom_CNN")
    
    # Compile model inside builder
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=metric_list
    )
    
    return model

# ==========================================
# 4. CALLBACKS SETUP
# ==========================================
def get_callbacks(model_name="best_crop_model.keras"):
    """Returns standard production callbacks for training stability."""
    early_stopping = callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=model_name,
        monitor='loss',
        save_best_only=True,
        verbose=1
    )
    
    return [early_stopping, reduce_lr, model_checkpoint]

# ==========================================
# 5. EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    print("Initializing Custom CNN Model for Eye & Mouth Crops...")
    
    # Build model
    model = build_eye_mouth_cnn(input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), num_classes=NUM_CLASSES)
    model.summary()
    
    # Generate mock training & validation data (Replace with your tf.keras.utils.image_dataset_from_directory)
    train_dataset = create_dummy_dataset(num_samples=800)
    val_dataset = create_dummy_dataset(num_samples=200)
    
    print("\nStarting model training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=get_callbacks()
    )
    
    print("\nTraining complete. Model weights saved successfully!")
