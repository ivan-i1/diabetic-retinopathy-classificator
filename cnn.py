import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models

image_dir = '../MLImages/Messidor1/All'
metadata_path = '../MLImages/Messidor1/Annotation_Base_CSV/Annotation_All.csv'
img_height = 224
img_width = 224
batch_size = 32
num_classes = 4
validation_split_ratio = 0.2
epochs = 15
random_seed = 580

# Read Data
try:
    df = pd.read_csv(metadata_path)
    print("Successfully read metadata CSV.")
    print("CSV Headers:", df.columns.tolist())

    filename_column_name = 'Image name'
    label_column_name = 'Retinopathy grade'

    if filename_column_name not in df.columns:
        raise ValueError(f"Metadata CSV missing required filename column: '{filename_column_name}'")
    if label_column_name not in df.columns:
        raise ValueError(f"Metadata CSV missing required label column: '{label_column_name}'")

    # df['image_filename'] = os.path.splitext(df[filename_column_name])[0] + '.jpg'
    df['image_filename'] = df[filename_column_name].astype(str).str.split('.').str[0] + '.jpg'
    image_filenames = df['image_filename'].tolist()

    labels = df[label_column_name].tolist()
    print(f"Extracted {len(image_filenames)} filenames and {len(labels)} labels.")

except FileNotFoundError:
    print(f"Error: Metadata file not found at {metadata_path}")
    exit()
except ValueError as ve:
    print(f"Error: {ve}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while reading metadata: {e}")
    exit()

# File existence check
file_paths = [os.path.join(image_dir, fname) for fname in image_filenames]

missing_files = [fp for fp in file_paths if not os.path.exists(fp)]
if missing_files:
    print(f"Warning: {len(missing_files)} image files listed in metadata are missing.")
    existing_files_data = [(fp, lbl) for fp, lbl in zip(file_paths, labels) if os.path.exists(fp)]
    if not existing_files_data:
        print("Error: No valid image files found.")
        exit()
    file_paths, labels = zip(*existing_files_data)
    file_paths = list(file_paths)
    labels = list(labels)
    print(f"Proceeding with {len(file_paths)} existing image files.")

# Banana split
train_paths, val_paths, train_labels, val_labels = train_test_split(
    file_paths,
    labels,
    test_size=validation_split_ratio,
    random_state=random_seed,
    stratify=labels
)
print(f"Training set size: {len(train_paths)}, Validation set size: {len(val_paths)}")

# CNN
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    # image = image / 255.0
    return image, label


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

# Shuffle Batch Prefetch
train_ds = train_ds.shuffle(buffer_size=len(train_paths)).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

print("Training and validation datasets prepared.")

model = models.Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(32, (3, 3), activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(64, (3, 3), activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(128, (3, 3), activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Eval
test_loss, test_acc = model.evaluate(val_ds, verbose=2)
print("Validation accuracy:", test_acc)

# Plot history
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.tight_layout()
plt.savefig('training_validation_plot.png')
