import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
from tensorflow.keras.applications import vgg16
from tensorflow.keras import layers, models

image_dir = '../MLImages/Messidor1/All'
metadata_path = '../MLImages/Messidor1/Annotation_Base_CSV/Annotation_All.csv'
img_height = 224
img_width = 224
img_channels = 3
validation_split_ratio = 0.2
num_classes = 4
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

# Exceptions
except FileNotFoundError:
    print(f"Error: Metadata file not found at {metadata_path}")
    exit()
except ValueError as ve:
    print(f"Error: {ve}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while reading metadata: {e}")
    exit()


try:
    df = pd.read_csv(metadata_path)
    filename_column_name = 'Image name'
    label_column_name = 'Retinopathy grade'
    df['image_filename'] = df[filename_column_name].astype(str).str.split('.').str[0] + '.jpg'
    image_filenames = df['image_filename'].tolist()
    labels = df[label_column_name].tolist()

    file_paths = [os.path.join(image_dir, fname) for fname in image_filenames]

    existing_files_data = [(fp, lbl) for fp, lbl in zip(file_paths, labels) if os.path.exists(fp)]
    if not existing_files_data:
        print("Error: No valid image files found.")
        exit()
    file_paths, labels = zip(*existing_files_data)
    file_paths = list(file_paths)
    labels = [int(l) for l in list(labels)]

    print(f"Using {len(file_paths)} images.")

except Exception as e:
    print(f"Error loading data: {e}")
    exit()

train_paths, val_paths, train_labels, val_labels = train_test_split(
    file_paths,
    labels,
    test_size=validation_split_ratio,
    random_state=42,
    stratify=labels
)
print(f"Training samples: {len(train_paths)}, Validation samples: {len(val_paths)}")

# VGG16
conv_base = vgg16.VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(img_height, img_width, img_channels))
conv_base.trainable = False
print("VGG16 base model loaded.")
conv_base.summary()


# Feature Extraction
def load_and_preprocess_for_vgg(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=img_channels)
    image = tf.image.resize(image, [img_height, img_width])
    image = vgg16.preprocess_input(image)
    return image

def extract_features(file_paths, batch_size=32):
    num_images = len(file_paths)

    dummy_input = tf.zeros((1, img_height, img_width, img_channels))
    dummy_output = conv_base(dummy_input)
    feature_shape = dummy_output.shape[1:]
    features = np.zeros((num_images,) + feature_shape)

    print(f"Extracting features with shape {feature_shape} for {num_images} images...")
    start_time = time.time()

    for i in range(0, num_images, batch_size):
        batch_paths = file_paths[i:min(i + batch_size, num_images)]
        batch_images = np.array([load_and_preprocess_for_vgg(p) for p in batch_paths])
        batch_features = conv_base.predict(batch_images, verbose=0)
        features[i:i + len(batch_paths)] = batch_features
        print(f"  Processed {min(i + batch_size, num_images)}/{num_images}", end='\r')

    print(f"\nFeature extraction completed in {time.time() - start_time:.2f} seconds.")
    return features.reshape((num_images, -1))

X_train_features = extract_features(train_paths)
X_val_features = extract_features(val_paths)

y_train = np.array(train_labels)
y_val = np.array(val_labels)

print("Training features shape:", X_train_features.shape)
print("Validation features shape:", X_val_features.shape)
print("Training labels shape:", y_train.shape)
print("Validation labels shape:", y_val.shape)


# Benchmark SVM
# print("\nTraining SVM classifier...")
# start_time = time.time()

# svm_model = SVC(kernel='rbf', C=1.0, probability=True, random_state=random_seed)
# svm_model.fit(X_train_features, y_train)

# print(f"SVM training completed in {time.time() - start_time:.2f} seconds.")


# Eval
# print("\nEvaluating SVM model...")
# y_pred_train = svm_model.predict(X_train_features)
# y_pred_val = svm_model.predict(X_val_features)

# train_accuracy = accuracy_score(y_train, y_pred_train)
# val_accuracy = accuracy_score(y_val, y_pred_val)

# print(f"Training Accuracy: {train_accuracy:.4f}")
# print(f"Validation Accuracy: {val_accuracy:.4f}")

# print("\nValidation Classification Report:")
# print(classification_report(y_val, y_pred_val, target_names=[f'Grade {i}' for i in range(num_classes)]))


print("Training SVM classifier...")
start_time = time.time()
param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
grid_search = GridSearchCV(SVC(random_state=random_seed), param_grid, refit=True, verbose=2, cv=3)
grid_search.fit(X_train_features, y_train)
print(f"SVM training completed in {time.time() - start_time:.2f} seconds.")

print("Best SVM parameters")
print(grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)
best_svm = grid_search.best_estimator_
y_pred_val_tuned = best_svm.predict(X_val_features)
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val_tuned))
print(classification_report(y_val, y_pred_val_tuned, target_names=[f'Grade {i}' for i in range(num_classes)]))

print("Generating Confusion Matrix...")

# Plot confusion matrix
cm = confusion_matrix(y_val, y_pred_val_tuned, labels=best_svm.classes_)
class_names = [f'Grade {i}' for i in range(num_classes)]

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('SVM Confusion Matrix (Validation Set)')
plt.savefig('svm_confusion_matrix.png')