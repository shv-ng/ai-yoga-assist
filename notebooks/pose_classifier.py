#!/usr/bin/env python
# coding: utf-8

# ## Install and Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import os
import pickle

print('TensorFlow version:', tf.__version__)
print('All libraries loaded ✅')


# ## Load Data

# In[2]:


df = pd.read_csv('../data/poses_full.csv')

feature_cols = [c for c in df.columns if c not in ['frame_id', 'sequence_id', 'label']]

print('Dataset shape:', df.shape)
print('\nSamples per pose:')
print(df['label'].value_counts())
df.head()


# ## Normalize Landmarks
# 
# Normalize each frame so the classifier is invariant to body position and distance from camera.
# 
# Method:
# - Translate hip midpoint to origin
# - Scale by torso height (hip-mid → shoulder-mid distance)
# 
# This must match exactly what `pipeline.py` does at inference time.

# In[3]:


# MediaPipe landmark indices used for normalization
_LEFT_HIP       = 23
_RIGHT_HIP      = 24
_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12
N_LANDMARKS     = 33

def normalize_frame(row: np.ndarray) -> np.ndarray:
    """
    Normalize a single (99,) landmark row.
    Layout: [x0,y0,z0, x1,y1,z1, ..., x32,y32,z32]
    Returns normalized (99,) array.
    """
    arr = row.reshape(N_LANDMARKS, 3).copy()   # (33, 3)

    hip_mid      = (arr[_LEFT_HIP] + arr[_RIGHT_HIP]) / 2.0
    shoulder_mid = (arr[_LEFT_SHOULDER] + arr[_RIGHT_SHOULDER]) / 2.0
    torso_height = np.linalg.norm(shoulder_mid - hip_mid)

    arr -= hip_mid
    if torso_height > 1e-6:
        arr /= torso_height

    return arr.flatten()


X_raw = df[feature_cols].values.astype(np.float32)
X_norm = np.array([normalize_frame(row) for row in X_raw], dtype=np.float32)

print('Raw X shape    :', X_raw.shape)
print('Normalized shape:', X_norm.shape)

# Sanity check: hip midpoint of first sample should be ~0 after normalization
sample = X_norm[0].reshape(33, 3)
hip_mid_check = (sample[_LEFT_HIP] + sample[_RIGHT_HIP]) / 2
print(f'Hip midpoint after normalization (should be ~0): {hip_mid_check}')


# ## Visualise the Data

# In[4]:


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Samples per pose
df['label'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
axes[0].set_title('Samples per Pose')
axes[0].set_xlabel('Pose')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

# Raw landmark distribution
axes[1].hist(X_raw.flatten(), bins=50, color='coral', edgecolor='black')
axes[1].set_title('Raw Landmark Values')
axes[1].set_xlabel('Value (0 to 1)')
axes[1].set_ylabel('Frequency')

# Normalized landmark distribution
axes[2].hist(X_norm.flatten(), bins=50, color='mediumseagreen', edgecolor='black')
axes[2].set_title('Normalized Landmark Values')
axes[2].set_xlabel('Value (body-relative units)')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# ## Encode Labels & Split Data

# In[5]:


le = LabelEncoder()
y  = le.fit_transform(df['label'])

print('Label encoding:')
for i, name in enumerate(le.classes_):
    print(f'  {i} → {name}')

y_onehot = to_categorical(y)

# Split by sequence_id so no sequence leaks across train/test
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_norm, y, groups=df['sequence_id']))

X_train, X_test = X_norm[train_idx], X_norm[test_idx]
y_train, y_test = y_onehot[train_idx], y_onehot[test_idx]

print(f'\nTraining samples : {len(X_train)}')
print(f'Testing samples  : {len(X_test)}')

# Verify no sequence overlap
train_seqs = set(df.iloc[train_idx]['sequence_id'])
test_seqs  = set(df.iloc[test_idx]['sequence_id'])
print(f'Sequence overlap : {train_seqs.intersection(test_seqs)}  (should be empty)')


# ## Build the Model

# In[6]:


model = Sequential([
    Dense(128, activation='relu', input_shape=(99,)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(len(le.classes_), activation='softmax')
], name='PoseClassifier')

model.summary()


# ## Compile & Train

# In[7]:


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=8, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=1
    ),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=60,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)


# ## Visualise Training Curves

# In[8]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'],     label='Train', color='steelblue')
axes[0].plot(history.history['val_accuracy'], label='Val',   color='coral')
axes[0].set_title('Accuracy over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'],     label='Train', color='steelblue')
axes[1].plot(history.history['val_loss'], label='Val',   color='coral')
axes[1].set_title('Loss over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()


# ## Confusion Matrix

# In[9]:


y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ## Metrics

# In[10]:


print('=' * 50)
print('FINAL MODEL EVALUATION')
print('=' * 50)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'\nOverall Accuracy : {accuracy*100:.2f}%')
print(f'Overall Loss     : {loss:.4f}')

top1 = tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred_probs, k=1).numpy().mean()
top2 = tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred_probs, k=2).numpy().mean()
print(f'\nTop-1 Accuracy   : {top1*100:.2f}%')
print(f'Top-2 Accuracy   : {top2*100:.2f}%')

print('\n' + '=' * 50)
print('PER POSE BREAKDOWN')
print('=' * 50)
print(classification_report(
    y_true, y_pred,
    labels=range(len(le.classes_)),
    target_names=le.classes_,
    zero_division=0
))


# In[11]:


# Per-Class F1-Score Bar Chart
from sklearn.metrics import f1_score

f1_scores = f1_score(y_true, y_pred, average=None, labels=range(len(le.classes_)))

plt.figure(figsize=(10, 5))
bars = plt.bar(le.classes_, f1_scores, color='steelblue', edgecolor='black')
plt.title('Per-Class F1-Score')
plt.xlabel('Pose')
plt.ylabel('F1-Score')
plt.ylim(0, 1.05)
plt.xticks(rotation=45, ha='right')

# Add value labels on top of each bar
for bar, score in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{score:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()


# ## Save Model & Encoder

# In[12]:


os.makedirs('../models', exist_ok=True)

model.save('../models/pose_classifier.h5')

with open('../models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print('Model saved   → ../models/pose_classifier.h5')
print('Encoder saved → ../models/label_encoder.pkl')


# ## Export to TFLite (for Raspberry Pi)

# In[13]:


# Convert to TFLite with float32 (safe default)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_path = '../models/pose_classifier.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f'TFLite model saved → {tflite_path}')
print(f'Size: {len(tflite_model) / 1024:.1f} KB')

# Quick correctness check — run one sample through TFLite
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

in_idx  = interpreter.get_input_details()[0]['index']
out_idx = interpreter.get_output_details()[0]['index']

sample = X_test[0:1].astype(np.float32)
interpreter.set_tensor(in_idx, sample)
interpreter.invoke()
tflite_probs = interpreter.get_tensor(out_idx)[0]
keras_probs  = model.predict(sample, verbose=0)[0]

print(f'\nKeras prediction : {le.classes_[np.argmax(keras_probs)]}  ({max(keras_probs)*100:.1f}%)')
print(f'TFLite prediction: {le.classes_[np.argmax(tflite_probs)]}  ({max(tflite_probs)*100:.1f}%)')
print('Outputs match:', np.allclose(keras_probs, tflite_probs, atol=1e-4))


# ## Smoke Test — Simulated Inference
# 
# Reproduces exactly what `realtime.py` does per frame, so you can verify end-to-end before touching the camera.

# In[14]:


CONFIDENCE_THRESHOLD = 0.6

def predict_pose(raw_landmark_row: np.ndarray):
    """Mirrors the realtime.py pipeline: normalize → predict."""
    normalized = normalize_frame(raw_landmark_row)
    probs      = model.predict(normalized.reshape(1, -1), verbose=0)[0]
    top_idx    = np.argmax(probs)
    confidence = probs[top_idx]
    label      = le.inverse_transform([top_idx])[0]

    if confidence < CONFIDENCE_THRESHOLD:
        return 'Unknown Pose', confidence, probs
    return label, confidence, probs

# Test on a real sample from the test set
raw_sample = X_raw[test_idx[0]]
true_label = df.iloc[test_idx[0]]['label']

label, confidence, probs = predict_pose(raw_sample)
print(f'True label     : {true_label}')
print(f'Predicted      : {label}  ({confidence*100:.1f}%)')
print()
print('All probabilities:')
for name, prob in sorted(zip(le.classes_, probs), key=lambda x: -x[1]):
    bar = '█' * int(prob * 40)
    print(f'  {name:<15} {prob*100:5.1f}%  {bar}')


# In[15]:


get_ipython().system('jupyter nbconvert --to script pose_classifier.ipynb')


# In[ ]:




