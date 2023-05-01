import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import shutil
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, classification_report


# base_path = "./data_folder"

# file_paths = []

# labels = []

label_dict = {'happy': 0, 'angry': 1, 'sad': 2, 'fear': 3}


# Specify the base path to the directory containing the training data
base_path = "./newData_folder/train"
labels = ['happy', 'angry', 'sad', 'fear']
# List to hold features
features = []
feature_labels = []

# Loop over all labels
for label in labels:
    emotion_dir = os.path.join(base_path, label)
    
    # Get all the file names in current directory
    files = os.listdir(emotion_dir)
    
    # Loop over the files
    for file in files:
        # Load the audio file
        y, sr = librosa.load(os.path.join(emotion_dir, file))
        
        # Extract various features
        mag, phase = librosa.magphase(librosa.stft(y))
        #mag_phase = librosa.magphase(librosa.stft(y))
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

        
        # Calculate the mean of each feature
        # Then add all features to the features list
        features.append([np.mean(mag), np.mean(mfcc), np.mean(zcr), np.mean(chroma_stft), np.mean(mel_spectrogram)])
        # Add the label to the test_labels list
        feature_labels.append(label_dict[label])

# Convert list to numpy array
features = np.array(features)
feature_labels = np.array(feature_labels)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# Train SVM
svc = SVC()
svc.fit(features_scaled, feature_labels)

# Train Random Forest
rf = RandomForestClassifier()
rf.fit(features_scaled, feature_labels)

# Train MLP
mlp = MLPClassifier()
mlp.fit(features_scaled, feature_labels)


# Testing


# Specify the base path to the directory containing the test data
base_path = "./newData_folder/test"

# List to hold test features and labels
test_features = []
test_labels = []

# Loop over all labels
for label in labels:
    emotion_dir = os.path.join(base_path, label)
    
    # Get all the file names in current directory
    files = os.listdir(emotion_dir)
    
    # Loop over the files
    for file in files:
        # Load the audio file
        y, sr = librosa.load(os.path.join(emotion_dir, file))
        
        # Extract various features and calculate their mean
        mag, phase = librosa.magphase(librosa.stft(y))
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        
        # Add these features to the test_features list
        test_features.append([np.mean(mag), np.mean(mfcc), np.mean(zcr), np.mean(chroma_stft), np.mean(mel_spectrogram)])
        
        # Add the label to the test_labels list
        test_labels.append(label_dict[label])

# Convert lists to numpy arrays and scale the features
test_features = scaler.transform(np.array(test_features))
test_labels = np.array(test_labels)

# Make predictions using the trained models
svc_predictions = svc.predict(test_features)
rf_predictions = rf.predict(test_features)
mlp_predictions = mlp.predict(test_features)


# Create a reverse dictionary to map numerical labels back to their string representations
reverse_label_dict = {v: k for k, v in label_dict.items()}

# Generate classification reports
svc_report = classification_report(test_labels, svc_predictions, target_names=list(reverse_label_dict.values()))
rf_report = classification_report(test_labels, rf_predictions, target_names=list(reverse_label_dict.values()))
mlp_report = classification_report(test_labels, mlp_predictions, target_names=list(reverse_label_dict.values()))

# Print the classification reports
print("SVC Classification Report")
print(svc_report)

print("Random Forest Classification Report")
print(rf_report)

print("MLP Classification Report")
print(mlp_report)
#print(classification_report(test_labels, svc_predictions))

# Generate confusion matrices
svc_cm = confusion_matrix(test_labels, svc_predictions)
rf_cm = confusion_matrix(test_labels, rf_predictions)
mlp_cm = confusion_matrix(test_labels, mlp_predictions)

# Display confusion matrices
svc_disp = ConfusionMatrixDisplay(confusion_matrix=svc_cm, display_labels=list(reverse_label_dict.values()))
rf_disp = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=list(reverse_label_dict.values()))
mlp_disp = ConfusionMatrixDisplay(confusion_matrix=mlp_cm, display_labels=list(reverse_label_dict.values()))

# Plot confusion matrices
plt.figure(figsize=(10,10))
svc_disp.plot(ax=plt.subplot(3,1,1))
plt.title("SVC Confusion Matrix")

rf_disp.plot(ax=plt.subplot(3,1,2))
plt.title("Random Forest Confusion Matrix")

mlp_disp.plot(ax=plt.subplot(3,1,3))
plt.title("MLP Confusion Matrix")

plt.tight_layout()
plt.show()