# Sensor Image Classification & Audio Separation Project

## 📁 Project Overview

This repository contains an end-to-end machine learning pipeline focused on classifying images from sensor-based manufacturing data and separating audio signals using Independent Component Analysis (ICA). The project is structured into three major parts:

### Part 1: Convolutional Neural Network (CNN)

* Task: Multi-class image classification.
* Dataset: 7 labeled image classes - `bad_Zn`, `good_Zn`, `bad_Ti`, `good_Ti`, `bad_Tin`, `good_Tin`, `medium_Tin`.
* Preprocessing: Image resizing (128x128), grayscale conversion, normalization.
* Techniques: Data augmentation, class weighting for imbalance handling.
* Model: CNN built using TensorFlow/Keras with Conv2D, MaxPooling2D, Dense, and Dropout layers.
* Evaluation: Accuracy, F1 Score, Precision, Recall, and confusion matrices.

### Part 2: Random Forest & XGBoost

* Task: Traditional ML-based image classification.
* Feature Engineering:

  * Statistical descriptors (mean, std)
  * Edge detection (Sobel, Canny)
  * Filtering (Gaussian, Median)
* Dimensionality Reduction: PCA retaining 95% variance.
* Data Balancing: SMOTE (Synthetic Minority Oversampling Technique).
* Models:

  * Baseline: Random Forest with GridSearchCV for hyperparameter tuning.
  * Advanced: XGBoost classifier with extensive parameter grid.
* Evaluation: Precision, Recall, F1 Score, Accuracy, confusion matrices.

### Part 3: Audio Signal Separation via ICA

* Task: Separate music and white noise mixed into stereo audio.
* Techniques:

  * Signal synchronization and stereo conversion.
  * ICA using `FastICA` to separate source signals.
  * Spectrogram visualization for original and separated signals.
* Output: Mixed and separated `.wav` files.

---

## 🔧 Setup & Requirements

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/sensor-image-classification-audio-separation.git
cd sensor-image-classification-audio-separation
pip install -r requirements.txt
```

### `requirements.txt`

```txt
numpy
pandas
opencv-python
matplotlib
seaborn
scikit-learn
tensorflow
keras
joblib
imbalanced-learn
xgboost
soundfile
scipy
```

---

## 📀 Test Dataset Download

To test the models, download the image dataset from the following Google Drive link:

🔗 [Download Test Dataset](https://drive.google.com/file/d/1pxTXRGIBSeFyYRCskVBuFZA_qcHw8Oug/view?usp=sharing)

Make sure to extract and organize the folders according to class labels before running the scripts.

---


## 📊 Performance Summary

| Model       | Accuracy | Precision | Recall | F1 Score                                         |
| ----------- | -------- | --------- | ------ | ------------------------------------------------ |
| CNN         | \~94%    | High      | High   | High                                             |
| XGBoost     | \~94%    | High      | High   | High                                             |
| ICA (Audio) | N/A      | N/A       | N/A    | Subjective evaluation (audio separation success) |

---


## 🌐 License

This project is intended for academic use. Contact authors for further usage rights or commercial applications.

---

## 🌐 Acknowledgments

* Based on methods from LeCun et al. (2015), Chen & Guestrin (2016), Hyvärinen & Oja (2000).
* Inspired by best practices in machine learning and pattern recognition coursework.

---

