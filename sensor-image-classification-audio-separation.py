#Part 1

import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#--------------------------------------------------------------------------------------

# Set global plot settings
plt.rcParams.update({'font.size': 15, 'font.family': 'Arial'})

# Directories for good, medium, and bad folders
directories = {
    'bad_Zn':r'C:\Users\Mohammad Arqam\ML_DL_AI_CODE\Zink-20241125T130757Z-001\Zink\Bad',
    'good_Zn': r'C:\Users\Mohammad Arqam\ML_DL_AI_CODE\Zink-20241125T130757Z-001\Zink\Good',
    'bad_Ti':r'C:\Users\Mohammad Arqam\ML_DL_AI_CODE\Titanium-20241125T130800Z-001\Titanium\bad',
    'good_Ti': r'C:\Users\Mohammad Arqam\ML_DL_AI_CODE\Titanium-20241125T130800Z-001\Titanium\good',
    'bad_Tin':r'C:\Users\Mohammad Arqam\ML_DL_AI_CODE\Tin-20241125T130801Z-001\Tin\Tin\bad',
    'good_Tin': r'C:\Users\Mohammad Arqam\ML_DL_AI_CODE\Tin-20241125T130801Z-001\Tin\Tin\good',
    'medium_Tin': r'C:\Users\Mohammad Arqam\ML_DL_AI_CODE\Tin-20241125T130801Z-001\Tin\Tin\medium'
}

#--------------------------------------------------------------------------------------

# Resize dimensions
resize_dim = (128, 128)  # Resize to 128x128

def load_images_from_directory(directories, resize_dim):
    images = []
    labels = []
    for label, folder in directories.items():
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            if image is not None:
                image_resized = cv2.resize(image, resize_dim)
                images.append(image_resized)
                labels.append(label)
    return np.array(images, dtype='float32'), np.array(labels)
#--------------------------------------------------------------------------------------

# Load images and labels
X, y = load_images_from_directory(directories, resize_dim)

# Encode labels as integers
label_mapping = {'bad_Zn': 0, 'good_Zn': 1,'bad_Ti': 2, 'good_Ti': 3,'bad_Tin' : 4,'good_Tin' : 5,'medium_Tin' : 6}

y_encoded = np.array([label_mapping[label] for label in y])

# Normalize pixel values
X = X / 255.0

# Reshape for the neural network
X = X.reshape(X.shape[0], resize_dim[0], resize_dim[1], 1)

#--------------------------------------------------------------------------------------


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=11, stratify=y_encoded)

#--------------------------------------------------------------------------------------


# Create an ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the generator to the training data
datagen.fit(X_train)

# Augment the training data to increase the number of images by 1.5 times
augment_size = int(0.5 * X_train.shape[0])
X_train_augmented = np.empty((0, resize_dim[0], resize_dim[1], 1))
y_train_augmented = np.empty(0, dtype=int)

for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=augment_size, shuffle=False):
    X_train_augmented = np.vstack((X_train_augmented, X_batch))
    y_train_augmented = np.hstack((y_train_augmented, y_batch))
    if X_train_augmented.shape[0] >= X_train.shape[0] + augment_size:
        break

X_train_augmented = np.vstack((X_train, X_train_augmented[:augment_size]))
y_train_augmented = np.hstack((y_train, y_train_augmented[:augment_size]))
#--------------------------------------------------------------------------------------

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_augmented), y=y_train_augmented)
class_weights_dict = dict(enumerate(class_weights))
#--------------------------------------------------------------------------------------

# Build the Convolutional Neural Network
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(resize_dim[0], resize_dim[1], 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))  # Using softmax for multi-class classification

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#--------------------------------------------------------------------------------------

# Train the model with class weights
history = model.fit(X_train_augmented, y_train_augmented, epochs=18, batch_size=32, validation_data=(X_test, y_test), class_weight=class_weights_dict)

#--------------------------------------------------------------------------------------

# Evaluate the model
train_loss, train_accuracy = model.evaluate(X_train_augmented, y_train_augmented)
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Make predictions
y_train_pred = np.argmax(model.predict(X_train_augmented), axis=1)
y_test_pred = np.argmax(model.predict(X_test), axis=1)

# Evaluate performance
train_precision = precision_score(y_train_augmented, y_train_pred, average='weighted')
train_recall = recall_score(y_train_augmented, y_train_pred, average='weighted')
train_f1 = f1_score(y_train_augmented, y_train_pred, average='weighted')
train_conf_matrix = confusion_matrix(y_train_augmented, y_train_pred)

test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')
test_conf_matrix = confusion_matrix(y_test, y_test_pred)

# Print the performance metrics
performance_metrics = pd.DataFrame({
    'Dataset': ['Training', 'Testing'],
    'Accuracy': [train_accuracy, test_accuracy],
    'Precision': [train_precision, test_precision],
    'Recall': [train_recall, test_recall],
    'F1 Score': [train_f1, test_f1]
})
print(performance_metrics)

# Define class labels
class_labels = ['bad_Zn', 'good_Zn','bad_Ti','good_Ti','bad_Tin','good_Tin','medium_Tin']
#--------------------------------------------------------------------------------------

# Plot the final confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(pd.DataFrame(test_conf_matrix, index=class_labels, columns=class_labels), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Testing Data Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot the effect of epochs on accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.title('Effect of Epochs on Accuracy')
plt.show()

# Plot the effect of epochs on the loss function
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0, max(max(history.history['loss']), max(history.history['val_loss'])) * 1.1)
plt.legend()
plt.title('Effect of Epochs on Loss Function')
plt.show()

#--------------------------------------------------------------------------------------

# PART 2
import os
import cv2
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
import winsound
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time 
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier  # Importing XGBoost classifier
import winsound  # For playing a sound on Windows

# Function to apply filters and extract features
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image: {image_path}")
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize image to speed up processing
    gray = cv2.resize(gray, (256, 256))
    
    features = {}

    # Calculate mean and standard deviation of grayscale pixel values
    features['gray_mean'] = np.mean(gray)
    features['gray_std'] = np.std(gray)

    # Combined mean of Gaussian Blur, Median Filter, and original gray
    gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 3)
    median_blur = cv2.medianBlur(gray, 3)
    combined_blur_mean = np.mean([np.mean(gray), np.mean(gaussian_blur), np.mean(median_blur)])
    combined_blur_std = np.mean([np.std(gray), np.std(gaussian_blur), np.std(median_blur)])
    features['combined_blur_mean'] = combined_blur_mean
    features['combined_blur_std'] = combined_blur_std
    
    # Sobel Filter: Computes the gradient of the image intensity, useful for edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 3, 1, ksize=7)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 3, 1, ksize=7)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    combined_sobel = np.mean([np.mean(sobel_mag), np.std(sobel_mag)])
    features['combined_sobel'] = combined_sobel
    
    # Canny Edge Detection: Detects edges by looking for areas of rapid intensity change
    canny_edges = cv2.Canny(gray, 1, 3)
    combined_canny = np.mean([np.mean(canny_edges), np.std(canny_edges)])
    features['combined_canny'] = combined_canny

    return features
#--------------------------------------------------------------------------------------

# Function to process images in a specified directory
def process_images(directory, label):
    data = []

    def process_file(filename):
        file_path = os.path.join(directory, filename)
        try:
            print(f"Processing {file_path}...")
            features = extract_features(file_path)
            if features:
                features['label'] = label
                return features
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    filenames = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(filenames)} images in {directory} for label '{label}'.")

    results = Parallel(n_jobs=-1)(delayed(process_file)(filename) for filename in filenames)

    data = [res for res in results if res is not None]
    
    return data
#--------------------------------------------------------------------------------------

# Directories for good and bad folders
directories = {
    'bad_Zn':r'C:\Users\Mohammad Arqam\ML_DL_AI_CODE\Zink-20241125T130757Z-001\Zink\Bad',
    'good_Zn': r'C:\Users\Mohammad Arqam\ML_DL_AI_CODE\Zink-20241125T130757Z-001\Zink\Good',
    'bad_Ti':r'C:\Users\Mohammad Arqam\ML_DL_AI_CODE\Titanium-20241125T130800Z-001\Titanium\bad',
    'good_Ti': r'C:\Users\Mohammad Arqam\ML_DL_AI_CODE\Titanium-20241125T130800Z-001\Titanium\good',
    'bad_Tin':r'C:\Users\Mohammad Arqam\ML_DL_AI_CODE\Tin-20241125T130801Z-001\Tin\Tin\bad',
    'good_Tin': r'C:\Users\Mohammad Arqam\ML_DL_AI_CODE\Tin-20241125T130801Z-001\Tin\Tin\good',
    'medium_Tin': r'C:\Users\Mohammad Arqam\ML_DL_AI_CODE\Tin-20241125T130801Z-001\Tin\Tin\medium'
}


# Process images for each directory
data = []
for label, directory in directories.items():
    data.extend(process_images(directory, label))

df = pd.DataFrame(data)


df_pca = df.copy()
df_pca = df_pca.iloc[:, 1:-1]
# Perform Principle component analysis, the output is placed at the last column
pca = PCA()
pca.fit(df_pca)

# show principal components
print("Principal Components:")
print(pca.components_)

#display Scree plot- this shows the important parts
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(df_pca.columns) + 1), pca.explained_variance_ratio_, marker='o', linestyle='-')
plt.title('Scree Plot')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.show()
#--------------------------------------------------------------------------------------

# explained variance ratio table- this shows the importance of each component as a ratio
explained_variance_ratio_table = pd.DataFrame({'Component': np.arange(1, len(df_pca.columns) + 1),
                                               'Explained Variance Ratio': pca.explained_variance_ratio_})
print("\nExplained Variance Ratio Table:")
print(explained_variance_ratio_table)

# find important components
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
# use 95% as the threshold varience
num_important_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"\nNumber of important components: {num_important_components}")


# Perform PCA
pca = PCA(n_components=5)  # Use only the first 5 components
pca_components = pca.fit_transform(df.iloc[:, 1:-1])
# Use the last column as the output
output = df.iloc[:, -1].values

# Combine principal components and output
X = pca_components
y = output

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11, stratify=y)

#Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [10,20, 30, 40, 50],
    'max_depth': [None, 10, 15, 20],
    'min_samples_split': [10,20,30],
    'min_samples_leaf': [10, 20, 30],
   
    'criterion': ['gini' ]
}

# Initialize the Random Forest classifier
clf = RandomForestClassifier(random_state=11)

# Perform GridSearchCV
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# Use the best estimator from GridSearchCV to evaluate on the test set
best_clf = grid_search.best_estimator_

# Train the best model on the training data
best_clf.fit(X_train, y_train)
start_time = time.time()
# Make predictions on the training and testing data for the optimized model
y_train_pred = best_clf.predict(X_train)
y_test_pred = best_clf.predict(X_test)
end_time = time.time()
evaluation_time = end_time - start_time
# Evaluate the optimized model's performance on the training data
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, average='macro')
train_recall = recall_score(y_train, y_train_pred, average='macro')
train_f1 = f1_score(y_train, y_train_pred, average='macro')
train_conf_matrix = confusion_matrix(y_train, y_train_pred)

# Evaluate the optimized model's performance on the testing data
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='macro')
test_recall = recall_score(y_test, y_test_pred, average='macro')
test_f1 = f1_score(y_test, y_test_pred, average='macro')
test_conf_matrix = confusion_matrix(y_test, y_test_pred)

# Print performance metrics of the model for both train and test sets
performance_metrics = pd.DataFrame({
    'Dataset': ['Training', 'Testing'],
    'Accuracy': [train_accuracy, test_accuracy],
    'Precision': [train_precision, test_precision],
    'Recall': [train_recall, test_recall],
    'F1 Score': [train_f1, test_f1]
})
print(performance_metrics)

# Print the confusion matrix of the train set
print("\nTraining Data Confusion Matrix:")
train_conf_matrix_df = pd.DataFrame(train_conf_matrix, index=best_clf.classes_, columns=best_clf.classes_)
print(train_conf_matrix_df)

# Print the confusion matrix of the test set
print("\nTesting Data Confusion Matrix:")
test_conf_matrix_df = pd.DataFrame(test_conf_matrix, index=best_clf.classes_, columns=best_clf.classes_)
print(test_conf_matrix_df)

# Plot the final confusion matrix
plt.figure(figsize=(6, 8))
sns.set(font_scale=1.5)  # Adjust to make labels larger
sns.heatmap(pd.DataFrame(test_conf_matrix, index=best_clf.classes_, columns=best_clf.classes_), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('RF Data Confusion Matrix', fontsize=20, fontname='Arial')
plt.xlabel('Predicted Label', fontsize=20, fontname='Arial')
plt.ylabel('True Label', fontsize=20, fontname='Arial')
plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')
plt.show()
#--------------------------------------------------------------------------------------

#SMOTE to balance the dataset.
# Load the CSV file
# df = pd.read_csv('Zn_file1.csv')

# Separate features and target variable
X = df.drop(columns=['label'])  # Replace 'target' with your actual target column name
y = df['label']  # Replace 'target' with your actual target column name
print('SMOTE Started')
# Initialize SMOTE with desired parameters
smote = SMOTE(random_state=42)

# Apply SMOTE to balance the dataset
X_resampled, y_resampled = smote.fit_resample(X, y)
#--------------------------------------------------------------------------------------

# Combine resampled features and target into a new DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['target'] = y_resampled  # Re-add the target column

# Save the balanced data to a new CSV file
# df_resampled.to_csv('Zn_balanced.csv', index=False)

# print("Balanced dataset saved as Zn_balanced.csv")
#Using Esemble Algo to optimizing the output.
# Load the dataset from a CSV file
df = df_resampled.copy()

# EDA
print(df.shape)  # Check out the rows and columns the data has
print(df.head(10))  # Check out the starting 10 lines of the dataset
# print(df.isna().sum())
# Change the names of the columns
df.columns = ['gray_mean', 'gray_std', 'combined_blur_mean', 'combined_blur_std', 'combined_sobel', 'combined_canny', 'label']
print(df.isna().sum())
X = df.drop(['label'], axis=1)
y = df['label']
# Define label names
label_names = ['bad_Zn', 'good_Zn','bad_Ti','good_Ti','bad_Tin','good_Tin','medium_Tin']

# Encode the labels
label_encoding = {'bad_Zn': 0, 'good_Zn': 1,'bad_Ti': 2, 'good_Ti': 3,'bad_Tin' : 4,'good_Tin' : 5,'medium_Tin' : 6}
y = y.map(label_encoding)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11, stratify=y)

# Feature scaling (optional for XGBoost, as it handles scaling well)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [10,20,30,40,50],
   'max_depth': [None, 10, 15, 20],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3]
}
print('XGBoost Started')
# Initialize the XGBoost Classifier
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Initialize GridSearchCV with the XGBoost model and the parameter grid
grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, 
                           cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=11), 
                           n_jobs=-1, verbose=2)

# Perform the grid search on the training data
grid_search.fit(X_train, y_train)

# Print the best parameters found by GridSearchCV
print(f'Best parameters: {grid_search.best_params_}')

# Evaluate the best model on the training data
best_model = grid_search.best_estimator_
y_train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, average='macro')
train_recall = recall_score(y_train, y_train_pred, average='macro')
train_f1 = f1_score(y_train, y_train_pred, average='macro')

# Evaluate the best model on the testing data
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='macro')
test_recall = recall_score(y_test, y_test_pred, average='macro')
test_f1 = f1_score(y_test, y_test_pred, average='macro')

# Print evaluation metrics
print(f'Train Accuracy: {train_accuracy:.2f}')
print(f'Train Precision: {train_precision:.2f}')
print(f'Train Recall: {train_recall:.2f}')
print(f'Train F1 Score: {train_f1:.2f}')
print(f'Test Accuracy: {test_accuracy:.2f}')
print(f'Test Precision: {test_precision:.2f}')
print(f'Test Recall: {test_recall:.2f}')
print(f'Test F1 Score: {test_f1:.2f}')

# Plot the confusion matrix for the training data
conf_matrix_train = confusion_matrix(y_train, y_train_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted', fontsize=15, fontname='Arial')
plt.ylabel('Actual', fontsize=15, fontname='Arial')
plt.title('Training Confusion Matrix', fontsize=20, fontname='Arial')
plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')
plt.show()

# Plot the confusion matrix for the testing data
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 8))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted', fontsize=15, fontname='Arial')
plt.ylabel('Actual', fontsize=15, fontname='Arial')
plt.title('Testing Confusion Matrix', fontsize=20, fontname='Arial')
plt.xticks(fontsize=15, fontname='Arial')
plt.yticks(fontsize=15, fontname='Arial')
plt.show()

# Play a sound after execution
winsound.Beep(1000, 500)  # For Windows only; plays 1000 Hz for 500 ms

#--------------------------------------------------------------------------------------

# PART 3
import numpy as np
import soundfile as sf
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


#audio files
song, sample_rate_song = sf.read(r'C:\Users\Mohammad Arqam\ML_DL_AI_CODE\Audio files for part 3\song.wav')
noise, sample_rate_noise = sf.read(r'C:\Users\Mohammad Arqam\ML_DL_AI_CODE\Audio files for part 3\source_2_white_noise.wav')

print(f"Song data shape: {song.shape}")
print(f"Noise data shape: {noise.shape}")

# match lengths
minimum_length = min(song.shape[0], len(noise))
noise_trunc = noise[:minimum_length]

#column stack used for making to make the mono noise into stereo
noi_chan_2 = np.column_stack([noise_trunc, noise_trunc])

mixing_matrix = np.array([[0.8, 0.1],
                         [0.9, 0.05]])

sources = np.vstack([song[:, 0], noi_chan_2[:, 0]])

observed_signals = np.dot(mixing_matrix, sources)
#--------------------------------------------------------------------------------------

sf.write('mixed_signal_1.wav', observed_signals[0], sample_rate_song)
sf.write('mixed_signal_2.wav', observed_signals[1], sample_rate_song)

X = observed_signals.T
ica = FastICA(n_components=2)
separated_sources = ica.fit_transform(X)
#--------------------------------------------------------------------------------------

sf.write('separated_1.wav', separated_sources[:, 0], sample_rate_song)
sf.write('separated_2.wav', separated_sources[:, 1], sample_rate_song)


plt.figure(figsize=(10, 5))
f, t, Sxx = spectrogram(song[:, 0], fs=sample_rate_song)
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
plt.title('Spectrogram of Original Music Signal 1')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.show()


plt.figure(figsize=(10, 5))
f, t, Sxx = spectrogram(noise, fs=sample_rate_noise)
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
plt.title('Spectrogram of Original Noise Signal 2')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.show()


plt.figure(figsize=(10, 5))
f, t, Sxx = spectrogram(separated_sources[:, 0], fs=sample_rate_song)
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
plt.title('Spectrogram of Separated Noise Signal 1')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.show()



plt.figure(figsize=(10, 5))
f, t, Sxx = spectrogram(separated_sources[:, 1], fs=sample_rate_song)
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
plt.title('Spectrogram of Separated Music Signal 2')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.show()



print("Files created:")
print("1. mixed_signal_1.wav - First mix")
print("2. mixed_signal_2.wav - Second mix")
print("3. separated_1.wav - First separated source")
print("4. separated_2.wav - Second separated source")
#--------------------------------------------------------------------------------------
