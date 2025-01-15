from function import *
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

# Define actions and parameters
actions = np.array(['A'])  # Define actions
no_sequences = 30  # Number of sequences per action
sequence_length = 30  # Number of frames per sequence
DATA_PATH = os.path.join('MP_Data')  # Path to saved keypoints

# Map actions to numerical labels
label_map = {label: num for num, label in enumerate(actions)}

# Prepare sequences and labels
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            try:
                # Load keypoints from .npy files
                res = np.load(npy_path, allow_pickle=True)  # Fix: allow_pickle=True
                window.append(res)
            except Exception as e:
                print(f"Error loading file {npy_path}: {e}")
                window.append(np.zeros(63))  # Fallback for missing data
        sequences.append(window)
        labels.append(label_map[action])

# Convert to numpy arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Set up TensorBoard callback
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Define the model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

# Print the model summary
model.summary()

# Save the model architecture and weights
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')

print("Model training and saving completed.")
