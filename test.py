import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Normalization
# Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Preprocessing function
def preprocess_data(data):
    # Fill missing values
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    if 'Fare' in data.columns:
        data['Fare'].fillna(data['Fare'].median(), inplace=True)

    # Ensure 'Embarked' is of string type before mapping
    data['Embarked'] = data['Embarked'].astype(str)
    data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    print(data['Embarked'].unique())
    # Convert categorical columns to numeric
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

    # Select relevant features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    return data[features]

# Preprocess train and test data
X_train = preprocess_data(train_data)  # Extract features from train_data
y_train = train_data['Survived']      # Extract labels from train_data
X_test = preprocess_data(test_data)   # Extract features from test_data
# Split train data into training and validation sets
train_size = int(0.8 * len(X_train))  # 80% for training, 20% for validation
X_train_split, X_val = X_train[:train_size], X_train[train_size:]
y_train_split, y_val = y_train[:train_size], y_train[train_size:]

# Normalize the features using TensorFlow
normalizer = Normalization(axis=-1)
normalizer.adapt(X_train_split)

X_train_split = normalizer(X_train_split)
X_val = normalizer(X_val)
X_test = normalizer(X_test)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_split.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model on the training split and validate on the validation set
history = model.fit(X_train_split, y_train_split, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Predict on test data
predictions = (model.predict(X_test) > 0.5).astype(int)

# Save predictions to a CSV file
output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions.flatten()})
output.to_csv('submission.csv', index=False)
print("Predictions saved to submission.csv")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()