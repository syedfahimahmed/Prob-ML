import numpy as np
import pandas as pd

def one_hot_encode(df, categorical_columns):
    return pd.get_dummies(df, columns=categorical_columns)

def softmax(z):
    z = np.clip(z, -500, 500)
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def cost_function(y, y_pred, w):
    return -np.sum(y * np.log(y_pred)) + 0.5 * np.sum(w**2)

def predict(X, w):
    z = np.dot(X, w)
    y_pred = softmax(z)
    return y_pred

def train(X, y, epochs=500, lr=0.01):
    n_samples, n_features = X.shape
    _, n_classes = y.shape
    w = np.random.normal(size=(n_features, n_classes))
    for epoch in range(epochs):
        y_pred = predict(X, w)
        gradient = np.dot(X.T, y_pred - y) + w
        w -= lr * gradient
    return w

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Load the dataset
train_df = pd.read_csv('Assignment-2/car-1/train.csv')
test_df = pd.read_csv('Assignment-2/car-1/test.csv')

# Get the attribute names from data-desc.txt
with open('Assignment-2/car-1/data-desc.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines()]

columns_line = lines[-1]  # Get the last line, which contains the column names
columns = columns_line.split(',')
train_df.columns = columns
test_df.columns = columns

# One-hot encode the categorical attributes and labels
categorical_columns = train_df.columns  # Include the last column (label)
train_df_encoded = one_hot_encode(train_df, categorical_columns)
test_df_encoded = one_hot_encode(test_df, categorical_columns)

X_train = train_df_encoded.iloc[:, :-4].values.astype(float)
y_train = train_df_encoded.iloc[:, -4:].values.astype(float)
X_test = test_df_encoded.iloc[:, :-4].values.astype(float)
y_test = test_df_encoded.iloc[:, -4:].values.astype(float)
# Train the model
w = train(X_train, y_train)

# Evaluate the model
y_pred = predict(X_test, w)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

acc = accuracy(y_test_labels, y_pred_labels)
print(f"Test Data Prediction Accuracy: {acc * 100:.2f}%")