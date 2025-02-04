import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Load dataset
data = pd.read_csv("/home/srabanti/Downloads/Electric_Production.csv")
data['DATE'] = pd.to_datetime(data['DATE'])  # Ensure the date column is in datetime format
data.set_index('DATE', inplace=True)

# Feature and target selection
values = data['IPG2211A2N'].values 
scaler = MinMaxScaler()
values_scaled = scaler.fit_transform(values.reshape(-1, 1))

# Create sliding windows
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

window_size = 30  # Look-back period
X, y = create_sequences(values_scaled, window_size)

# Reshape for CNN input
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Scale back to original values
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

# print("Predictions:", predictions)
# print("Actual Values:", y_test_original)

# Create a DataFrame to compare predictions and actual values
comparison_table = pd.DataFrame({
    'Actual Values': y_test_original.flatten(),
    'Predicted Values': predictions.flatten()
})

# Print the table
print(comparison_table)

# Compare predictions with actual values
import matplotlib.pyplot as plt

plt.plot(y_test_original, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()

