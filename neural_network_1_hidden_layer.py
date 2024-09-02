import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return z > 0

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    return A1, A2

def compute_cost(A2, Y):
    m = Y.shape[1]
    cost = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
    return np.squeeze(cost)

def backward_propagation(X, Y, A1, A2, W2):
    m = X.shape[1]  # Number of training examples
    
    # Compute dZ2: Gradient of the loss with respect to output activation
    dZ2 = A2 - Y  # Difference between prediction and actual label
    
    # Compute dW2: Gradient of the loss with respect to weights in the output layer
    dW2 = np.dot(dZ2, A1.T) / m  # Average gradient over all examples
    
    # Compute db2: Gradient of the loss with respect to biases in the output layer
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m  # Average gradient over all examples
    
    # Compute dA1: Gradient of the loss with respect to hidden layer activations
    dA1 = np.dot(W2.T, dZ2)  # Backpropagated error from output to hidden layer
    
    # Compute dZ1: Gradient of the loss with respect to hidden layer pre-activations
    dZ1 = dA1 * relu_derivative(A1)  # Apply the derivative of ReLU function
    
    # Compute dW1: Gradient of the loss with respect to weights in the hidden layer
    dW1 = np.dot(dZ1, X.T) / m  # Average gradient over all examples
    
    # Compute db1: Gradient of the loss with respect to biases in the hidden layer
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m  # Average gradient over all examples
    
    return dW1, db1, dW2, db2


def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

# Example data
X = np.array([[70, 80, 90], [1.75, 1.80, 1.65], [36.5, 37.0, 37.2]]) # Example features: weight, height, temperature
Y = np.array([[1, 0, 1]]) # Labels: 1 = healthy, 0 = not healthy

# Initialize parameters
np.random.seed(1)
W1 = np.random.randn(4, 3) * 0.01  # 4 neurons in hidden layer, 3 features
b1 = np.zeros((4, 1))
W2 = np.random.randn(1, 4) * 0.01  # 1 neuron in output layer, 4 hidden neurons
b2 = np.zeros((1, 1))

# Forward propagation
A1, A2 = forward_propagation(X, W1, b1, W2, b2)

# Compute cost
cost = compute_cost(A2, Y)
print("Cost:", cost)

# Backward propagation
dW1, db1, dW2, db2 = backward_propagation(X, Y, A1, A2, W2)

# Update parameters
W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate=0.01)

print("w1:", w1)

print("b1:", b1)

print("w2:", w2)

print("b2:", b2)