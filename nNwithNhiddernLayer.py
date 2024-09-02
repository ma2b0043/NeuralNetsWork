import numpy as np

# Step 1: Initialize parameters
def initialize_parameters(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    return parameters

# Step 2: Forward propagation
def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    return Z

def relu(Z):
    return np.maximum(0, Z)

def linear_activation_forward(A_prev, W, b, activation):
    Z = linear_forward(A_prev, W, b)
    
    if activation == "relu":
        A = relu(Z)
    elif activation == "linear":  # for the output layer in regression
        A = Z
    
    return A, Z

def forward_propagation(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Forward propagation for hidden layers
    for l in range(1, L):
        A_prev = A 
        A, Z = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append((A_prev, Z))

    # Forward propagation for the output layer
    A_out, Z_out = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "linear")
    caches.append((A, Z_out))
    
    return A_out, caches

# Step 3: Compute cost (Mean Squared Error for regression)
def compute_cost(A_out, Y):
    m = Y.shape[1]
    cost = (1 / (2 * m)) * np.sum((A_out - Y) ** 2)
    return np.squeeze(cost)

# Step 4: Backward propagation
def relu_derivative(Z):
    dZ = np.array(Z, copy=True)
    dZ[Z <= 0] = 0
    dZ[Z > 0] = 1
    return dZ

def linear_backward(dZ, cache, W):
    A_prev, Z = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, W, activation):
    A_prev, Z = cache
    
    if activation == "relu":
        dZ = dA * relu_derivative(Z)
    elif activation == "linear":
        dZ = dA  # for linear activation, dZ is just dA

    dA_prev, dW, db = linear_backward(dZ, cache, W)
    return dA_prev, dW, db

def backward_propagation(X, Y, caches, parameters):
    grads = {}
    L = len(caches)
    m = X.shape[1]
    A_out = caches[-1][1]
    
    # Derivative of cost with respect to A_out
    dA_out = A_out - Y

    # Backward propagation for output layer
    current_cache = caches[-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dA_out, current_cache, parameters['W' + str(L)], "linear")

    # Backward propagation for hidden layers
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(grads["dA" + str(l+1)], current_cache, parameters['W' + str(l+1)], "relu")
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l+1)] = dW
        grads["db" + str(l+1)] = db

    return grads

# Step 5: Update parameters
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    return parameters

# Step 6: Training the model
def model(X, Y, layer_dims, learning_rate=0.01, num_iterations=1000):
    parameters = initialize_parameters(layer_dims)

    for i in range(num_iterations):
        # Forward propagation
        A_out, caches = forward_propagation(X, parameters)

        # Compute cost
        cost = compute_cost(A_out, Y)

        # Backward propagation
        grads = backward_propagation(X, Y, caches, parameters)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 iterations
        if i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")

    return parameters

# Example usage
# X is your input data matrix (n_features x m_examples)
# Y is your output vector (1 x m_examples)
# layer_dims specifies the number of neurons in each layer, e.g., [n_features, 5, 4, 3, 1] for a 4-layer network

X = np.array([[2100, 1600, 2400], [3, 3, 4], [30, 20, 10], [7, 8, 5]])  # Features: square footage, number of rooms, age, neighborhood rating
Y = np.array([[400000, 330000, 369000]])  # Target: house prices
layer_dims = [4, 5, 4, 3, 1]  # Example: 4 features, 3 hidden layers, 1 output

# Normalize the features (e.g., divide by maximum value or standardize)
X = X / np.max(X, axis=1, keepdims=True)

# Train the model
parameters = model(X, Y, layer_dims, learning_rate=0.01, num_iterations=1000)

# Now you can use the trained parameters to predict prices for new houses
