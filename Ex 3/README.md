# Three-Layer Neural Network for MNIST Classification  

## Objective  
Write a program to implement a three-layer neural network using the TensorFlow library (without Keras) to classify the MNIST handwritten digits dataset. The implementation demonstrates feed-forward and back-propagation approaches.  

## Model Description  
This is a three-layer neural network implemented using TensorFlow for classifying handwritten digits from the MNIST dataset.  

### Model Architecture  
- **Input Layer (784 neurons):** Accepts flattened 28x28 pixel images.  
- **Hidden Layer 1 (128 neurons):** Uses ReLU activation to learn non-linear features.  
- **Hidden Layer 2 (64 neurons):** Uses ReLU activation for deeper representation.  
- **Output Layer (10 neurons):** Uses softmax activation to classify digits (0-9).  

### Training Details  
- **Loss Function:** Categorical cross-entropy.  
- **Optimizer:** Adam Optimizer.  
- **Training Method:** Mini-batch Gradient Descent with batch size = 32.  

---

## Code Explanation  

### 1. Load & Preprocess Data  
- Normalize images (`x_train` & `x_test`) to range [0,1].  
- Flatten images (28x28 → 784).  
- Convert labels to one-hot encoding.  

### 2. Initialize Model Parameters  
- **Weights (W1, W2, W3):** Initialized with small random values.  
- **Biases (b1, b2, b3):** Initialized as zeros.  

### 3. Feed-Forward Propagation  
- **Layer 1:** `a1 = ReLU(X * W1 + b1)`  
- **Layer 2:** `a2 = ReLU(a1 * W2 + b2)`  
- **Output Layer:** `softmax(a2 * W3 + b3)`  

### 4. Loss Calculation  
- Uses `softmax_cross_entropy_with_logits_v2`.  

### 5. Backpropagation & Optimization  
- Uses `AdamOptimizer()`.  
- Updates weights and biases.  

### 6. Training (Mini-Batch Gradient Descent)  
- Iterates through 10 epochs with batch size = 32.  
- Prints training loss & accuracy.  

### 7. Testing  
- Evaluates the trained model on test data.  

---

## Comments  

### ✅ Good Implementation  
- Uses raw TensorFlow without Keras for full control.  
- Efficient training with mini-batch gradient descent.  

### ✅ Basic Model  
- Works well for MNIST but may not generalize to complex tasks.  

### 🔹 Improvement Suggestions  
- Use **Adam optimizer** instead of SGD for faster convergence.  
- Ensure **TensorFlow v2 Compatibility** by using `tf.compat.v1.disable_eager_execution()` for TF2 users.  

---  