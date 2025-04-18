# Experiment 2: Implementation of Multi-Layer Perceptron

## 1. Objective
Write a program to implement a multi-layer perceptron (MLP) network with one hidden layer using NumPy in Python. Demonstrate that it can learn the XOR Boolean function.

## 2. Description of the Model
For this experiment, an MLP is designed with one hidden layer to solve the XOR function, which is a non-linearly separable problem. Since a single-layer perceptron cannot solve XOR, there is a need to introduce a hidden layer with multiple neurons.

### Model Components
- **Input Layer**: 2 neurons (for two input bits)
- **Hidden Layer**: 4 perceptrons (to learn intermediate patterns)
- **Activation Function**: Step function (binary output)
- **Output Layer**: 1 perceptron (final XOR output)

## 3. Python Implementation 

## 4. Description of Code
### Perceptron Class
- Implements a simple perceptron with a step activation function.
- Performs forward propagation and weight updates using the perceptron learning rule.

### Hidden Layer Training
- Trains 4 separate perceptrons to learn intermediate patterns for XOR.

### Final Output Layer Training
- Uses the outputs from the hidden layer as inputs to a final perceptron.
- This final perceptron learns the XOR function.

## 5. Evaluation
### Accuracy
- 100% accuracy shows that the perceptron has perfectly learned and classified the XOR logic gate.

### Confusion Matrix
- Shows correct classifications for XOR, proving the MLP learns nonlinear functions.

## 6. Comments
### Limitations
- A manually designed 4-perceptron hidden layer works for XOR but might need tuning for different problems or may not work for complex problems.
- Training takes longer than a single-layer perceptron requires.

### Scope for Improvement
- Use of feedforward and backward propagation can more easily solve the XOR without explicitly training the hidden layer.
- Using a Sigmoid Activation Function: Instead of a step function allows smoother weight updates.