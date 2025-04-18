# Evaluation of a Three-Layer Neural Network in TensorFlow

## Objective
To evaluate the performance of a three-layer neural network with variations in activation functions, size of hidden layers, learning rate, batch size, and number of epochs using TensorFlow.

## Description of the Model
This neural network consists of:
- **Input layer** with 784 neurons (28x28 images)
- **Two hidden layers** (sizes vary in experiments)
- **Output layer** with 10 neurons (one per class)

Experiments vary:
- Activation functions (ReLU, Sigmoid)
- Hidden layer sizes
- Learning rate
- Batch size
- Number of epochs

## Performance Evaluation
Each experiment logs:
- Test accuracy
- Loss per epoch

You can visualize or extend this to include:
- Confusion matrix
- Loss curve plots

## My Comments
- Accuracy improves with more neurons and proper activation.
- Sigmoid gives lower performance compared to ReLU due to vanishing gradients.
- Learning rate and batch size significantly affect convergence speed.
- Can be further improved by adding dropout or batch normalization.