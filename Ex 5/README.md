# Fashion MNIST Classification with CNN

## Objective
The goal of this project is to build a Convolutional Neural Network (CNN) to classify grayscale images of clothing items (from the Fashion MNIST dataset) into 10 different categories such as shirts, shoes, bags, etc. The model should be able to accurately predict the class of unseen clothing images.

---

## 🔹 Dataset
The dataset used is Fashion MNIST, which consists of:
- **60,000 training images**
- **10,000 test images**
- Each image is **28x28 pixels**, grayscale (1 channel).
- There are **10 categories** (e.g., T-shirt, Trouser, Pullover, etc.).




## 🔹 Preprocessing
To prepare the dataset for training:
1. **Reshape and normalize the images**:
    - Reshape to `(28, 28, 1)` to include the channel dimension.
    - Normalize pixel values from `[0, 255]` to `[0, 1]`.

    


## 🔹 Architecture Overview
The CNN model is created using a Sequential API and includes:
1. **Conv2D layer** (32 filters, 5x5 kernel, ReLU, L2 regularization)
2. **MaxPooling2D** (2x2)
3. **Conv2D layer** (64 filters, 5x5 kernel, ReLU, L2 regularization)
4. **MaxPooling2D** (2x2)
5. **Flatten**
6. **Dense layer** (128 units, ReLU, L2 regularization)
7. **Output Dense layer** (10 units, Softmax)



## 🔹 Training Strategy
-       Optimizer: Adam with exponential decay schedule
        Loss Function: Categorical Crossentropy with label smoothing
        Callbacks: Early stopping and model checkpoint



## 🔹 Evaluation
Best model loaded from saved weights
Achieved high validation accuracy with reduced overfitting
```

### Results:
- **Final Test Accuracy**: ~89.26%
- **Final Test Loss**: ~0.397


---

## 🔹 My Comments
- The model shows steady improvement over the 5 epochs and ends with a high validation accuracy (~89%).
- L2 regularization helps in generalizing well and preventing overfitting.
- Fashion MNIST is a good starter dataset, and the model performs quite well for a simple architecture.

 