# Object 6: Train and Evaluate a Recurrent Neural Network using PyTorch  

## Problem Statement  
Write a program to train and evaluate a Recurrent Neural Network (RNN) using the PyTorch library to predict the next value in a sample time series dataset.  

---

## Description of the Model  

### 1. Dataset  
The dataset used for training the model is the **International Airline Passengers dataset**, which records the monthly total number of airline passengers from January 1949 to December 1960.  

### 2. Model Architecture  
This code defines a simple Recurrent Neural Network (RNN) model for handling sequential data.  
- The input is processed through an RNN layer, which captures patterns across the sequence over time.  
- The output from the final time step is passed through a fully connected layer to generate the final prediction.  

### 3. Experimental Variations  
- **Input and Output Sizes**: Both are set to 1, as the task involves predicting a single continuous value at each step.  
- **Hidden Layer**: The model uses a hidden layer with 32 units and a single recurrent layer.  
- **Loss Function**: Mean Squared Error (MSE).  
- **Optimizer**: Adam optimizer with a learning rate of 0.01.  
- **Number of Epochs**: 100.  

---

## Description of the Code  

### 1. Import Required Libraries  
- **torch** and **torch.nn**: Core PyTorch libraries for creating the model and defining layers like RNN and Linear.  
- **pandas**: Used for reading and handling the dataset.  
- **numpy**: Provides efficient operations on arrays, particularly for data processing and manipulation.  
- **matplotlib.pyplot**: For visualizing the data, loss curves, and accuracy curves.  
- **MinMaxScaler** from `sklearn.preprocessing`: Used to scale the data between 0 and 1, as neural networks typically perform better with normalized data.  

### 2. Load and Preprocess the Data  
- The dataset `airline-passengers.csv` is loaded using pandas.  
- Only the `Passengers` column is used, representing monthly international airline passenger numbers.  
- The passenger numbers are normalized between 0 and 1 using `MinMaxScaler` from scikit-learn.  

### 3. Create Sequences for RNN Input  
- A helper function `create_dataset` prepares sequences of length `seq_length` (set to 10).  
- For each sequence, the model learns to predict the next passenger count.  
- Features `X` and labels `y` are created and converted into PyTorch tensors.  

### 4. Define the RNN Model  
- A class `RNNModel` is created by extending `nn.Module`.  
- The model consists of:  
    - An RNN layer (`nn.RNN`) with specified `input_size`, `hidden_size`, and `num_layers`.  
    - A fully connected (`Linear`) layer to output the final prediction.  
- **Forward Pass**:  
    - The sequence output from the RNN is taken.  
    - Only the last time step’s output is passed through the linear layer to predict the next value.  

### 5. Train the Model  
- **Loss Function**: Mean Squared Error Loss (`nn.MSELoss`).  
- **Optimizer**: Adam optimizer (`torch.optim.Adam`).  
- Training runs for 100 epochs.  
- In each epoch:  
    - Forward pass through the model.  
    - Compute loss and backpropagate gradients.  
    - Update model parameters.  
- Loss and a custom accuracy (based on Mean Absolute Error) are calculated and stored for each epoch.  
- Every 10 epochs, the current loss and accuracy are printed.  

### 6. Final Evaluation  
- After training, the model is evaluated.  
- Final Mean Squared Error (MSE) and custom accuracy are calculated between the predicted and actual passenger counts.  
- Predictions and actual values are de-normalized (inverse of MinMax scaling) before evaluation.  
- Plots for **"Predictions vs Actual Values"**, **"Loss Curve"**, and **"Accuracy Curve"** are displayed.  

---

## Comments  
- The batch size is not defined here; instead, the whole dataset is fed into the network because the size of the dataset is small.  
- A plain RNN is used, which works well for small datasets. For larger datasets, RNNs may not perform well due to the vanishing gradient problem. In such cases, models like **Long Short-Term Memory (LSTM)** should be preferred.  
- In this case, using LSTM gives even lower accuracy because the dataset is small and does not require storing long-term information. A plain RNN performs well for this dataset.  

---  