# Object 7: Retrain a Pretrained ImageNet Model for Medical Image Classification  

## Project Description  
This project uses transfer learning with the VGG16 model to classify CT scan images into two categories: **COVID** and **Non-COVID**. The dataset is organized into two folders and split into training and validation sets using an 80-20 ratio. Data augmentation techniques like resizing, flipping, and rotation are applied to improve the model's generalization.  

The original VGG16 layers are mostly frozen to retain learned features, with only the last convolutional block (from layer 24 to 30) fine-tuned. A custom classifier is added to the model to perform binary classification using sigmoid activation.  

The model is trained using the **Binary Cross Entropy Loss** function and optimized with the **Adam optimizer**. It runs for 20 epochs, tracking both training and validation accuracy and loss. After training, the model is saved for future use, and its performance is visualized using plots. The final accuracy on the validation set is also reported. This approach provides a reliable way to apply deep learning to medical image classification, even with a relatively small dataset.  

---

## Code Description  

### 1. Libraries and Hyperparameters  
- Essential libraries such as `torch`, `torchvision`, and `matplotlib` are imported for deep learning, image processing, and plotting.  
- Key hyperparameters include:  
    - **Image size**  
    - **Batch size**  
    - **Number of epochs**  
    - **Learning rate**  
- The computation device is selected (GPU if available, else CPU).  

### 2. Data Transformations  
- Training data is augmented with resizing, random horizontal flips, and rotations to improve model generalization.  
- Both training and validation data are normalized to match the input requirements of the pre-trained VGG16 model (ImageNet mean and std).  

### 3. Dataset Loading and Splitting  
- The dataset is loaded using `ImageFolder`, where each subfolder represents a class (`CT_COVID` and `CT_NonCOVID`).  
- The full dataset is split into training and validation sets using an 80-20 split.  
- `DataLoader` is used to feed images in batches for both training and validation, with shuffling enabled for training data.  

### 4. Model Setup: VGG16  
- A pre-trained VGG16 model is loaded to leverage features learned from ImageNet.  
- All layers are initially frozen to preserve learned features, except the last convolutional block (layers 24 to 30), which is unfrozen for fine-tuning.  

### 5. Custom Classifier Design  
- The original classifier of VGG16 is replaced with a custom fully connected network tailored for binary classification.  
- The custom classifier includes:  
    - Multiple `Linear`, `ReLU`, and `Dropout` layers  
    - A single neuron with a **Sigmoid activation** to output a probability between 0 and 1 (COVID or Non-COVID).  

### 6. Loss Function and Optimizer  
- **Binary Cross Entropy Loss (BCELoss)** is used for binary classification.  
- The **Adam optimizer** is chosen for its efficiency and adaptive learning rate.  
- Only the parameters requiring gradients (unfrozen layers and custom classifier) are passed to the optimizer.  

### 7. Training Loop  
- The model is trained over 20 epochs.  
- For each batch:  
    - A forward pass is performed.  
    - Loss is computed.  
    - Errors are backpropagated.  
    - Weights are updated.  
- Training loss and accuracy are tracked after every epoch.  
- The model is evaluated on the validation set to track performance on unseen data.  

### 8. Validation and Metric Tracking  
- During validation, the model is set to evaluation mode to prevent dropout and batch norm updates.  
- Accuracy and loss are computed without updating the weights.  
- Metrics are stored for visualization to monitor overfitting or generalization.  

### 9. Saving and Plotting Results  
- After training, the model is saved to a file named `covid_classifier_vgg16.pt` for future use.  
- Accuracy and loss trends for both training and validation sets are plotted using `matplotlib`.  

### 10. Final Evaluation  
- The model is evaluated one final time on the validation set to report its test accuracy.  
- The result is printed to the console for quick assessment.  

---

## Comments  

- The architecture referred from OpenAI ChatGPT achieved around **72% test accuracy**.  
- Modifying the classifier portion of the architecture resulted in minimal improvement (~1%).  
- Significant improvement in accuracy was observed when the output from the last convolutional block (layers 24 to 30) was unfrozen and included in the training process.  
- The potential of changing the dropout rate and exploring batch normalization for improving test accuracy has not been extensively explored.  
- Further exploration in these directions might yield a better architecture.  
