# CIFAR-10 Deep Learning Project

## 1. Problem Description
In this project, we address a **supervised image classification** problem using the **CIFAR-10** dataset. The dataset comprises 60,000 color images (32×32 pixels), evenly split into 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Our goal is to learn a model that accurately predicts the correct class label for any given input image.

## 2. EDA (Exploratory Data Analysis)
We begin by exploring the composition and characteristics of the training dataset. Key steps include:
- **Class Distribution Analysis**: We examine the counts of each class within our training subset to ensure that the data is roughly balanced. This information is plotted as a bar chart (see `outputs/class_distribution.png`).
- **Sample Image Inspection**: We visualize a few sample images from the training subset (see `outputs/sample_images.png`), which helps us qualitatively confirm that the data aligns with expectations (e.g., correct labels, data quality, and potential variations in image style).

From these EDA steps, we observe that each of the 10 classes is approximately equally represented (5000 images per class across the full CIFAR-10 dataset). The images have diverse content and color distribution, underscoring the importance of building a model robust to variations in object pose, lighting, and background clutter.

## 3. Model & Training Analysis
### Model Architecture
We implement a **four-layer Convolutional Neural Network (CNN)**:
1. **Convolution + ReLU** layers to learn local feature maps.
2. **Max Pooling** layers to reduce spatial dimensions and control overfitting.
3. **Fully Connected** layers to perform final classification.

The model takes a 3-channel (RGB) image of size 32×32 as input and outputs probabilities for each of the 10 classes.

### Data Augmentation
To improve generalization, we apply simple random transformations during training:
- **Random Crop** (with padding) to simulate object shifts.
- **Random Horizontal Flip** to handle left-right orientation changes.

### Training Procedure
- **Loss Function**: CrossEntropyLoss, a natural choice for multi-class classification.
- **Optimizer**: Adam (learning rate = 1e-3) for adaptive step sizes, allowing relatively fast convergence.
- **Train/Validation Split**: We hold out 5,000 images from the official CIFAR-10 training set (50,000 total) as a validation subset, leaving 45,000 for training. Validation accuracy guides hyperparameter tuning.
- **Epochs**: We train for 5 epochs in this demonstration (though more epochs often yield better accuracy).

This training pipeline captures key best practices: data augmentation, a clear train/validation split, a well-established optimizer, and a standard classification loss function.

## 4. Results
During training, we monitor:
1. **Training & Validation Loss**  
2. **Training & Validation Accuracy**  

Plots of these metrics are saved to:
- `outputs/training_loss.png`
- `outputs/training_accuracy.png`

At the end of training, we use the model with the **highest validation accuracy** to evaluate on the held-out test set of 10,000 images. We compute:
- **Test Accuracy & Loss**  
- **Confusion Matrix** (visualized in `outputs/confusion_matrix.png`)  
- **Classification Report** (precision, recall, F1-score) for each class  

Depending on hyperparameters, the final test accuracy is typically in the **mid-70% to mid-80% range** with this small CNN. The classification report highlights potential weaknesses in distinguishing visually similar classes (e.g., cats vs. dogs, automobiles vs. trucks), offering guidance for future improvements.

## 5. Discussion & Conclusion
Our CNN successfully classifies images into one of 10 categories, demonstrating how a basic convolutional architecture combined with data augmentation can handle modest image variability. Key observations include:
- **Class Accuracy Variations**: Certain classes with distinct color/shapes (e.g., ship, airplane) tend to yield higher accuracy, while classes with overlapping features (e.g., cats vs. dogs) may be more challenging.
- **Data Augmentation**: Even simple augmentations help the model generalize, suggesting that additional transforms (e.g., random rotations, color jitter) might further boost performance.
- **Model Capacity**: While sufficient for demonstration, deeper networks (ResNet, VGG, etc.) would likely achieve higher accuracy. Such architectures, or further hyperparameter tuning (e.g., adjusting learning rate schedules), could be tested for future improvements.

Overall, this project provides an **end-to-end** demonstration of constructing, training, and evaluating a CNN on CIFAR-10. It showcases essential deep learning practices such as careful EDA, data augmentation, consistent validation, and clear documentation of results. By building on this foundation, one can explore more advanced architectures and training techniques to further enhance performance on image classification tasks.
