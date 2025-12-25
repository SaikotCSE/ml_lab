# CNN Image Classification using CIFAR-10 (PyTorch)

## 👤 Author
**Emon Das Saikot**  
Student ID: **210109**  
Session: **2021–22**  
Department of Computer Science & Engineering  
Jashore University of Science and Technology  

---

## Project Overview
This project implements a complete **Convolutional Neural Network (CNN)** image classification pipeline using **PyTorch**.  
The model is trained on the **CIFAR-10 dataset** and evaluated on **real-world images captured using a smartphone**, highlighting the challenges of applying deep learning models trained on benchmark datasets to real-world data.

The project strictly follows assignment requirements, including **automation, reproducibility, visualization, and real-world testing**.

---

## Objectives
- Train a CNN from scratch on the CIFAR-10 dataset
- Apply proper image preprocessing and normalization
- Evaluate performance on unseen test data
- Test the trained model on real-world phone images
- Analyze model generalization and limitations
- Maintain a fully automated and reproducible pipeline

---

## Repository Structure
cnn-cifar10-assignment/
│
├── dataset/
│   └── phone_images/          # 13 real-world images captured using phone
│
├── model/
│   └── 210109.pth             # Saved trained model weights
│
├── 210109.ipynb               # Google Colab notebook
├── README.md                  # Project documentation


---

## Dataset Description

### CIFAR-10 Dataset
- 60,000 RGB images (32×32)
- 10 classes:
  - airplane
  - automobile
  - bird
  - cat
  - deer
  - dog
  - frog
  - horse
  - ship
  - truck
- 50,000 training images
- 10,000 test images

### Custom Phone Images
- 13 real-world images captured using a smartphone
- Objects correspond to CIFAR-10 classes
- Used **only for inference**, not training

---

## Data Preprocessing

### Training & Test Preprocessing
- Resize images to **32×32**
- Convert to tensor
- Normalize using CIFAR-10 statistics

**Normalization values:**
   mean = [0.4914, 0.4822, 0.4465]
   std  = [0.2470, 0.2435, 0.2616]


### Phone Image Preprocessing (Domain Adaptation)
To reduce domain mismatch between CIFAR-10 and real-world images, phone images are **degraded at inference time**:

- Resize to 32×32
- Color jitter (brightness, contrast, saturation, hue)
- Gaussian blur
- CIFAR-10 normalization

---

## Model Architecture
The CNN architecture consists of:
- Convolutional layers (`nn.Conv2d`)
- ReLU activation functions
- Max-pooling layers
- Fully connected (`nn.Linear`) layers

**Loss Function:** CrossEntropyLoss  
**Optimizer:** Adam  

---

## Training & Evaluation
- Training and validation accuracy tracked per epoch
- Loss vs Epoch and Accuracy vs Epoch plots generated
- Confusion matrix plotted for CIFAR-10 test set
- Visual error analysis performed on misclassified samples

---

## Real-World Testing
- Predictions performed on 13 phone images
- Predicted class and confidence score displayed

---

## Key Observations
- High accuracy on CIFAR-10 test data
- Reduced performance on real-world images due to domain shift
- Domain-aligned transformations and TTA improve predictions
- Demonstrates limitations of small benchmark datasets

---

## Automation & Reproducibility
- CIFAR-10 dataset downloaded automatically using `torchvision`
- Custom images loaded via `git clone`
- No manual file uploads required
- Notebook runs end-to-end using **Run All**

---

## Conclusion
This project demonstrates a complete CNN workflow and highlights the challenges of deploying deep learning models trained on benchmark datasets in real-world scenarios.

---

## How to Run
1. Open the Google Colab notebook
2. Click **Run All**
3. The pipeline will:
   - Clone the GitHub repository
   - Download CIFAR-10
   - Train the CNN (or load saved weights)
   - Predict phone images
   - Display results automatically
