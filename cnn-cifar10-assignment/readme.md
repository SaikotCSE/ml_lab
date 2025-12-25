👤 Author
Emon Das Saikot
Student ID: 210109
Session: 2021-22
Department of Computer Science & Engineering,
Jashore University of Science and Technology.

#Project Overview
This project implements a complete Convolutional Neural Network (CNN) image classification pipeline using PyTorch. The model is trained on the CIFAR-10 dataset and then evaluated on real-world images captured using a smartphone, demonstrating the challenges of applying deep learning models trained on benchmark datasets to real-world data.
The project strictly follows the assignment requirements, including automation, reproducibility, visualization, and real-world testing.

#Objectives
 -Train a CNN from scratch on the CIFAR-10 dataset.
 -Apply proper image preprocessing and normalization.
 -Evaluate performance on unseen test data.
 -Test the trained model on real-world phone images.
 -Analyze model generalization and limitations.
 -Maintain a fully automated, reproducible pipeline.demonstrating the challenges of applying deep learning models trained on benchmark datasets to real-world data.
 -The project strictly follows the assignment requirements, including automation,   reproducibility, visualization, and real-world testing.

#Repo structure
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

#Dataset Description
 -CIFAR-10 Dataset
  60,000 RGB images (32×32)
  10 classes:
   [airplane, automobile, bird, cat, deer,
    dog, frog, horse, ship, truck]

  50,000 training images.
  10,000 test images.

 -Custom Phone Images
  13 real-world images captured using a smartphone.
  Objects correspond to CIFAR-10 classes.
  Used only for inference, not training.

#Data Preprocessing
 Training & Test Preprocessing
   Resize → ToTensor → Normalize (CIFAR-10 mean & std)
 Normalization values:
   mean = [0.4914, 0.4822, 0.4465]
   std  = [0.2470, 0.2435, 0.2616]

#Phone Image Preprocessing (Domain Adaptation)
 To reduce domain mismatch between CIFAR-10 and real-world images, phone images are degraded at inference time:
  -Resize to 32×32
  -Color jitter (brightness, contrast, saturation, hue)
  -Gaussian blur
  -CIFAR-10 normalization
This helps the model handle real-world images more effectively without retraining.

#Model Architecture
The CNN is implemented using PyTorch and consists of:
 -Convolutional Layers (nn.Conv2d)
 -ReLU Activations
 -Max Pooling Layers
 -Fully Connected Layers (nn.Linear)
 -The model is trained using:
 -Loss Function: CrossEntropyLoss
 -Optimizer: Adam

#Training Process
 -Batch size: 64
 -Optimizer: Adam
 -Loss: CrossEntropyLoss
 -Training and validation accuracy tracked per epoch

#Outputs:
 -Training Loss vs Epochs
 -Training Accuracy vs Epochs
 -Validation Accuracy vs Epochs

#Evaluation & Visualization
✔ CIFAR-10 Test Set
  -Confusion matrix (heatmap)
  -Accuracy and loss analysis
  -Visual error analysis (misclassified images)
✔ Phone Image Prediction
  -Grid display of 13 phone images
  -Predicted class with confidence score

#Key Observations
 -The model performs well on CIFAR-10 test data.
 -Performance drops on real-world phone images due to domain shift.
 -Applying domain-aligned transformations and test-time augmentation improves predictions.
 -This demonstrates the limitation of training CNNs on small benchmark datasets without real-world fine-tuning.

#Conclusion
This project demonstrates a complete CNN workflow, highlights real-world generalization challenges, and follows best practices in deep learning experimentation. The results emphasize the importance of dataset diversity and domain adaptation when deploying models outside controlled benchmarks.