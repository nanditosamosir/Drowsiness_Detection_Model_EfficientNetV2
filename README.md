# **Drowsiness Detection Model**

This project aims to detect Drowsiness in users using an EfficientNetV2-based model enhanced with Attention and Dual-Attention (CBAM and SE Block) modules. 
This model is designed to improve detection accuracy by considering channel and spatial features in the image, 
so that it can provide high sensitivity to small changes in the facial area that indicate sleepiness.

The dataset used consists of two classes, namely "open" and "closed", which are processed through a preprocessing stage, including image normalization. 
The training process is carried out on Google Colab by utilizing GPUs to accelerate computation. 
This model achieves high accuracy in testing with evaluation metrics such as confusion matrix, precision, recall, and F1-score.

The main features of this project include:

- Implementation of EfficientNetV2 as a backbone for computational efficiency.
- Addition of Attention and Dual-Attention modules to improve detection of important features in images.
- In-depth model performance evaluation and hyperparameter optimization.
- This project is suitable for applications that require real-time drowsiness detection, such as driving safety systems.

Testing:
1. EfficientNetV2-B2
2. EfficientNetV2-B2 + Attention (Squeeze and Excitation - SE)
3. EfficientNetV2-B2 + Dual-attention (Convolutional Block Attention Module - CBAM)
