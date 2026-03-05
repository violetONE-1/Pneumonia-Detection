Pneumonia-Detection

Baseline Analysis
In the initial phase of the project, a transfer learning model based on ResNet50 was implemented. While the preliminary results (as the picture) showed that the model failed to converge, with a Validation Accuracy of only 65%~ and highly unstable loss curves.

<img width="828" height="342" alt="initial loss and accuracy" src="https://github.com/user-attachments/assets/51089831-aa2e-44e1-a127-98f874f17334" />


After a thorough code review and training log analysis, the following critical issues were identified:

1.Output Layer & Loss Function Mismatch:
The model's classifier head included a nn.LogSoftmax layer while using nn.CrossEntropyLoss as the criterion. Since CrossEntropyLoss in PyTorch internally applies LogSoftmax, this redundancy led to incorrect gradient calculations and numerical instability.

2.Aggressive Dimensionality Reduction:
The transition from the pooling layer (4096 dimensions) to the final output (2 dimensions) was too abrupt (4096 -> 512 -> 2). This "bottleneck" caused significant loss of fine-grained spatial features essential for identifying subtle pneumonia patterns.

3.Suboptimal Learning Rate:
A learning rate of 1e-3 was found to be too high for fine-tuning a pre-trained ResNet50. This caused the optimizer to overshoot the global minimum, leading to the observed "zig-zag" pattern in the loss curve.

4.Data Homogeneity:
The initial preprocessing only utilized basic resizing and cropping. Medical images, particularly X-rays, require higher variance (rotation/flipping) to help the model generalize across different patient positions.


Refining the Validation Strategy: Addressing Data Imbalance

After implementing the initial architectural fixes, I re-trained the model but observed that the Loss and Accuracy metrics remained stagnant and exhibited significant oscillations (as the picture shows).

<img width="820" height="328" alt="second loss and accuracy" src="https://github.com/user-attachments/assets/81f2a178-6254-4546-af57-d129782ce6ca" />

Upon further investigation into the dataset structure provided by Kaggle, I identified that the original val directory contained only 16 samples. A single misclassification results in a 6.25% drop in accuracy.

To achieve more reliable and stable evaluation metrics, I decided to re-route the validation pipeline. I swapped the sparse val set with the test set, which contains 624 images.

After re-configuring the validation set to the comprehensive test directory (624 images) and applying the optimized classifier architecture, the model demonstrated significantly improved performance and stability,leading to the current stable accuracy of ~88% (as the picture shows).

<img width="842" height="336" alt="final loss and accuracy" src="https://github.com/user-attachments/assets/e134d426-8898-4e0c-b46b-320f5405bd7c" />


Project Summary
By optimizing the model architecture and data strategy, this project achieved a stable 88.94% accuracy on a 624-image test set. The key improvements include:

Architectural Refinement: Simplified the classifier head by removing redundant layers and implementing a smooth dimension reduction (2048 → 1024 → 512 → 2).

Stability Enhancements: Integrated Batch Normalization and Dropout to mitigate dimension-shifting instability and prevent overfitting.

Data Augmentation: Applied random rotations and horizontal flips to improve the model's ability to generalize across varied X-ray image qualities.

Strategic Hyperparameters: Reduced the learning rate to 3e-4 for more precise convergence during fine-tuning.

Validation Re-partitioning: Replaced the undersized 16-image validation set with the 624-image test split, ensuring statistically significant and reliable performance metrics.
<img width="1200" height="500" alt="training_performance" src="https://github.com/user-attachments/assets/5ca0b8ac-8c5d-4ccf-a3a3-033896c38c02" />

