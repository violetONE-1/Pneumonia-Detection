Pneumonia-Detection

Baseline Analysis
In the initial phase of the project, a transfer learning model based on ResNet50 was implemented. While the preliminary results (as the picture) showed that the model failed to converge, with a Validation Accuracy of only ~11% and highly unstable loss curves.
<img width="554" height="358" alt="initial loss and accuracy" src="https://github.com/user-attachments/assets/d8ea6708-9565-4e71-8886-ac80c15f54af" />

After a thorough code review and training log analysis, the following critical issues were identified:

1.Output Layer & Loss Function Mismatch:
The model's classifier head included a nn.LogSoftmax layer while using nn.CrossEntropyLoss as the criterion. Since CrossEntropyLoss in PyTorch internally applies LogSoftmax, this redundancy led to incorrect gradient calculations and numerical instability.

2.Aggressive Dimensionality Reduction:
The transition from the pooling layer (4096 dimensions) to the final output (2 dimensions) was too abrupt (4096 -> 512 -> 2). This "bottleneck" caused significant loss of fine-grained spatial features essential for identifying subtle pneumonia patterns.

3.Suboptimal Learning Rate:
A learning rate of 1e-3 was found to be too high for fine-tuning a pre-trained ResNet50. This caused the optimizer to overshoot the global minimum, leading to the observed "zig-zag" pattern in the loss curve.

4.Data Homogeneity:
The initial preprocessing only utilized basic resizing and cropping. Medical images, particularly X-rays, require higher variance (rotation/flipping) to help the model generalize across different patient positions.
