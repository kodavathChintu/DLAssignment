# DLAssignment
# CIFAR-10 Image Classification

This project implements a feedforward neural network using TensorFlow and Keras to classify images from the CIFAR-10 dataset. The model is flexible, allowing customization of layers, units, activation functions, optimizers, and weight initialization methods.

## Dataset
The CIFAR-10 dataset consists of 60,000 color images (32x32 pixels) across 10 categories:

- Airplane
- Car
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset is split into:
- **50,000** training images (with 10% used for validation)
- **10,000** test images

## Features
- **Customizable Neural Network**: Configurable number of layers, units, activation functions, and optimizers.
- **Hyperparameter Tuning**: Multiple configurations are tested to determine the best model.
- **Performance Evaluation**: Includes test accuracy comparison and confusion matrix visualization.
- **Loss Function Comparison**: Compares Cross-Entropy Loss with Mean Squared Error (MSE) Loss.
- **Visualization**: Sample image display and heatmap for confusion matrix.

## Installation
Ensure you have Python installed along with the necessary libraries:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

## Running the Code
To execute the script, simply run:

```bash
python cifar10_classifier.py
```

## Model Customization
You can modify the following hyperparameters:
- **Layers**: Number of hidden layers.
- **Units**: Neurons per layer.
- **Activation Function**: (`relu`, `sigmoid`, etc.).
- **Optimizer**: (`adam`, `sgd`, `nesterov`, etc.).
- **Learning Rate**: Optimizer learning rate.
- **Weight Initialization**: (`he_uniform`, `glorot_normal`, etc.).

Example Configuration:
```python
configs = [
    {"layers": 4, "units": 128, "activation_fn": "relu", "opt_choice": "adam", "learning_rate": 2e-3, "init_method": "glorot_normal"},
    {"layers": 3, "units": 64, "activation_fn": "relu", "opt_choice": "sgd", "learning_rate": 5e-3, "init_method": "random_uniform"},
]
```

## Results & Evaluation
After training, the script:
1. Evaluates the model on the test set.
2. Displays a confusion matrix.
3. Compares Cross-Entropy Loss with MSE Loss.
4. Prints recommended hyperparameters for the MNIST dataset.

Example Output:
```
Config: {'layers': 4, 'units': 128, 'activation_fn': 'relu', 'opt_choice': 'adam', 'learning_rate': 0.002, 'init_method': 'glorot_normal'} -> Test Accuracy: 0.7215
Best Model Test Accuracy: 0.7215
Cross Entropy Accuracy: 0.7215, Mean Squared Error Accuracy: 0.6712
```

## Next Steps
- Implement **Dropout** and **Batch Normalization** to improve generalization.
- Use **Data Augmentation** for better performance.
- Extend the project to a **CNN model** for higher accuracy.



