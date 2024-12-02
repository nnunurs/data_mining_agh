# Fashion Classification Neural Network

A neural network implementation for classifying fashion items using PyTorch. The project consists of training a neural network on the Fashion MNIST dataset and using it to classify real fashion images.

## Project Structure

- `model.py` - Neural network model definition
- `train_network.py` - Training script for the neural network
- `predict.py` - Script for making predictions on new images
- `test_network.py` - Unit tests for the project

## Files Description

### model.py
Contains the neural network architecture definition. The network consists of fully connected layers designed to classify fashion items into 10 categories.

**Key Components:**
- `NeuralNetwork` class: Implementation of a feed-forward neural network
- Network architecture: 784 -> 512 -> 256 -> 10 neurons
- ReLU activation functions between layers

### train_network.py
Handles the training process of the neural network using the Fashion MNIST dataset.

**Features:**
- Loads and preprocesses the Fashion MNIST dataset
- Implements training and validation loops
- Saves the trained model to 'fashion_model.pth'
- Displays training progress and loss curves
- Uses SGD optimizer with momentum

**Training Parameters:**
- Learning rate: 0.01
- Momentum: 0.9
- Epochs: 10
- Batch size: 64

### predict.py
Provides functionality for making predictions on new fashion images.

**Key Functions:**
- `predict_image(image_path)`: Processes a single image and returns predictions
- `process_folder(folder_path, sample_size)`: Processes multiple images from a folder
- `display_predictions(image_path, predicted_class, top_predictions)`: Visualizes results

**Features:**
- Image preprocessing (resizing, normalization)
- Batch processing of images
- Progress bar for bulk processing
- Results saving to text file
- Interactive visualization of results

### test_network.py
Contains unit tests for verifying the functionality of the entire system.

**Test Cases:**
- Neural network structure validation
- Image prediction functionality
- Folder processing
- Model saving and loading
- Error handling
- Input/output validation

## Usage

1. Training the model:

```bash
python train_network.py
```

2. Making predictions:

```bash
python predict.py
```

3. Running tests:

```bash
python test_network.py
```

## Dependencies

- PyTorch
- torchvision
- PIL (Python Imaging Library)
- numpy
- matplotlib
- tqdm

## Classes

### Fashion Categories
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Error Handling

The system includes comprehensive error handling for:
- Missing model files
- Invalid image formats
- Incorrect image dimensions
- Memory issues
- File system errors

## Testing

The test suite covers:
- Model architecture verification
- Prediction accuracy
- File handling
- Error cases
- Input/output validation

## Performance

The model achieves approximately 85-90% accuracy on the Fashion MNIST test set after 10 epochs of training.

## Notes

- The model expects grayscale images resized to 28x28 pixels
- Results are saved in a 'results' directory
- The system supports common image formats (PNG, JPG, JPEG, BMP, GIF)