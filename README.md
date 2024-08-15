# Cat or Dog Image Classification

This project is a Convolutional Neural Network (CNN) based image classification model that differentiates between images of cats and dogs. The project utilizes TensorFlow and Keras to build, train, and evaluate the model on a dataset of cat and dog images. The code also includes preprocessing steps such as filtering out corrupted images, data augmentation, and data standardization.

## Project Overview

1. Data Preparation
   - The dataset used in this project is the "Kaggle Cats and Dogs" dataset.
   - The dataset is extracted and filtered to remove any corrupted images that may cause issues during training.

2. Data Augmentation
   - Data augmentation techniques, such as random horizontal flipping and rotation, are applied to the training images. This helps improve the model's ability to generalize to new data.

3. Model Architecture
   - A CNN model is built using Keras, which includes several layers:
     - Convolutional Layers: Extract features from the images.
     - Batch Normalization: Normalize the output of each layer to speed up training.
     - Separable Convolutional Layers: Improve computational efficiency while maintaining accuracy.
     - Residual Connections: Help prevent the degradation of the gradient and improve the flow of information through the network.
     - Global Average Pooling: Reduce each feature map to a single value, minimizing overfitting.
     - Dropout Layer: Reduce overfitting by randomly setting a fraction of input units to 0 at each update during training.
   - The output layer uses a sigmoid activation function to provide a binary classification output (cat or dog).

4. Model Training
   - The model is trained for 10 epochs using the Adam optimizer and binary cross-entropy loss function.
   - Checkpoints are saved after each epoch to allow for later recovery and analysis.

5. Prediction
   - After training, the model can be used to predict whether a new image is a cat or a dog. The prediction is expressed as a percentage likelihood of the image being a cat or a dog.

## How to Run the Project

1. Install Dependencies:
   Ensure you have Python and TensorFlow installed. You can install TensorFlow using pip:
   ```
   pip install tensorflow
   ```

2. Run the Script:
   Simply run the `CatOrDog.ipynb`. The script will:
   - Download and extract the dataset.
   - Preprocess and filter the data.
   - Augment the training data.
   - Build and train the CNN model.
   - Save model checkpoints after each epoch.
   - Make predictions on a sample image.

3. Prediction:
   After training, the script will load a sample image and predict whether it is a cat or a dog, printing the result to the console.

## Requirements

- Python
- TensorFlow
- Keras (included with TensorFlow)
- Matplotlib (for data visualization)

## Notes

- The model architecture is based on the Xception architecture but simplified for this binary classification task.
- The dataset is relatively small, so results may vary depending on the quality of the images and the specifics of the training process.
