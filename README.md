
# Documentation: Deep Learning with TensorFlow

## 1. day2_DL_with_tensorflow.ipynb

**Description:** This notebook demonstrates logistic regression using TensorFlow for the Iris dataset.

**Code Snippet:**

Data loading and preprocessing:
- Loading the Iris dataset using `load_iris` from `sklearn.datasets`.
- Splitting the dataset into input features `x` and target labels `y`.
- Creating a DataFrame dataset to combine the features and labels.

Model building and training:
- Defining the model architecture using `tf.keras.layers.Dense`.
- Compiling the model with an optimizer, loss function, and metrics.
- Fitting the model to the training data using `model.fit`.

Prediction:
- Making predictions on new data using the trained model.

## 2. DL_with_TF_day3.ipynb

**Description:** This notebook introduces the concept of perceptrons and demonstrates the use of single-layer perceptrons for the Iris dataset.

**Code Snippet:**

Perceptron Model:
- Loading the Iris dataset using `load_iris` from `sklearn.datasets`.
- Splitting the dataset into input features `x` and target labels `y`.
- Creating a Perceptron model using `sklearn.linear_model.Perceptron`.
- Fitting the model to the training data using `model.fit`.

Multilayer Perceptron:
- Loading the MNIST dataset using `tf.keras.datasets.mnist.load_data`.
- Preprocessing the data by normalizing and reshaping.
- Defining a multilayer perceptron model using `tf.keras.models.Sequential` and `tf.keras.layers.Dense`.
- Compiling and fitting the model to the training data.
- Evaluating the model on the test data.

## 3. cnn_mnist.ipynb

**Description:** This notebook showcases the use of Convolutional Neural Networks (CNNs) for the MNIST handwritten digit classification task.

**Code Snippet:**

Data Preprocessing:
- Loading the MNIST dataset using `keras.datasets.mnist.load_data`.
- Reshaping and normalizing the data.

Model Architecture:
- Building a CNN model using `keras.layers.Convolution2D`, `keras.layers.MaxPooling2D`, and `keras.layers.Dense`.
- Compiling the model with appropriate loss and optimizer.

Training and Evaluation:
- Fitting the model to the training data using `model.fit`.
- Predicting on new images and displaying the results.

## 4. data_augmentation.ipynb

**Description:** This notebook demonstrates various data augmentation techniques using TensorFlow's `ImageDataGenerator` and `tf.image` functions to augment an image.

**Code Snippet:**

Importing Libraries and Uploading Image:
- Importing TensorFlow, Matplotlib, and the necessary modules from Keras and Google Colab.
- Uploading an image using `files.upload()` from `google.colab`.

Preprocessing the Image:
- Resizing and rescaling the image using Resizing and Rescaling layers from `tf.keras.preprocessing.image`.

Image Augmentation Techniques:
- Randomly flipping the image horizontally and vertically using `RandomFlip` from `tf.keras.preprocessing.image`.
- Randomly rotating the image by a specified angle using `RandomRotation` from `tf.keras.preprocessing.image`.
- Applying a random invert function to invert the image based on a given probability.
- Flipping the image upside down using `flip_up_down` from `tf.image`.
- Converting the image to grayscale using `rgb_to_grayscale` from `tf.image`.
- Adjusting the saturation of the image using `adjust_saturation` from `tf.image`.
- Adjusting the brightness of the image using `adjust_brightness` from `tf.image

`.

Using `ImageDataGenerator` for Augmentation:
- Installing the necessary library `keras_preprocessing`.
- Importing `ImageDataGenerator` from `keras.preprocessing.image`.
- Defining an instance of `ImageDataGenerator` with various augmentation settings.
- Reshaping the image to match the expected input shape.
- Generating augmented images using `flow` from `datagen` with a specified batch size.
- Displaying the augmented images using Matplotlib.

## 5. intro_to_tensorflow.ipynb

**Description:** This notebook provides an introduction to TensorFlow and covers various concepts and operations.

**Code Snippet:**

Importing TensorFlow:
- Importing TensorFlow and checking the version.

Eager Execution:
- Checking if eager execution is enabled.

GPU Check:
- Checking if a GPU is available.

Constant and Variable Tensors:
- Creating constant and variable tensors using `tf.constant` and `tf.Variable`.

Concatenation of Tensors:
- Concatenating tensors along rows and columns using `tf.concat`.

Zeros and Ones Tensors:
- Creating tensors filled with zeros and ones using `tf.zeros` and `tf.ones`.

Tensor Transpose:
- Transposing a tensor using `tf.transpose`.

Type Conversion:
- Converting the tensor data type using `tf.cast`.

Multiplication and Matrix Multiplication:
- Performing element-wise multiplication and matrix multiplication using `tf.multiply` and `tf.matmul`.

Determinant:
- Computing the determinant of a tensor using `tf.linalg.det`.

Identity:
- Creating an identity matrix using `tf.eye`.

Reshape:
- Reshaping a tensor using `tf.reshape`.

Keras:
- Importing the necessary modules from TensorFlow Keras.

Data Split:
- Splitting the CIFAR-10 dataset into training and testing sets using `cifar10.load_data()`.

Visualizing the Dataset:
- Visualizing a subset of images from the dataset using Matplotlib.

Text Vectorization:
- Performing text vectorization using `TextVectorization` layer from TensorFlow Keras.

Normalization:
- Normalizing the input data using the `Normalization` layer from TensorFlow Keras.