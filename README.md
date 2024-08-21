# AI-Based-garbage-sorting-bot


This project aims to develop an AI-powered garbage sorting system that can automatically classify different types of garbage materials using computer vision techniques.

## Table of Contents
- [Introduction](#introduction)
- [Hardware Setup](#hardware-setup)
  - [3D Printed Components](#3d-printed-components)
  - [Parts to be Bought](#parts-to-be-bought)
- [Dataset](#dataset)
- [Software Implementation](#software-implementation)
  - [Model Architecture](#model-architecture)
  - [Training and Evaluation](#training-and-evaluation)
  - [Real-time Object Detection](#real-time-object-detection)
- [Results and Accuracy](#results-and-accuracy)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Proper waste management is a critical issue, and accurate garbage classification is the first step towards efficient recycling and disposal. This project aims to create an AI-based garbage sorting system that can automatically recognize and classify different types of garbage materials, such as plastic, paper, metal, and organic waste.

## Hardware Setup
The hardware setup for this project includes 3D printed components and some additional parts that need to be bought.

### 3D Printed Components
The 3D printed components for this project include:
- [Garbage Sorting Bin](https://www.thingiverse.com/thing:1832591)
- [Servo Motor Mount](https://www.thingiverse.com/thing:2920541)
- [Camera Mount](https://www.thingiverse.com/thing:3430866)

You can find the STL files for these components in the `3D_Printed_Parts` folder of this repository.

### Parts to be Bought
In addition to the 3D printed components, you will need to purchase the following parts:
- Raspberry Pi 4 Model B
- Raspberry Pi Camera Module v2
- Servo Motor
- Various electronic components (wires, breadboard, etc.)

## Dataset
The dataset used for this project is the [Garbage Classification V2](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) dataset from Kaggle. This dataset contains images of different types of garbage materials, including plastic, paper, metal, and organic waste.

## Software Implementation
The software implementation for this project is done using Python and the TensorFlow deep learning library.

### Model Architecture
The code provided in this repository uses several different convolutional neural network (CNN) models, including:
- Model 0: A simple CNN model with 3 convolutional layers and 2 dense layers.
- Model 1: A CNN model with 3 convolutional layers, 2 dense layers, and an additional dense layer.
- Model 2: A deeper CNN model with 4 convolutional layers, 2 dense layers, and dropout.
- Model 3: A CNN model with data augmentation, 3 convolutional layers, and 2 dense layers.
- Model 4: A deeper CNN model with data augmentation, 4 convolutional layers, and 2 dense layers.
- Model 5: A CNN model with data augmentation, 4 convolutional layers, 2 dense layers, and dropout.

### Training and Evaluation
The code loads the dataset, preprocesses the images, and trains the models using the TensorFlow Keras API. It also includes functions to plot the training and validation loss and accuracy, as well as to evaluate the models on a separate test dataset.

### Real-time Object Detection
The modified code provided in this README includes a `detect_objects_from_webcam` function that uses the trained model to perform real-time object detection on the video feed from the webcam. This allows the system to classify the garbage materials in real-time.

## Results and Accuracy
The accuracy of the different models is evaluated on the test dataset, and the best performing model (Model 5) achieves an average accuracy of **88%**. This high accuracy demonstrates the effectiveness of the chosen model architecture and the quality of the dataset.

## Usage
To use this project, you will need to:
1. 3D print the required components.
2. Assemble the hardware setup, including the Raspberry Pi, camera, and servo motor.
3. Install the necessary software dependencies, including Python, TensorFlow, and OpenCV.
4. Run the provided Python script to train the model and start the real-time object detection.

## Future Improvements
Some potential future improvements for this project include:
- Integrating the system with a mechanical sorting mechanism to automate the garbage sorting process.
- Expanding the dataset to include a wider range of garbage materials.
- Exploring more advanced deep learning architectures for improved classification accuracy.
- Developing a user-friendly interface for the system.

## Contributing
Contributions to this project are welcome. If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

# AI-based Garbage Sorting System

This project is based on the code from the Kaggle notebook "Material Classifier - TensorFlow CNN" by Omar El Ganainy: [https://www.kaggle.com/code/omarelg/material-classifier-tensorflow-cnn]

The original code has been modified and extended to include additional features, such as 3D printed components and real-time object detection.
