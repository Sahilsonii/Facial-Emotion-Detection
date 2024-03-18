# Facial Emotion Detection

This repository contains a project for real-time facial emotion detection using OpenCV and a Convolutional Neural Network (CNN) model trained with Keras. The system can be used in two modes: training the model on a dataset of facial expressions and displaying the detected emotions in real-time using a webcam feed.

[![Picture5.png](https://i.postimg.cc/nzqysW22/Picture5.png)](https://postimg.cc/phXG4qg5)

## Table of Contents

- [Initial Setup](#initial-setup)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Approach](#approach)
- [Repository Files Navigation](#repository-files-navigation)

## Initial Setup

To get started with this project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Facial-Emotion-Detection.git
   ```
2. Navigate to the project directory:
   ```
   cd Facial-Emotion-Detection
   ```
3. Install required dependencies:
   - Option 1: Use `pip install -r requirements.txt`.
   - Option 2: Install dependencies individually:
     ```
     pip install tensorflow
     pip install opencv-python
     pip install matplotlib
     ```
4. Obtain the Haar cascade XML file for face detection:
   - Download the `haarcascade_frontalface_default.xml` file from the [OpenCV GitHub repository](https://github.com/opencv/opencv/tree/master/data/haarcascades).

## Dependencies

- **TensorFlow**: A powerful library for machine learning and deep learning.
- **OpenCV**: An open-source computer vision library used for image and video processing.
- **Matplotlib**: A plotting library used for visualizing the model's accuracy and loss over epochs.

## Usage

To train the model, use:
```
python emotions.py --mode train
```

To view the predictions without training again, you can download the pre-trained model from the repository and then run:
```
python emotions.py --mode display
```

## Approach

The project uses a CNN model for emotion detection. The model is trained on a dataset of facial expressions, and the trained model is used to predict emotions in real-time from a webcam feed. The approach involves:

- Capturing video from the webcam.
- Detecting faces within the video stream using a Haar Cascade Classifier.
- Preprocessing the detected faces to match the input size expected by the model.
- Making emotion predictions using the trained model.
- Displaying the predicted emotions on the video frames in real-time.

## Repository Files Navigation

- `src`: Contains the main script `emotions.py` for running the emotion detection system.
- `data`: Contains the dataset for training and testing the model.
- `haarcascade_frontalface_default.xml`: The Haar cascade XML file for face detection.
- `model.h5`: The pre-trained model weights.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you find any bugs or have suggestions for improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
