# Emotion-detection
This repository contains a real-time face detection and emotion classification model. The face detection utilizes MTCNN and OpenCV, while the emotion classification model is based on a Convolutional Neural Network (CNN) architecture called VGGFace, with weights trained on the FER2013 dataset. This modification is built on top of Parkhi et. al 2023.

## The Model
The model operates on the VGGFace architecture, a CNN designed specifically for face recognition tasks.

<p align="center">
    <img src="https://raw.githubusercontent.com/travistangvh/emotion-detection-in-real-time/master/images/VGGFaceNetwork.jpg">
</p>

## Updated Features
- The emotion classification code has been optimized for better performance and readability.
- Real-time radar plot integration for visualizing the probabilities of different emotions.
- Improved face detection and emotion classification accuracy.
- Removal of the display of the cropped image from the webcam feed.
- Added a radar plot with a circular border and emotion labels, positioned center-left on the video frame.

## Instructions on getting started
### To run the demo.
* Clone this commit to your local machine using `git clone https://github.com/travistangvh/emotion-detection-in-real-time.git`

* Install these dependencies with pip install 
`pip install -r ../REQUIREMENTS.txt`

* Download pretrained model and weight `trained_vggface.h5` from [here](https://drive.google.com/file/d/1Wv_Z4lAa7BgYqSAeceK9TxJNfwoLTwKy/view?usp=sharing).

* Place `trained_vggface.h5` into `../datasets/trained_models/`.

* Run `emotion_webcam_demo.py` using `python3 emotion_webcam_demo.py`

### To train previous/new models for emotion classification:

* Download the fer2013.tar.gz file from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

* Move the downloaded file to the `../datasets/raw/` directory inside this repository.

* Untar the file:
`tar -xzf fer2013.tar`

* Ensure that the file `../datasets/raw/fer2013.csv` exists

* Run the `training_emotion_classification.py` file
`python3 training_emotion_classifier.py`

# Citations
* [Deep Face Recognition](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf) by Parkhi et. al.
