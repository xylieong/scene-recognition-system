# 62322 Scene Recognition System
The 62322 new scene recognition system uses Yolo v3 object detection model as feature extractor and MLP as scene classifier. Evalution is conducted via k-fold validation method.


# SECTION 1: Yolo v3 Object Detection with Tensorflow 2.0
Yolo v3 is an algorithm that uses deep convolutional neural networks to detect objects. <br> <br>

## Getting started

### Prerequisites
This project is written in Python 3.7 using Tensorflow 2.0 (deep learning), NumPy (numerical computing), Pillow (image processing), OpenCV (computer vision) and seaborn (visualization) packages.

```
pip install -r requirements.txt
```

### Download Database
Download database (folders images3, images4) from Ong_62322_database folder

### Downloading official pretrained weights
For Linux: Let's download official weights pretrained on COCO dataset. 

```
wget -P weights https://pjreddie.com/media/files/yolov3.weights
```
For Windows:
You can download the yolov3 weights by clicking [here](https://pjreddie.com/media/files/yolov3.weights) and adding them to the weights folder.

### Using Custom trained weights
Get the info in README_YOLO.md
  
### Save the weights in Tensorflow format
Get the info in README_YOLO.md

## Running the model
Get the info in README_YOLO.md

## To-Do List
* Finish migration to full TF 2.0 (remove tf.compat.v1)
* Run 62322_fyp2.ipynb for Yolo v3 object detection
* Numbers in console indicates progress

## Acknowledgments
Get the info in README_YOLO.md


# SECTION 2: MLP as scene classifier
Multilayer perceptron (MLP) is a simple artificial neural network (ANN) model that is capable of carrying out classifications.

## To-Do List
* Make sure that ./detections/detection.csv is saved with [80 attributes][1 scene label]
* Run mlp.ipynb for MLP model training and testing


# FOLDERS

## ./data
Images sources

### ./data/images3
Experiment 1 image input

### ./data/images4
Experiment 2 image input

## GridSearch.ipynb
Grid search cv method that helps find the best parameters among all classifier