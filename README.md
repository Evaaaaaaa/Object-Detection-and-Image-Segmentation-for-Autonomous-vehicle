# Image-Recognition-for-Autonomous-Driving
This project effectively built and trained a model that can produce a top-down view of the surrounding area of an ego car by

(1) locating and classifying objects on the road from images captured by six cameras on the car. <br>
(2) generating the road map layout by semantic image segmentation.

Here is my [prensentation](https://docs.google.com/presentation/d/1tINbCsJSMcSCokd-kKIEKPDE8lH12jwAx99CUmpx5ZQ/edit?usp=sharing).

## Folders
- **configs** : Enable modification of all preset variables within single directory (consisting of one or many config files for separate tasks)

## Setup
Clone repository and update python path
```
repo_name=Image-Recognition-for-Autonomous-Driving.git
username=Evaaaaaaa
git clone https://github.com/$username/$repo_name
cd $repo_name
echo "export $repo_name=${PWD}" >> ~/.bash_profile
echo "export PYTHONPATH=$repo_name/src:${PYTHONPATH}" >> ~/.bash_profile
source ~/.bash_profile
```
## Data
Data could be downloaded at [data](https://drive.google.com/file/d/1oq83pFKNxrz-1E06_4YMazTABhSqY5su/view?usp=sharing).

The data folder contains two datasets:

 1. an unlabeled dataset for pre-training<br>
 2. a labeled dataset for both training and validation<br>
 
### The dataset is organized into three levels: scene, sample and image
 1. A scene is 25 seconds of a car's journey.<br>
 2. A sample is a snapshot of a scene at a given timeframe. Each scene will be divided into 126 samples, so about 0.2 seconds between consecutive samples.<br>
 3. Each sample contains 6 images captured by camera facing different orientation. Each camera will capture 70 degree view. To make it simple, you can safely assume that the angle between the cameras is 60 degrees.

There are 106 scenes in the unlabeled dataset and 28 scenes in the labeled dataset.

## Training
1. Please run train.py to generate two model paramter files.
3. A pretraining script on Resnet18 is provided, but is not used by default.
4. Model state_dict is not provided. Feel free to play around with different model parameter settings in *config*.

## Evaluation
The model's performance for binary road map is evaluated by using the average threat score (TS) across the test set:
$$\text{TS} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$$

The model's performance for object detection is evaluated by using the average mean threat score at different intersection over union (IoU) thresholds.<br>
There will be five different thresholds (0.5, 0.6, 0.7, 0.8, 0.9). For each thresholds, we will calculate the threat score. <br>The final score will be a weighted average of all the threat scores:
$$\text{Final Score} = \sum_t \frac{1}{t} \cdot \frac{\text{TP}(t)}{\text{TP}(t) + \text{FP}(t) + \text{FN}(t)}$$

