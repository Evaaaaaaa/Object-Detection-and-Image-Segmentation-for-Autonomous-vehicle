# Image-Recognition-for-Autonomous-Driving
This project effectively built and trained a model that can produce a top-down view of the surrounding area of an ego car by

(1) locating and classifying objects on the road from images captured by six cameras on the car. <br>
(2) generating the road map layout by semantic image segmentation.

Here is my [prensentation](https://docs.google.com/presentation/d/1tINbCsJSMcSCokd-kKIEKPDE8lH12jwAx99CUmpx5ZQ/edit?usp=sharing).

## Folders
- **configs** : Enable modification of all preset variables within single directory (consisting of one or many config files for separate tasks)
- **data** : Include example a small amount of data in the Github repository so tests can be run to validate installation

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

## Training
1. Please run train.py to generate two model paramter files.
3. A pretraining script on Resnet18 is provided, but is not used by default.
4. Model state_dict is not provided. Feel free to play around with different model parameter settings in *config*.
