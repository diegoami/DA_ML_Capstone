# SPLIT LETSPLAY VIDEOS INTO SCENES

## BACKGROUND

For more of the background on this, check these [solutions.md](solutions.md) and [proposal.md](proposal.md).


## REQUIRED FILES

* Main repository on Github: _https://github.com/diegoami/DA_ML_Capstone_
* Companion project: _https://github.com/diegoami/DA_split_youtube_frames_s3.git_
* Data : _https://da-youtube-ml.s3.eu-central-1.amazonaws.com/_


## DOWNLOAD DATA

```
wget -nc https://da-youtube-ml.s3.eu-central-1.amazonaws.com/wendy-cnn/frames/wendy_cnn_frames_data_2b.zip
wget -nc https://da-youtube-ml.s3.eu-central-1.amazonaws.com/wendy-cnn/frames/wendy_cnn_frames_E67.zip
unzip -qq -n wendy_cnn_frames_data_2b.zip -d <YOUR SM_CHANNEL_TRAIN>
unzip -qq -n wendy_cnn_frames_67.zip -d <YOUR TEST DIR>

```

## SET ENVIRONMENT VARIABLES
Set up the _model_dir_ and _channel_train_ to point to where you pu

```
export SM_HOSTS=[]
export SM_CHANNEL_TRAIN=/media/diego/QData/youtube_ml/wendy-cnn-2b/frames/all
export SM_MODEL_DIR=/media/diego/QData/models/cnn-wendy/v2b-a
export SM_CURRENT_HOST=
```
## SET UP ENVIRONMENT

Set up a python environment using the libraries listed in `requirements-freezed.txt`

##  BASE DIRECTORY

Make sure that you `cd letsplay_classifier` before executing these scripts.

## TRAIN

This script creates a model to categorize the frame

```
python train.py --epochs=5 --img-width=320 --img-height=180 --layer-cfg=B --batch-size=16
```

## VERIFY_MODEL

This script verifies the model over the dataset and print a classification report and a confidence matrix over the whole dataset

```
python verify_model.py 
```

## SPLIT VIDEOS IN SCENES

To test how well the model is able to split videos into scenes, you can use the following command, to verify how the created model splits a video whose frames have not been used during training

```
PYTHONPATH=$(pwd) python interval/predict_intervals_endpoint.py --data-dir=/media/diego/QData/youtube_ml/wendy-cnn-2b/frames/E67/
```
## ON SAGEMAKER

To deploy the model on sagemaker, check the Juypter Notebooks, especially [CNN_Third_iteration.ipynb](CNN_Third_iteration.ipynb).