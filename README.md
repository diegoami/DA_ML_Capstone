# REQUIRED FILES

* Main repository on Github: _https://github.com/diegoami/DA_ML_Capstone_
* Companion project: _https://github.com/diegoami/DA_split_youtube_frames_s3.git_
* Data : _https://da-youtube-ml.s3.eu-central-1.amazonaws.com/_


## DOWNLOAD DATA

```
wget -nc https://da-youtube-ml.s3.eu-central-1.amazonaws.com/wendy-cnn/frames/wendy_cnn_frames_data_2.zip
unzip -qq -n wendy_cnn_frames_data_2.zip -d wendy_cnn_frames_data_2 
```

## SET ENVIRONMENT VARIABLES

```
export SM_HOSTS=[]
export SM_CHANNEL_TRAIN=/media/diego/QData/youtube_ml/wendy-cnn-2b/frames/all
export SM_MODEL_DIR=/media/diego/QData/models/cnn-wendy/v2b-a
export SM_CURRENT_HOST=
```
## SET UP ENVIRONMENT

Set up a python environment using the libraries listed in `requirements-freezed.txt`

## TRAIN

This script creates a model

```
python train.py --epochs=5 --img-width=320 --img-height=180 --layer-cfg=B --batch-size=16
```

## VERIFY_MODEL

This script verifies the model over the dataset

```
python verify_model.py 
```

