# SPLIT LETSPLAY VIDEOS INTO SCENES

## BACKGROUND

For more of the background on this project, check these [solutions.md](solutions.md) and [proposal.md](proposal.md).


## REQUIRED FILES

* Main repository on Github: _https://github.com/diegoami/DA_ML_Capstone_
* Companion project: _https://github.com/diegoami/DA_split_youtube_frames_s3.git_
* Data : _https://da-youtube-ml.s3.eu-central-1.amazonaws.com/_


## DOWNLOAD DATA

```
wget -nc https://da-youtube-ml.s3.eu-central-1.amazonaws.com/wendy-cnn/frames/wendy_cnn_frames_data_5.zip
unzip -qq -n wendy_cnn_frames_data_5.zip -d <YOUR SM_CHANNEL_TRAIN>


wget -nc https://da-youtube-ml.s3.eu-central-1.amazonaws.com/wendy-cnn/frames/wendy_cnn_frames_E69.zip
unzip -qq -n wendy_cnn_frames_69.zip -d <YOUR TEST DIR>
## Optional: repeat the two above steps for episodes E71, E72, E73, E74, E75, E76, E77
```


## SET ENVIRONMENT VARIABLES

Set up the _model_dir_ and _channel_train_ to point to where you put the files you downloaded, and where you want to keep the generated models, respectively

```
export SM_HOSTS=[]
export SM_CHANNEL_TRAIN=<YOUR-DATA-DIR>
export SM_MODEL_DIR=<YOUR-MODEL-DIR>
export SM_CURRENT_HOST=
```
## SET UP ENVIRONMENT

Set up a python environment using the libraries listed in `requirements-freezed.txt`

##  BASE DIRECTORY

Make sure that you `pushd letsplay_classifier` before executing scripts locally, and set PYTHON_PATH to the current directory.

## BENCHMARK MODEL
 
```
pushd letsplay_classifier
PYTHONPATH=$(pwd) python pca/pca_sklearn.py --data-dir=/media/diego/QData/youtube_ml/wendy-cnn-5/frames/all/
popd
``` 
 

## TRAIN

This script creates a model to categorize frames in the train data directory

```
pushd letsplay_classifier
PYTHONPATH=$(pwd) python train.py --epochs=5 --img-width=320 --img-height=180 --layer-cfg=B --batch-size=16
popd
```

## VERIFY_MODEL

This script verifies the model over the dataset and print a classification report and a confidence matrix over the whole dataset

```
pushd letsplay_classifier
PYTHONPATH=$(pwd) python verify_model.py 
popd
```

## VISUALIZE MODEL

You can visualize the model and see how principal components of data set are spread.

```
pushd letsplay_classifier
PYTHONPATH=$(pwd) python pca/pca_vgg.py 
```


## SPLIT VIDEOS IN SCENES

To test how well the model is able to split videos into scenes, you can use the following command, to verify how the created model splits a video whose frames have not been used during training.
*Data-dir* here is the directory of the images that you want to classify.

```
pushd letsplay_classifier
PYTHONPATH=$(pwd) python interval/predict_intervals_walkdir.py --data-dir=<TEST-DIR>
popd
```

## ON SAGEMAKER

To deploy the model on sagemaker, check the Juypter Notebooks, especially [CNN_Third_iteration.ipynb](CNN_Third_iteration.ipynb), and follow the instructions on them.