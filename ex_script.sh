
export SM_HOSTS=[]
# replace with your data
 export SM_CHANNEL_TRAIN=/media/diego/QData/youtube_ml/wendy-cnn-5/frames/all
export SM_MODEL_DIR=/media/diego/QData/models/cnn-wendy/v5n
export SM_CURRENT_HOST=
pushd letsplay_classifier
##PYTHONPATH=$(pwd) python pca/pca_sklearn.py --img-width=160 --img-height=90 --n-components=100
##PYTHONPATH=$(pwd) python train.py --epochs=2 --img-width=160 --img-height=90 --layer-cfg=B --batch-size=16
##PYTHONPATH=$(pwd) python verify_model.py
##PYTHONPATH=$(pwd) python pca/pca_vgg.py
##PYTHONPATH=$(pwd) python interval/predict_intervals_walkdir.py --data-dir=/media/diego/QData/youtube_ml/wendy-cnn-5/frames/E69
popd