cd letsplay_classifier
export SM_HOSTS=[]
export SM_CHANNEL_TRAIN=/media/diego/QData/youtube_ml/wendy-cnn-3/frames/all
export SM_MODEL_DIR=/media/diego/QData/models/cnn-wendy/v3
export SM_CURRENT_HOST=
python train.py --epochs=6 --img-width=320 --img-height=180 --layer-cfg=B --batch-size=16
