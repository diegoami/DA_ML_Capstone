# CAPSTONE Project - Machine Learning Engineer Nanodegree

October 2020

## PROBLEM DEFINITION

### BROAD CONTEXT 

During the last few years it has become more and more common to stream on platforms such as Youtube and Twitch while playing video games, or to upload recorded sessions. The volume of videos produced is overwhelming. In many of the videos games being streamed there are different types of scenes. Both for content producers and consumers it would be useful to be able to automatically split videos, to find out in what time intervals different types of scenes run. For instance, having as an input the video recording of a Minecraft speedrun, we could be able to produce the time intervals when the game is taking place in the Overworld surface, in caves, in the Nether and the End respectively - the four main settings of this game.

### PROJECT LOCATION & FILES

* Main repository on Github: _https://github.com/diegoami/DA_ML_Capstone_
* Companion project: _https://github.com/diegoami/DA_split_youtube_frames_s3.git_
* Data : _https://da-youtube-ml.s3.eu-central-1.amazonaws.com/_


### PROJECT SCOPE 


 
The game that I have chosen to analyze is _Mount of Blade: Warband_, of which I made several walkthroughs. In this game, I have identified seven types of scenes to which an image belongs:

* BATTLE: any battle taking place in an open field or in a village 
* TOURNAMENT: Tournaments in arena 
* HIDEOUT: the warband assaults a bandit hideout 
* TRAINING: the hero trains in a field or in a village (later remapped to OTHER)
* SIEGE: a town is sieged 
* TRAP: hero falls into a trap and must fight their way out (later remapped to BATLE)
* TOWN (escape): escape from the town or castle prison (later remapped to BATTLE)
* OTHER : everything else 

To create a dataset I took some videos from a game walkthrough of mine, the adventures of Wendy. I used the episodes from 41 to 66 from following public playlists on youtube: 

* CNN-Wendy-I: _https://www.youtube.com/playlist?list=PLNP_nRm4k4jfVfQobYTRQAXV_uOzt8Bov_
* CNN-Wendy-II: _https://www.youtube.com/playlist?list=PLNP_nRm4k4jdEQ-OM31xNqeE64svvx-aT_ 

These are the episodes I went through and manually split into scenes. I wrote down how they were split in the video description on youtube. 

For instance, in episode 54, I have identified following scenes, of the category "Hideout", "Battle", "Tournament", "Town". All the other parts of the video are categorized as "Other". These lines can be found in the video description.  

- 09:51-12:21 Hideout Tundra Bandits (Failed)
- 18:47-19:44 Battle with Sea Raiders
- 20:50-21:46 Battle with Sea Raiders
- 22:54-23:42 Battle with Sea Raiders
- 34:06-37:44 Tournament won in Tihr
- 38:46-40:48 Town escape for Boyar Vlan 

To prepare the data set, I had set up a companion project under _https://github.com/diegoami/DA_split_youtube_frames_s3/tree/support_playlists_:
 
This project:
- Downloads the relevant videos from youtube, using the youtube-dl python library, in a 640x360 format
- Extract at every two seconds a frame and save it an jpeg file, using the opencv python library, resizing to the practical format 320x180
- Download the text from the youtube description and save it along the video ( _metadata_ )
- Distribute the files over directories named by the categories.

This way, I created first a dataset that I uploaded to a S3 bucket: _https://da-youtube-ml.s3.eu-central-1.amazonaws.com/wendy-cnn/frames/wendy_cnn_frames_data.zip_ (3.2 Gb) and made public.


## PROBLEM ANALYSIS

### DATASET ANALYSIS

The amount of images I had  generated first, in this way,  was 45718, split in eight categories. 

This was the breakdown of the images I collected over the 8 classes I mentioned above 

* BATTLE: ~13%
* TOURNAMENT: ~14.8%
* HIDEOUT: ~2.5%
* TRAINING: ~1.7%
* SIEGE: ~0.4%
* TRAP: ~0.2%
* TOWN (escape): ~0.2%
* OTHER : ~67.6%

As it can be seen, some categories have few samples, so it was going to be expected that we could have trouble with those
A sanity check whether the images are in the correct directory can be done using the notebook _analysis.ipynb_.

For instance battle images should look like that:

![Battle images](docimages/battle-images.png)


### FIRST ITERATION

Notebook: _Wendy_CNN.ipynb_

The simplest way I chose to verify whether a model is viable was to start and set up a Convolutional Neural Network in Pytorch, as I was pretty sure that 

* this was pretty much the most sensible way to approach the problem
* I could use standard CNN topologies available in Pytorch
* analyzing the result of the model would give me more information on what I would have to be looking for

Convolutional Neural Network are, as a matter of fact, a very standard approach for categorizing images. A simple to use and flexible topology I decided to use was VGG, which is included in the Pytorch library. 

As the images extracted from game walkthrough are not related to real world images, using a pretrained net and expand it with transfer learning did not seem to make sense. Instead, I opted for a full train.
In the preprocessing phase, in this iteration, I resized images to 128 x 72, which should be enough for the algorithm to recognize features ( original images were all 640 x 360). As I already have over 45000 images, I thought I would do not need any kind of data augmentation (like, use mirrored images), also because it is not given that the game may actually show mirrored images.

The flaw in the dataset, regrettably, is that some categories, such as *Siege*, *Trap* and *Town*, have relatively few samples. Looking at the confidence matrix I got, there were however some surprises in the report I got when I decided to test the produced model with 10% of the dataset. 


Confidence Matrix
 
| X| 0    | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 
|--|------|-----|-----|-----|-----|-----|-----|-----|
| 0|  5671|    9|   93|    1|   19|    0|    1|    0|
| 1|    59| 1041|   49|    1|    3|    0|    0|    0|
| 2|   315|   19|30312|    1|  156|    0|   77|    0|
| 3|    26|    2|    2|  164|    0|    0|    0|    0|
| 4|    67|    7|  290|    1| 6401|    0|    4|    0|
| 5|    17|    0|   40|    0|    0|    3|    0|    0|
| 6|     8|    1|  175|    0|   49|    0|  537|    0|
| 7|    13|    1|   62|    0|    1|    4|    0|    8|
  



  |class| class_name |precision|recall |f1-score|   support |
  |-----|------------|---------|-------|--------|---------- |
  | 0   | Battle     |  0.92   |0.98   | 0.95   |       5974|
  | 1   | Hideout    |  0.96   |0.90   | 0.93   |       1153|
  | 2   | Other      |  0.98   |0.98   | 0.98   |      30880|
  | 3   | Siege      |  0.98   |0.85   | 0.91   |        194|
  | 4   | Tournament |  0.97   |0.95   | 0.98   |       6770|
  | 5   | Trap       |  0.43   |0.05   | 0.09   |         60|
  | 6   | Training   |  0.87   |0.70   | 0.77   |        770|
  | 7   | Town       |  1.00   |0.09   | 0.16   |        89 |

It turned out that the *Siege* class is not that big a problem (as a matter of fact, images belonging to this category are pretty distinctive). However, the classes *Trap*, *Town* and *Training* tended to be misclassified often. After checking the confidence matrix, I decided that it would make sense to remove these three categories, so that the category *Training* is classified as Other (Training is not interesting anyway) while Trap and Town are classified as Battle.

### SECOND ITERATION

First, I make sure to create a second dataset, where I map Trap and Town to Battle, and Training to Other. So that I end up with 5 categories:

* Battle      : 5943 (13.0%) 
* Hideout     : 1153 (2.5%)
* Other       : 31650  (69.2%)
* Siege       : 194 (0.4%)
* Tournament  : 6770 (14.8%)

TOTAL : 45710

I chose a smaller format for the images I save, as 640x480 is  too big for any model I can realistically train. The dataset becomes therefore much smaller: 1.0 Gb and can be found at _https://da-youtube-ml.s3.eu-central-1.amazonaws.com/wendy-cnn/frames/wendy_cnn_frames_data_2.zip_

I also added a preprocessing step to correct some of the wrongly classified images that are in the dataset, and that I discovered using a simple self-made tool. 

Now, creating a basic VGG net (type B) on the full images, having image_height x image_width = 160 x 90, with 5 epochs, and just 5 categories, and then running the model on the full dataset, gives this result:

Avg acc (test): 0.9915

Confidence Matrix

| X| 0    | 1   | 2   | 3   | 4   | 
|--|------|-----|-----|-----|-----|
| 0|  6077|    1|   43|    0|    6|
| 1|    19| 1128|    4|    4|    2|    
| 2|   135|   10|31405|    1|   23|    
| 3|     3|    1|    0|  189|    0|    
| 4|     9|    1|  122|    3| 6524|    

|class|precision | recall | f1-score |support|
|-----|----------|--------|----------|-------|
|    0|      0.97|    0.99|      0.98|   6127|
|    1|      0.99|    0.97|      0.98|   1157|
|    2|      0.99|    0.99|      0.99|  31574|
|    3|      0.96|    0.98|      0.97|    193|
|    4|      1.00|    0.98|      0.99|   6659|

which is a much better result than the first run. I decided that I could keep this model.

## IMPLEMENTATION

I set up scripts and notebooks so that they would work both locally and on Sagemaker, if b. However, some things work better locally, while some other work better on Sagemaker.

A pytorch/conda environment, as the one in Sagemaker, is assumed - the missing libaries from the default sagemaker conda pytorch environment are in the _/requirements.txt_ file.

The code root directory is letsplay_classifier - scripts should be executed from this  directory, or the directory should be included in PYTHONPATH.

### REQUIRED ENVIRONMENT VARIABLES

All scripts require following environment variables, which are the ones required by Sagemaker containers.

* SM_CHANNEL_TRAIN: location of data - the directory where you unzipped the required data
* SM_MODEL_DIR: where to save the model 
* SM_HOSTS: should be "[]"
* SM_CURRENT_HOST: should be ""

### TRAINING SCRIPT

The training script ,at _train.py_ accepts following arguments:
* img-width: width to which resize images
* img-height: height to which resize images
* epochs: for how many epochs to train
* batch-size: size of the batch while training
* layer-cfg: what type of VGG net to use

These are the steps that are executed:

* preprocessing to  to move misclassifed frames to their correct directory  
* use an image loader from pytorch to create a generator scanning all files in the data directory.
* use a pytorchvision transformer to resize images
* divide the dataset in train and validation sets, using stratification and shuffling
* load a VGG neural network, modified so that the output layers produce a category from our domain (5 in total in the final version)
* For each epoch, execute a training step and evaluation step, trying to minimize the cross entropy loss in the validation set
* Save the model so that it can be further used by the following steps
 
The cross entropy is the most useful metrics while training a  classifier with C classes, therefore it is used here.

### VERIFICATION SCRIPT

The verification script  _verify_model.py_ works only locally, as it assumes the model and the dataset is saved locally from the previous step. It requires the same environment variables as the training script.

* Loads the model created in the previous step
* Walks through all the images in the dataset, one by one, and retrieve the predicted label
* Print average accuracy, a classification report based on discrepancies, and a confidence matrix
* Save discrepancies in a _misclassified_ file, that can be used later in a preprocessing step 
 

### MISCLASSIFIED IMAGES

To find out about images in the training / validation set that I might have misclassified, I created a simple graphical tool under  _letsplay_classifier/requirements.txt_ . 

For this, I use a list of images that were misclassified (_misclassified.json_) produced by the script _verify_model.py_ . Using a small GUI in the file _sel_image.py_ I try and sort between images that have a very high probability of being classified wrongly, and write down the images  where I reject the expected label for the predicted one under _rejected.json_. All of t



### MODEL DEPLOY

### ENDPOINT CALL
The way we exchange data between the client and the service could be improved, as the performance and the time needed to get results on a batch of images is considerable.

## RESULTS

This is the classification report of the selected model. Avg accuracy is 0.98. Classes having enough samples get a F1-value of 0.97, while classes not having enough samples have much worse results. To get better values on this, we would definitely need more samples. Oversampling or working with weight might not give results which are good enough, as the variance between images is too great. 	

 
## CONCLUSIONS

This project proved to me that it is possible to reliably build a classification model for images. I could apply this technique also to other video games, as the breakdown in scenes is something that is common.

The next step would be
I am planning to use this tool to build a client that would take a video as input, extract frames every two seconds and suggest the splitting of videos in sequences. 


