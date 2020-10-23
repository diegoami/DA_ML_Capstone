# CAPSTONE Project - Machine Learning Engineer Nanodegree

October 2020

## PROBLEM DEFINITION

### BROAD CONTEXT 

During the last few years it has become more and more common to stream on platforms such as Youtube and Twitch while playing video games, or to upload recorded sessions. The volume of videos produced is overwhelming. In many of the videos games being streamed there are different types of scenes. Both for content producers and consumers it would be useful to be able to automatically split videos, to find out in what time intervals different types of scenes run. For instance, having as an input the video recording of a Minecraft speedrun, we could be able to produce the time intervals when the game is taking place in the Overworld surface, in caves, in the Nether and the End respectively - the four main settings of this game.

### PROJECT SCOPE 
 
The game that I have chosen to analyze is _Mount of Blade: Warband_, of which I made several walkthroughs. In this game, I have identified seven types of scenes to which an image belongs:

* BATTLE: any battle taking place in an open field or in a village 
* TOURNAMENT: Tournaments in arena 
* HIDEOUT: the warband assaults a bandit hideout 
* TRAINING: the hero trains in a field or in a village 
* SIEGE: a town is sieged 
* TRAP: hero falls into a trap and must fight their way out 
* TOWN (escape): escape from the town or castle prison 
* OTHER : everything else 

To create a dataset I took some videos from a game walkthrough of mine, the adventures of Wendy. I used the episodes from 41 to 66 from following public playlists on youtube: 

* [CNN-Wendy-I](https://www.youtube.com/playlist?list=PLNP_nRm4k4jfVfQobYTRQAXV_uOzt8Bov)
* [CNN-Wendy-II](https://www.youtube.com/playlist?list=PLNP_nRm4k4jdEQ-OM31xNqeE64svvx-aT) 

These are some episodes I went through and manually split into scenes. I wrote down how they were split in the description. For instance, in episode 54, I have identified following scenes, of the category "Hideout", "Battle", "Tournament", "Town". All the other parts of the video are categorized as "Other".  

09:51-12:21 Hideout Tundra Bandits (Failed)
18:47-19:44 Battle with Sea Raiders
20:50-21:46 Battle with Sea Raiders
22:54-23:42 Battle with Sea Raiders
34:06-37:44 Tournament won in Tihr
38:46-40:48 Town escape for Boyar Vlan 

To prepare the data set, I had set up [a companion project](https://github.com/diegoami/DA_split_youtube_frames_s3/tree/support_playlists):
 
This project:
- Downloads the relevant videos from youtube, using the youtube-dl python library, in a 640x360 format
- Extract at every two seconds a frame and save it an jpeg file, using the opencv python library
- Scrape the description from the youtube description
- Distribute the files over directories named by the categories.

This way, I created first a dataset that [I uploaded to a S3 bucket](https://da-youtube-ml.s3.eu-central-1.amazonaws.com/wendy-cnn/frames/wendy_cnn_frames_data.zip) and made public.


## PROBLEM ANALYSIS

### DATASET ANALYSIS

The amount of images I have generated first in this way, using the current set of videos broken down in scenes, was 45718, divided over the three categorie . They are contained in [this zip file](https://da-youtube-ml.s3.eu-central-1.amazonaws.com/wendy-cnn/frames/wendy_cnn_frames_data.zip) on S3 (3.4 GB).

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
A sanity check whether the images are in the correct directory can be done using [this notebook](analysis.ipynb):

For instance battle images should look like that:
![Battle images](docimages/battle_images.png)


### FIRST ITERATION
Notebook: [Wendy_CNN.ipynb](Wendy_CNN.ipynb)

The simplest way I chose to verify whether a model is viable was to start and set up a Convolutional Neural Network in Pytorch, as I was pretty sure that 

* this was pretty much the most sensible way to approach the problem
* I could use standard CNN topologies available in Pytorch
* analyzing the result of the model would give me more information on what I would have to be looking for

Convolutional Neural Network are, as a matter of fact, a very standard approach for categorizing images. A simple to use and flexible topology I decided to use was VGG, which is included in the Pytorch library. 


As the images extracted from game walkthrough are not related to real world images, using a pretrained net and expand it with a transfer learning does not seem to make sense. Instead, I opted for a full train.
In the preprocessing phase, in this iteration, I resized images to 128 x 72, which should be enough for the algorithm to recognize features ( original images are all 640 x 360). As I already have around 48000 images, I thought I would do not need any kind of data augmentation (like, use mirrored images), also because it is not given that the game may actually show mirrored images.

The flaw in the dataset, regrettably, is that some categories, such as SIEGE, TRAP and TOWN, have relatively few samples. Looking at the confidence matrix I got, there were however some surprises in the report I got when I decided to test the produced model with 10% of the dataset. 


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

It turned out that the Siege class is not a problem (as a matter of fact, images belonging to this category are pretty distinctive). However, the classes Trap, Town and Training tended all to be misclassified. After checking the confidence matrix, I decided that it would make sense to remove these three categories, so that Training is classified as Other (Training is not interesting anyway) whil Trap and Town are classified as Battle.

### SECOND ITERATION

First, I make sure to create a second dataset, where I map Trap and Town to Battle, and Training to Other. So that I end up with 5 categories:

* Battle      : 5943 
* Hideout     : 1153
* Other       : 31650  
* Siege       : 194
* Tournament  : 6770

I chose a smaller format for the images I save, as they are already too big for any model I can realistically train. The dataset becomes therefore much smaller: 

I also add a preprocessing step to correct the mistakes that I find in the dataset. For this, I use a [list of images that were misclassified](letsplay_classifier/misclassified.json) produced by the script [verify_model](letsplay_classifier/verify_model.py). Using [a little tool](letsplay_classifier/sel_image.py) I try and sort between images that have a very high probability of being classified wrongly, and write down in files those images for which [I confirm the expected label](letsplay_classifier/confirmed.json)), and those images where [I reject the expected label for the predicted one](letsplay_classifier/rejected.json).




Avg acc (test): 0.9915
Confidence Matrix
[[ 6077     1    43     0     6]
 [   19  1128     4     4     2]
 [  135    10 31405     1    23]
 [    3     1     0   189     0]
 [    9     1   122     3  6524]]
              precision    recall  f1-score   support

           0       0.97      0.99      0.98      6127
           1       0.99      0.97      0.98      1157
           2       0.99      0.99      0.99     31574
           3       0.96      0.98      0.97       193
           4       1.00      0.98      0.99      6659

    accuracy                           0.99     45710
   macro avg       0.98      0.98      0.98     45710
weighted avg       0.99      0.99      0.99     45710

              precision    recall  f1-score   support

           0       0.97      0.99      0.98      6127
           1       0.99      0.97      0.98      1157
           2       0.99      0.99      0.99     31574
           3       0.96      0.98      0.97       193
           4       1.00      0.98      0.99      6659

    accuracy                           0.99     45710
   macro avg       0.98      0.98      0.98     45710
weighted avg       0.99      0.99      0.99     45710

[[ 6077     1    43     0     6]
 [   19  1128     4     4     2]
 [  135    10 31405     1    23]
 [    3     1     0   189     0]
 [    9     1   122     3  6524]]

## IMPLEMENTATION

For the implementation, we work with Sagemaker notebooks and Pytorch version 1.6, which we take care to include with torchvision, torchdata and scikit-learn (for preprocessing) taking advantage of the possibility of including a requirement.txt file.
We load the full datasets of images, but we split in a random and stratified way in a train and validation dataset. We the provide a script, train.py, that allows to train a VGG16 neural network, specifying the desired image width and height to which to resize, the amount of epochs, and the kind of layer layout in the VGG16 model that we desire.
In the provided Jupyter Notebook, we can trigger this training, and a standard run on this training script gives us an average loss of 0.003, and an average accuracy of 0.97, proving that the images contain enough information to allow a categorization.

I saved then the model and created an endpoint and a predictor. For simplicity, this predictor accepts and returns messages in JSON format. In the file endpoint.py, which acts as a client we call this predictor to calculate metrics using a random subset of images from our full dataset. I make sure that the client using the service does not use pytorch. 
The client keeps track of labels and predictions so that at the end it is able to print a confidence matrix and f1, precision and recall of the analyed subsets.

## RESULTS

This is the classification report of the selected model. Avg accuracy is 0.98. Classes having enough samples get a F1-value of 0.97, while classes not having enough samples have much worse results. To get better values on this, we would definitely need more samples. Oversampling or working with weight might not give results which are good enough, as the variance between images is too great. 	

The way we exchange data between the client and the service could be improved, as the performance and the time needed to get results on a batch of images is considerable.
 
## CONCLUSIONS

This project proved to me that it is possible to reliably build a classification model for images. I could apply this technique also to other video games, as the breakdown in scenes is something that is common.

I am planning to use this tool to build a client that would take a video as input, extract frames every two seconds and suggest the splitting of videos in sequences. 


