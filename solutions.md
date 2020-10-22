# CAPSTONE Project - Machine Learning Engineer Nanodegree

October 2020

## PROBLEM DEFINITION


During the last few years it has become more and more common to stream on platforms such as Youtube and Twitch while playing video games, or to upload recorded sessions. The volume of videos produced is overwhelming. In many of the videos games being streamed there are different types of scenes. Both for content producers and consumers it would be useful to be able to automatically split videos, to find out in what time intervals different types of scenes run. For instance, having as an input the video recording of a Minecraft speedrun, we could be able to produce the time intervals when the game is taking place in the Overworld surface, in caves, in the Nether and the End respectively - the four main settings of this game.

The game that I have chosen to use is Mount of Blade, of which I have several walkthroughs. In this game, I have identified seven types of scenes to which an image belongs:

* BATTLE: any battle taking place in an open field or in a village 
* TOURNAMENT: Tournaments in arena 
* HIDEOUT: the warband assaults a bandit hideout 
* TRAINING: the hero trains in a field or in a village 
* SIEGE: a town is sieged 
* TRAP: hero falls into a trap and must fight their way out 
* TOWN (escape): escape from the town or castle prison 
* OTHER : everything else 

To create a dataset I took some videos from a game walkthrough of mine, the adventures of Wendy. I used the episodes from 41 to 66 from following public playlists on youtube: 

* [Wendy - Part I -  Let's play Mount and Blade](https://www.youtube.com/watch?v=ei-ZqMq0PDY&list=PLNP_nRm4k4jd-AJ0GwTPS1ld2YP8FdT4h)
* [Wendy - Part II - Let's play Mount and Blade - Retired](https://www.youtube.com/watch?v=pnP3b5wXMZM&list=PLNP_nRm4k4jfNLo7FkjXewFH9Xe5Uc2Pa) 

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

This way, I created a dataset that [I uploaded to a S3 bucket](https://da-youtube-ml.s3.eu-central-1.amazonaws.com/wendy-cnn/frames/wendy_cnn_frames_data.zip) and made public.


## PROBLEM ANALYSIS

Convolutional Neural Network are a very standard approach for categorizing images. There are several templates to create neural network. One that is included in Pytorch is VGG16, with several types of layers. 
As the images extracted from game walkthrough are not related to real world images, using a pretrained net possibly expanding it with a layer does not make sense. Instead, we would opt for a full train.
Before feeding them to the neural networks, images are resized to 128 x 72, which should be enough for the algorithm to recognize features ( original images are all 640 x 360). As we already have around 48000 images, we do not do data augmentation (like, use mirrored images), also because it is not given that the game may actually show mirrored images.

The flaw in the dataset, regrettably, is that some categories, such as SIEGE, TRAP and TOWN, have relatively few samples. However, in this first pass, we would not modify the dataset. 

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


