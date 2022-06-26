# GANs-for-Speech-Emotion-Recognition-and-Translation

**By**: Nestor Teodoro Chavez and Charudatta Manwatkar


## Introduction

Table of Contents:<br>
1. [Deep Learning Goal](#goal)<br>
2. [Dataset](#data)<br>
3. [Preprocessing Techniques](#techniques)<br>
4. [Model Architecture](#model)<br>
5. [Model Performance](#results)<br>
6. [Next Steps](#next)<br>

## <a name="goal">Deep Learning Goal </a>
Our Deep Learning Goal is to leverage General Adversarial Networks in order to recognize the type of emotion from audio files. Our ultimate goal is to modify original audio source in order to generate a new audio file that has been translated to a different emotion. 

## <a name="data">Dataset</a> 
We are using [RAVDESS dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio). The dataset consists of 24 voice actors. These voice actors provide audio in the form of .wav for various sentences and 8 emotions. The emotions we work with are Neutral, Calm, Happy, Sad, Angry, Fear, Disgust, and Surprise. 

## <a name="techniques">Techniques & Preprocessing</a> 
We preprocess our audio files by using Acoustic Feature Extraction. 
-- insert acoustic image -- 
  
Here, we generate images in the form of a Mel Spectrogram. 
-- insert Mel Spectrogram for all emotions --  
  
We then normalize these images. 
-- insert normalized image -- 
  
## <a name="model">Model Architecture</a>
[Gneral Adversarial Network](https://en.wikipedia.org/wiki/Generative_adversarial_network)
  
## <a name="results">Model Performance</a>
-- insert chart of training/validation --  
-- insert model metrics -- 
  
## <a name="next">Next Steps</a>
Additional improvements to increase model performance: 
- Add More Data 
- Hyperparameter Tuning 

** This project was done for the course MSDS 631 - Deep Learning, in partial completion of the Masters in Data Science degree program at the University of San Francisco.
