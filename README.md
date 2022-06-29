# GANs-for-Speech-Emotion-Recognition-and-Translation

**By**: Nestor Teodoro Chavez and Charudatta Manwatkar


## Introduction
Leveraging the power of deep learning model architectures has been at the forefront of progression in various fields. The agency in which audio processing has accelerated over the last decade has been astonishing. We've seen the power that CNN's hold on images and text. Our aim is to understand and build deep learning models on audio data. 

SER (Speech Emotion Recognition) is a task that focuses and emphasizes the recognition of emotional apsects in speech irrespective of semantic contents. This task partnered with [Neural Style Transfer] (https://en.wikipedia.org/wiki/Neural_style_transfer), in theory, should allow complex model architectures to be trained in order to learn a meaningful way of processing an audio file and translating from one emotion to another. 

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
We are using [RAVDESS dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio). The dataset consists of 24 voice actors. These voice actors provide audio in the form of .wav for various sentences and 8 emotions. The emotions we work with are: 
* Neutral
* Calm
* Happy 
* Sad
* Anger 
* Fear
* Disgust
* Surprise. 

## <a name="techniques">Techniques & Preprocessing</a> 
We preprocess our audio files by using Acoustic Feature Extraction. 
<p float="left">
  <img src="SampleImages/audio_ex.png" width="1200"/>
</p>
  
Here, we generate images in the form of a Mel Spectrogram. 
<p float="left">
  <img src="SampleImages/melSpec_ex.png" width="1200"/>
</p>

We then normalize these images. This is a plot of all emotions presented in the RAVDESS dataset that have been normalized. One can note, the mel spectrogram of emotions are all quite similar as the sentence the voice actor is stating is the same but in various varying emotions.  
<p float="left">
  <img src="SampleImages/sample_all_emotions.png" width="1200"/>
</p>

We can get a better sense of the dataset by looking at one more sample. Here, the mel spectrograms are different because the sentence and voice actor are different. 
<p float="left">
  <img src="SampleImages/sample_all_emotions_2.png" width="1200"/>
</p>  

## <a name="model">Model Architecture</a>
[Gneral Adversarial Network](https://en.wikipedia.org/wiki/Generative_adversarial_network)

<p float="left">
  <img src="SampleImages/GAN.png" width="800"/>
</p>
  
## <a name="results">Model Performance</a>
-- insert chart of training/validation --  
-- insert model metrics -- 
  
## <a name="next">Next Steps</a>
Additional improvements to increase model performance: 
- Add More Data 
- Hyperparameter Tuning 

---
*This project was done for the course MSDS 631 - Deep Learning, in partial completion of the Masters in Data Science degree program at the University of San Francisco.*
