### Sound Classification using Deep Convolutional Neural Networks in ML.NET

#### Background
As of September 2019, it's possible to use transfer learning to natively re-train an InceptionV3 or Resnet CNN in ML.NET.
This enables .NET developers to be able to create their own custom image classification models, for their specific use cases.

However, Convolutional Neural Networks (CNN's) can be used in so many other applications than just image classification.
In this repo we'll demonstrate how to build a rudimentary sound classifier using ML.NET

#### Approach
To be able to classify sounds using a CNN, we first need to create an image of the audio.
To do this, we can create something called an audio spectrogram, which is visual presentation of the energy levels of a sound clip.

Below are spectrograms of guitars playing
![guitar](https://github.com/aslotte/mlnet-sound-classifier/blob/master/images/acoustic_guitar_23-spectro.jpg)
![guitar](https://github.com/aslotte/mlnet-sound-classifier/blob/master/images/acoustic_guitar_26-spectro.jpg)

#### Result
