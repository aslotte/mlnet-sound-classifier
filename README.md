### Sound Classification using Deep Convolutional Neural Networks in ML.NET

#### Background
As of September 2019, it's possible to use transfer learning to natively re-train an InceptionV3 or Resnet CNN in ML.NET.
This enables .NET developers to be able to create their own custom image classification models, for their specific use cases.

However, Convolutional Neural Networks (CNN's) can be used in so many other applications than just image classification.
In this repo we'll demonstrate how to build a rudimentary sound classifier using ML.NET

#### Disclaimer
The data used for the training was retrieved from an online research paper. 
I've since starting on this repo lost track of the resarch paper, but is determined to provide credit to the original creators of the data used during the training, once I'm able to locate the research paper again.

#### Approach
To be able to classify sounds using a CNN, we first need to create an image of the audio.
To do this, we can create something called an audio spectrogram, which is visual presentation of the energy levels of a sound clip.

Below is an example in which we've transformed an audio file of a guitar playing to a spectrogram.

![guitar](https://github.com/aslotte/mlnet-sound-classifier/blob/master/images/acoustic_guitar_23-spectro.jpg)

The spectrograms where generated using the open-source library Spectrogram.NET.
Once we have our images, we can then train our deep neural network with ML.NET.

```
var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelAsKey", 
                                                                            inputColumnName: "Label",
                                                                            keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                        .Append(mlContext.Model.ImageClassification("ImagePath", "LabelAsKey",
                                        arch: ImageClassificationEstimator.Architecture.InceptionV3,
                                        epoch: 200,                     
                                        metricsCallback: (metrics) => Console.WriteLine(metrics),
                                        validationSet: transformedValidationDataView));
```

#### Result
The model currently only yields a 75% accuracy on the validation dataset, which under the circumstances is pretty good.
The accuracy can most likely be improved by increasing the size of the dataset used for training, or augmenting the spectrograms further by e.g. trasforming then to mel-spectrograms, which will provide even more detail.
