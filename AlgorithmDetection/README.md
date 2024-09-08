# Algorithm Detection

Training a ResNet model on a dataset of labeled images and using CLIP to verify generated keywords against images.

Defines a custom dataset class for loading images and their associated labels from a CSV file, and applies transformations to resize and convert images into tensors.

A ResNet-50 model is loaded and modified to output probabilities for the number of unique labels in the dataset.
The model is trained using a binary cross-entropy loss function with logits and Adam optimizer over multiple epochs.
Training involves forward passes, loss calculation, and backpropagation to update model weights.
The model's state is saved as a .pth file.
Loads and sets the model to evaluation mode.
The CLIP model is also loaded to use for verifying predictions.
Takes an image, uses the trained model to predict top 10 keywords associated with the image, and ranks them.
It then uses CLIP to encode the image and predicted keywords into vectors.
The similarities between the image features and keyword features are calculated, and the top 5 matching keywords are refined and returned.

In this approach, every word in the dataset is mapped to an algorithm. We train a model to predict five words related to a given input. Each of these five predicted words increases the score of the corresponding algorithm. The algorithm with the highest total score is then identified as the most likely one used in the NFT.


## ⚠️ Important Note

The approach described here uses a model that generates outputs strictly based on the data it was trained on, specifically focusing on labeling images with terms from the dataset. While this method identifies and describes images using the dataset's predefined labels, it may not capture broader or more general object compositions (e.g., trees, animals) beyond its training scope.

For comprehensive image descriptions and object detection, alternatives like the Google Vision API and BLIP offer more advanced capabilities. These tools can recognize a wide array of objects and provide detailed compositions of images. However, it's important to note that they are not specifically designed to detect generative algorithms. As a result, when tasked with identifying generative algorithms, these services may often fail or provide less relevant outputs compared to the targeted model used in this approach.

**[Descriptive Model](../DescriptiveModel/README.md)**
