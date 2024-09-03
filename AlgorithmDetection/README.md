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


## -- SimpleTrain.ipynb --

In this method we're training the model on the entire dataset and ploting loss over epoch.


## -- TrainValidAlgo.ipynb --

Train and validation datasets involves splitting the data into two separate parts:

Training Set: This portion of the data is used to train or fit the model. The model learns patterns, relationships, and features from this data.
Validation Set: This part is used to evaluate the model's performance during training. It helps in tuning parameters and assessing how well the model generalizes to unseen data, which is not used for learning but for validation.

The primary goal of this split is to ensure that the model can generalize well to new, unseen data and is not just memorizing the training data.

The Early Stopping is here to monitor validation loss and stops training if no improvement is observed for a set number of epochs, saving the best-performing model.

Performance Metrics are plotted and displayed to evaluate the final model using precision, recall, and F1 score, which are critical for multi-label classification tasks where correct label prediction matters for multiple labels per image.

## -- trainKfold.ipynb --

K-fold cross-validation is a technique used to evaluate the performance of an AI model. It involves splitting the dataset into 'k' equal-sized parts or "folds." The model is trained on 'k-1' folds and tested on the remaining fold. This process is repeated 'k' times, with each fold being used as the test set exactly once.

The results from all 'k' iterations are averaged to give a more reliable estimate of the model's performance, helping to ensure that the model generalizes well to new, unseen data. This method helps to reduce the variance associated with a single train-test split, providing a more robust evaluation of the model's accuracy.