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

## 🏃 Execution Instructions

This project is implemented as a Jupyter Notebook, and all code cells must be executed in sequence to ensure proper functioning. Follow these steps to run the model:

1. **Run the First Cell:**
   - The first cell of the notebook is crucial as it handles the training of the ResNet model on the dataset of labeled images.
   - This step includes defining a custom dataset class to load images and their associated labels from a CSV file, applying necessary transformations (resizing, converting to tensors), and setting up the ResNet-50 model.
   - The model is trained using a binary cross-entropy loss function with logits and the Adam optimizer over multiple epochs. After training, the model's state is saved as a `.pth` file for future use.

2. **Model Evaluation and Prediction:**
   - Subsequent cells load the saved model in evaluation mode.
   - The CLIP model is also loaded to validate predictions made by the trained ResNet model.
   - Provide an image as input, and the model will predict the top 10 keywords associated with the image, ranking them based on relevance.
   - CLIP is used to encode both the image and the predicted keywords into vectors. The similarities between the image features and keyword features are calculated to refine and return the top 5 matching keywords.
   - Each predicted keyword contributes to the score of a corresponding algorithm, and the algorithm with the highest score is identified as the most likely one used in the NFT.

3. **Running the Model on a Larger Sample:**
   - The `TrainValidAlgo.ipynb` notebook contains additional cells designed to run the model on a set of images located in the `ipfs` folder.
   - These cells allow the model to operate on a larger sample, and all outputs (predicted algorithms) are aggregated into a dictionary.
   - The goal of this step is to analyze the occurrences of each algorithm on the latest projects from the website, providing insights into which algorithms are most commonly used.

4. **Viewing Results:**
   - Outputs from the model, including the top 5 keywords and the detected algorithm, will be displayed in the notebook.
   - For the larger sample run, review the dictionary to understand the distribution and frequency of detected algorithms across the dataset.

### Additional Notes:
- Ensure that all required libraries and dependencies are installed and correctly configured in your environment.
- The execution flow of the notebook is sequential; running cells out of order may lead to errors or incorrect outputs.
- The performance of the model and the accuracy of the detected algorithms depend on the quality and representativeness of the training dataset.

For a more descriptive model focusing on broader object detection and image composition, see the [Descriptive Model](../DescriptiveModel/README.md).

