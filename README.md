# NFT Classification Project

This repository is the result of my work at fxhash and contains my intership project subdivided in 4 different approaches.

The project focuses on the classification of NFTs using various methods, primarily leveraging visual information. Each classification approach is organized separately in the GitHub repository, named according to the specific method or result. The approaches include:

- [Classification by Color](ClassificationbyColor/README.md): Categorizing NFTs based on their color schemes.
- [Classification by Tag](ClassificationbyTag/README.md): Categorizing NFTs based on their associated tags.
- [Algorithm Detection](AlgorithmDetection/README.md): A custom-trained AI model analyzes the NFTs to suggest keywords that likely indicate the generative algorithms used.
- [Descriptive Model](DescriptiveModel/README.md): Using CLIP and BLIP models to generate short, descriptive summaries of the images.

## Project Goals

**Support for Artists:** By suggesting relevant tags, artists can enhance their artwork's visibility when uploading to marketplaces.
**Improved Navigation:** Enables better search functionality on marketplaces by filling in missing tags or suggesting additional ones that authors might overlook.
**Market Insights:** Provides an overview of trends in the NFT marketplace by applying the models to a broader collection of artworks.

Mostly in python using jupyter notebook.

## Features

- Most likely used algorithm detection
- List of words related to the image
- Accurate description of the image
- Generating and clustering tags
- Occurences algorithm/tags from a large dataset


## Libraries used

- Python 3.7


## Models used

- CLIP
- BLIP
- ResNet-50
- Word2Vec
- spaCy


## Useful definitions

- **Epoch**: One complete pass through the entire training dataset by the model. During an epoch, the model processes each batch of data, learns patterns, and updates its weights.

- **Precision**: Percentage of correctly predicted positive results out of all predictions the model made for the positive class. It measures how many of the items predicted as positive are actually positive.

- **Recall (or Sensitivity)**: Percentage of correctly predicted positive results out of all actual positive cases in the dataset. It measures how well the model identifies all the true positive instances.

- **F1 Score**: Harmonic mean of precision and recall. It provides a balanced measure that considers both precision and recall, especially useful when the dataset has an uneven class distribution or when one metric is more important than the other.
