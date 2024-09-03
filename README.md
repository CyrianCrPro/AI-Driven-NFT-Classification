# AI NFT ANALYZER

This repository is the result of my work at fxhash.

It contains my intership project subdivided in 4 different approaches. All of them aim to ask an ai model information about a given nft from the fxhash.xyz website.

The project is mostly in python using jupyter notebook.

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