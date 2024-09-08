# NFT Classification Project 🖼️

This repository is the result of my work at fxhash and contains my internship project subdivided in 4 different approaches.

![Header Image](images/header.webp)

## 📚 Table of Contents
1. [Project Overview](#project-overview)
2. [Classification Approaches](#-classification-approaches)
3. [Project Goals](#-project-goals)
4. [Tools and Models](#-tools-and-models)
5. [Glossary of Key Terms](#-glossary-of-key-terms)

## 📜 Project Overview
**AI-Driven NFT Classification!**  
The project focuses on the classification of NFTs using various methods, primarily leveraging visual information. Each classification approach is organized separately in the GitHub repository, named according to the specific method or result.

## 🔍 Classification Approaches
- 🎨 **[Classification by Color](ClassificationbyColor/README.md):** Categorizes NFTs based on their color schemes.
- 🏷️ **[Classification by Tag](ClassificationbyTag/README.md):** Categorizes NFTs based on their associated tags.
- 🤖 **[Algorithm Detection](AlgorithmDetection/README.md):** Uses AI to detect the generative algorithms behind NFTs, providing insights into the creative process.
- 📜 **[Descriptive Model](DescriptiveModel/README.md):** Employs CLIP and BLIP models to generate concise and accurate descriptions of NFTs.

## 🎯 Project Goals
- **🎨 Support for Artists:** Enhance visibility with relevant tag suggestions, empowering artists on FXHASH.
- **🕵️ Improved Navigation:** Boost search functionality by filling in missing tags and suggesting additional ones.
- **📊 Market Insights:** Analyze trends across FXHASH by applying the models to a broader collection of artworks.

## 🛠️ Tools and Models

- **🐍 Python 3.7:** The primary programming language used for implementing models and scripts.
- **🖼️ CLIP & BLIP:** 
  - **CLIP (Contrastive Language–Image Pretraining):** A model that matches images and text, used for verifying predicted keywords against image content.
  - **BLIP (Bootstrapping Language-Image Pretraining):** A model that generates descriptive captions for images.
- **🧠 ResNet-50:** A convolutional neural network (CNN) used for visual classification tasks, particularly image recognition.
- **📝 Word2Vec & spaCy:** 
  - **Word2Vec:** A neural network model to generate word embeddings, capturing semantic meanings of words.
  - **spaCy:** An advanced library for NLP tasks like tokenization, part-of-speech tagging, and similarity calculations.
- **⚙️ PyTorch:** A deep learning framework for building and training neural networks.
- **📊 scikit-learn:** Used for machine learning tasks such as clustering, metrics calculation, and cross-validation.

### 📦 Libraries

- **torch:** PyTorch, a deep learning framework for creating neural networks and running them on GPUs.
- **torchvision:** Provides popular datasets, model architectures, and image transformations for computer vision.
- **pandas:** A data manipulation and analysis library, used for reading and handling CSV data.
- **numpy:** A fundamental package for numerical computations in Python, used for handling arrays and matrices.
- **matplotlib:** A plotting library used for creating static, interactive, and animated visualizations in Python.
- **PIL (Pillow):** A fork of the Python Imaging Library (PIL) that adds support for opening, manipulating, and saving many different image file formats.
- **scikit-learn:** A machine learning library that provides simple and efficient tools for data analysis and modeling.
  - **MultiLabelBinarizer:** Used for converting lists of labels to binary form for multi-label classification.
  - **KMeans:** A clustering algorithm for partitioning data into groups based on similarity.
  - **Metrics:** Functions like `precision_score`, `recall_score`, and `f1_score` for evaluating model performance.
- **gensim.models (Word2Vec):** A library for training and using word embeddings and topic models.
- **nltk:** The Natural Language Toolkit, used for text processing tasks such as tokenization.
- **json:** A module to work with JSON data.
- **requests:** A simple and elegant HTTP library for Python, used for making requests to APIs and handling the responses.
- **transformers:** A library by Hugging Face that provides state-of-the-art machine learning models like BERT, GPT, and more, including BLIP models.
- **clip (OpenAI):** A library for the CLIP model by OpenAI, which can understand images and text in a similar way.
  
## 📖 Glossary of Key Terms

- **🕰️ Epoch:** A complete pass through the entire training dataset. One epoch involves feeding all training data into the model once.
- **✅ Precision:** The ratio of correctly predicted positive observations to the total predicted positives. Precision is a measure of a model's exactness.

<pre>
Precision = True Positives / (True Positives + False Positives)
</pre>

- **📈 Recall (Sensitivity):** The ratio of correctly predicted positive observations to all the observations in the actual class. It measures a model's ability to find all relevant cases.

<pre>
Recall = True Positives / (True Positives + False Negatives)
</pre>

- **⚖️ F1 Score:** The harmonic mean of precision and recall, providing a balanced measure that considers both false positives and false negatives.

<pre>
F1 Score = 2 × (Precision × Recall / Precision + Recall)
</pre>

- **🔍 K-Fold Cross-Validation:** A resampling procedure used to evaluate machine learning models on a limited data sample. The dataset is split into 'k' smaller sets, and the model is trained and validated 'k' times, each time using a different fold as the validation set.
  
- **🚀 CUDA:** A parallel computing platform and API model created by NVIDIA. CUDA allows developers to use GPUs for general purpose processing (an approach known as GPGPU), accelerating computing tasks significantly.

- **🛑 Early Stopping:** A technique used during training to halt training when a monitored metric, such as validation loss, stops improving, helping to prevent overfitting.

- **📊 Binary Cross-Entropy Loss with Logits:** A loss function used for multi-label classification tasks, where each label is treated as a separate binary prediction problem.

- **⚙️ Adam Optimizer:** An adaptive learning rate optimization algorithm used for training deep learning models, known for its efficiency and effectiveness.

