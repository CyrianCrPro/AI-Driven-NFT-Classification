# NFT Classification Project 🖼️

This repository is the result of my work at fxhash and contains my internship project subdivided in 4 different approaches.

![Header Image](images/header.webp)

## 📚 Table of Contents
1. [Project Overview](#project-overview)
2. [Classification Approaches](#-classification-approaches)
3. [Project Goals](#-project-goals)
4. [Tools and Models](#-tools-and-models)
5. [Glossary of Key Terms](#-glossary-of-key-terms)
6. [System Requirements](#-system-requirements)
7. [Possible Improvements](#-possible-improvements)
8. [Known Caveats and Possible Issues](#-known-caveats-and-possible-issues)

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

- **🐍 Python (10.8 used):** The primary programming language used for implementing models and scripts.
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

## 🖥️ System Requirements

To run the provided codes efficiently, ensure your system meets the following hardware and software requirements:

### 1. Hardware Requirements (& Recommendations)

- **GPU:**
  - **NVIDIA GPU with CUDA support:** Required for accelerating deep learning model training and inference. (Can work on the CPU if you don't have a GPU but this is not recommanded)
  - **Minimum GPU Memory:** 8 GB or higher (recommended for larger models and batch sizes).
- **CPU:**
  - **Multi-core CPU:** At least a quad-core processor to handle data loading, pre-processing, and other tasks.
- **RAM:**
  - **Minimum:** 16 GB RAM.
  - **Recommended:** 32 GB or higher, especially for large datasets or multiple models.
- **Storage:**
  - **SSD:** Recommended for faster data loading and saving.
  - **Minimum Free Space:** 50 GB, depending on dataset and model size.

### 2. Software Requirements

- **Operating System:**
  - **Linux (Ubuntu 18.04 or newer):** Preferred for better compatibility with CUDA and PyTorch.
  - **Windows 10/11** or **macOS:** Supported but may require additional configuration for GPU acceleration (Not at the moment).
- **Libraries**
  - **PyTorch:** Version 2.4.1 (LTS) with CUDA support. [Pytorch Website](https://pytorch.org/get-started/locally/)
  - **CUDA Toolkit:** Version compatible with your PyTorch installation. [CUDA Toolkit Website](https://developer.nvidia.com/cuda-downloads) (Not needed if you use CPU)

Currently, PyTorch on Windows only supports Python 3.8-3.11; Python 2.x is not supported.

The project runs on Python 3.10.8 and PyTorch CPU

### 3. Configuration and Environment Setup

- **CUDA and cuDNN:**
  - Ensure correct versions are installed to match the PyTorch version (refer to PyTorch's official installation guide for compatibility).
- **Python Virtual Environment:**
  - Use `virtualenv`, `conda`, or another environment manager to isolate dependencies.
  
### 4. Network Requirements

- **Internet Connection:**
  - Required for downloading pre-trained models (e.g., ResNet-50, CLIP, BLIP) and dependencies from package managers. **pip install git+https://github.com/openai/CLIP.git**
  
### 5. Additional Recommendations

- **Docker (Optional):**
  - For creating a reproducible environment and simplifying dependency management.
- **Jupyter Notebook (Optional):**
  - For interactive development and testing, especially useful during experimentation.

### Summary
For optimal performance, use a system with a recent NVIDIA GPU, 32 GB of RAM, and a multi-core CPU. Ensure that CUDA and relevant Python libraries are correctly installed and configured.

## 🚀 Possible Improvements

To enhance the performance and capabilities of the current implementation, consider the following improvements:

### 1. Model Coupling
- **Ensemble Learning:** Use multiple models simultaneously and combine their outputs to improve prediction accuracy and robustness. This approach leverages the strengths of different models, such as combining CLIP with ResNet-50 and BLIP for richer feature extraction and more accurate keyword verification.
- **Model Fusion:** Explore techniques like weighted averaging, stacking, or other ensemble methods to create a fused output that captures diverse aspects of the data, potentially improving the overall model performance.

### 2. Improving the Dataset
- **Data Quality and Augmentation:** Enhancing the dataset can significantly boost model performance. This includes curating high-quality data, increasing the diversity of images, and applying advanced augmentation techniques. For more details, refer to the [how2dataset.md](./how2dataset.md).

### 3. Hyperparameter Tuning
- **Optimization:** Fine-tune hyperparameters such as learning rates, batch sizes, and optimizer settings to find the best configuration for model training.
- **Automated Tuning:** Utilize tools like Optuna, Hyperopt, or grid search methods to systematically explore hyperparameter spaces.

### 4. Leveraging Transfer Learning
- **Pre-trained Models:** Use more specialized pre-trained models (e.g., domain-specific models) that are fine-tuned on similar datasets to reduce training time and improve performance.
- **Layer Freezing:** Experiment with freezing lower layers of the neural networks to retain learned features from large datasets while adapting the higher layers to your specific task.

### 5. Advanced Evaluation Metrics
- **Beyond F1 Score:** Implement additional evaluation metrics such as ROC-AUC, Precision-Recall AUC, or metrics specific to multi-label classification to gain deeper insights into model performance.
- **Cross-Validation:** Extend K-Fold cross-validation to include stratified sampling to ensure balanced representation of labels across folds.

### 6. Implementing Explainability
- **Model Interpretability:** Integrate methods like Grad-CAM or SHAP to visualize and interpret model predictions, making the AI's decision-making process more transparent.
- **User Feedback Loop:** Incorporate user feedback mechanisms to refine model predictions continuously, creating a dynamic model that learns from real-world interactions.

### 7. Scaling and Deployment
- **Scalability:** Optimize the codebase for deployment on cloud platforms (e.g., AWS, Google Cloud, Azure) with GPU support to handle large-scale data efficiently.
- **API Integration:** Develop RESTful APIs or other interfaces to integrate the model into existing workflows, making it accessible for broader use.

These improvements aim to refine current approaches, enhancing both model performance and the overall robustness of the system.

## ⚠️ Known Caveats and Possible Issues

When using this project, be aware of the following caveats and potential issues:

### 1. Overfitting and Underfitting
- **Overfitting Risk:** The model may overfit if the training dataset is not sufficiently representative or if the model is trained for too many epochs. This can result in poor generalization to new, unseen data.
- **Underfitting Risk:** Insufficient training or overly simplistic models may lead to underfitting, where the model fails to capture underlying patterns in the data.

### 2. Dataset Limitations
- **Representation:** The dataset needs to adequately represent every algorithm; otherwise, the model may become biased towards the more prevalent algorithms.
- **Labeling Quality:** Current dataset labels may not be optimal, which can impact the performance and accuracy of the model. Improved labeling strategies and more balanced data distribution are recommended.
- **Content Quantity:** Additional data content is required to enhance model training and ensure robustness. Refer to the [how2dataset.md](./how2dataset.md) for guidance on improving the dataset.

### 3. Resource Constraints
- **High Resource Consumption:** Training models with a high number of epochs can significantly increase computational resource demands, including GPU memory and processing time. Plan resources accordingly, especially when scaling up.
- **Cloud API Costs:** If using external services like cloud vision or other APIs, be aware that costs can escalate quickly with high-volume requests. Monitor usage to manage expenses.

### 4. API Limitations
- **FXHash API:** The FXHash API may have limitations, such as only displaying the last 50 projects. This information is based on current understanding and may change; always check the latest API documentation for updates.

### 5. Model Viability
- **Output Monitoring:** The current model does not guarantee 100% accuracy or reliability. It is crucial to monitor and validate each output before applying them in critical applications to ensure correctness.

### 6. Model and Integration Issues
- **Compatibility:** Ensure that all dependencies are up to date and compatible with each other, particularly when integrating multiple models or using complex ensemble methods.
- **Version Control:** Keep track of model versions and training configurations, as inconsistencies can lead to unexpected results or degraded performance.

### 7. Performance
- **Performance Variability:** Model performance may vary across different environments, hardware configurations, or data conditions, which should be tested thoroughly.

### 8. External Dependencies
- **Dependency Stability:** Relying on external libraries or APIs introduces the risk of changes or deprecations that could affect the project’s functionality. Regular updates and checks are recommended to maintain compatibility.

### 9. Deprecated Elements
- **Deprecated elements:** Some of the parameters used in the algorithm detection and descriptive model approaches are deprecated and could be replace.
  
These known issues should be taken into account when deploying and using the model.
