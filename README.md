# fruity - A project for the course 02476 DTU-MLOps

## Group 14 - Members
* Anders Henriksen, s183904
* August Semrau Andersen, s183918
* Karl Byberg Ulbæk, s183931
* William Diedrichsen Marstrand, s183921

## Project Description

### Project Goal
The primary objective of this project is to develop a robust, accurate, and reliable machine learning model dedicated to classifying a diverse array of fruits and vegetables.

### Third-party Framework - TIMM
Our project will use the capabilities of the [`timm`](https://github.com/rwightman/pytorch-image-models) (PyTorch Image Models) framework, a rich repository of pre-trained deep neural network models and scripts designed for PyTorch. TIMM's comprehensive selection of models, tailored for image data, presents an ideal foundation for our classification objectives. We plan to utilize TIMM in several key areas:

* **Access to pre-trained models**: TIMM offers a wide range of pre-trained models, which serve as a robust starting point. These models, already trained on large and diverse datasets, can significantly improve learning efficiency and accuracy
* **Model Experimentation and Selection**: Given TIMM's extensive collection of image model architectures, we can easily experiment with different models to find the most suitable one for our project.
* **Customization and Fine-tuning**: TIMM not only offers pre-trained models but also allows customization. We can adjust layers, training parameters, and other aspects of the model to better suit our dataset and classification goals

### Dataset
Our initial data set will be the [`Kaggle Fruits-360 dataset`](https://www.kaggle.com/datasets/moltean/fruits/data?fbclid=IwAR3nV6QmcRhNnCAnHAxbYpyHgke-qujIYtBPymdTrrD_IZ9jSMWnnVcAZm4) (version 2020.05.18.0).

**Dataset properties**
* The total number of images: 90483.
* Training set size: 67692 images (one fruit or vegetable per image).
* Test set size: 22688 images (one fruit or vegetable per image).
* The number of classes: 131 (fruits and vegetables).
* Image size: 100x100 pixels.

### Models
We expect to investigate various architectures including:

* **ResNet Variants**: Utilizing their deep architecture for training deeper networks.
* **EfficientNet Models**: Chosen for their balance of accuracy and efficiency.
* **Vision Transformer (ViT)**: Leveraging transformer architecture as a CNN alternative for image classification.

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── conf                 <- Hydra configuration folder
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── fruity  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- MIT Open-source license
```