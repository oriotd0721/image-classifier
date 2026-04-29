# Image Classifier — Transfer Learning (PyTorch)

![Python](https://img.shields.io/badge/language-Python-blue)
![PyTorch](https://img.shields.io/badge/library-PyTorch-red)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)
![Build](https://img.shields.io/badge/build-train-lightgrey)

> Personal project by **Oriance Oternaud**, Computer Science student at the **University of Geneva (UNIGE)**.  
> This program implements an image classifier based on **transfer learning** from the **ResNet18** model pre-trained on **ImageNet**.  
> _Personal project in Artificial Intelligence — image classification with PyTorch._


## Key Features

- Transfer learning with **ResNet18 (pre-trained on ImageNet)**
- Binary classification: **Cats vs Dogs**
- Separate scripts for **training** and **inference**
- Compatible with **CPU / GPU / Apple MPS (Mac)**


## Project Structure
├── src/

│   ├── train.py

│   ├── infer.py

├── models/

│   ├── resnet18.pt

│   └── labels.json

├── requirements.txt

└── README.md




## Installation

Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:


```bash
pip install -r requirements.txt
```

## Training
Make sure your dataset is organised as follows:

data/
└── cats_vs_dogs/
    ├── train/
    │   ├── cats/
    │   └── dogs/
    └── val/
        ├── cats/
        └── dogs/


## Run training:

```bash
python src/train.py --data data/cats_vs_dogs --epochs 5
The model will be saved in the models/ directory:
```

models/resnet18.pt
models/labels.json
Inference / Prediction

To test an image after training:

python src/infer.py --weights models/resnet18.pt --labels models/labels.json data/cats_vs_dogs/val/dogs/dog.4001.jpg

Example output:
Predicted: dogs (p=0.70)
Results
The model was trained for 3 epochs on the Cats vs Dogs dataset.

## Technologies & Tools
Language: Python
Framework: PyTorch
Model: ResNet18 (ImageNet)
Dataset: Cats vs Dogs (Kaggle)
OS: macOS
Version control: Git & GitHub


## License
This project is released under the MIT License.
You are free to reuse, modify, and distribute the code for educational or non-commercial purposes.

See the LICENSE file for the full text.



