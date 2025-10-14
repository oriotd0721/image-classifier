# Classificateur d’Images — Apprentissage par Transfert (PyTorch)

![Python](https://img.shields.io/badge/langage-Python-blue)
![PyTorch](https://img.shields.io/badge/bibliothèque-PyTorch-red)
![License: MIT](https://img.shields.io/badge/Licence-MIT-yellow)
![Build](https://img.shields.io/badge/build-train-lightgrey)

> Projet personnel réalisé par **Oriance Oternaud**, étudiante en **Sciences Informatiques** à l’**Université de Genève (UNIGE)**.  
> Ce programme implémente un classificateur d’images basé sur **l’apprentissage par transfert** à partir du modèle **ResNet18** pré-entraîné sur **ImageNet**.  
> _Projet personnel en Intelligence Artificielle — classification d’images avec PyTorch._


## 🚀 Fonctionnalités principales

-  Apprentissage par transfert avec **ResNet18 (pré-entraîné sur ImageNet)**  
-  Classification binaire : **Chats vs Chiens**  
-  Scripts séparés pour l’**entraînement** et l’**inférence**  
- Compatible **CPU / GPU / Apple MPS (Mac)**  
-  Code modulaire et clair pour un usage éducatif  


##  Structure du projet
├── src/

│   ├── train.py   

│   ├── infer.py    

├── models/

│   ├── resnet18.pt   

│   └── labels.json   

├── requirements.txt   

└── README.md



##  Installation

1️ Crée un environnement virtuel et active le :
```bash
python -m venv venv
source venv/bin/activate
```

Installe les dépendances :
```
pip install -r requirements.txt
```

## Entraînement du modèle

Assure-toi que ton dataset est organisé comme ceci :

```
data/
 ├── cats_vs_dogs/
     ├── train/
     │   ├── cats/
     │   └── dogs/
     └── val/
         ├── cats/
         └── dogs/

```

Lance l’entraînement :
```
python src/train.py --data data/cats_vs_dogs --epochs 5
```
Le modèle sera sauvegardé dans le dossier models/ :

```
models/resnet18.pt
models/labels.json
```

## Inférence/ prédiction:

Pour tester une image après l’entraînement :

```
python src/infer.py --weights models/resnet18.pt --labels models/labels.json data/cats_vs_dogs/val/dogs/dog.4001.jpg
```
Exemple de sortie :
```
Predicted: dogs (p=0.70)
```
## Résultats

Le modèle a été entraîné pendant 3 époques sur le dataset Cats vs Dogs.

## Technologies & outils

•	Langage : Python

•	Framework : PyTorch

•	Modèle : ResNet18 (ImageNet)

•	Dataset : Cats vs Dogs (Kaggle)

•	Système d’exploitation : macOS

•	Gestion de versions : Git & GitHub

## Licence

Ce projet est publié sous licence MIT.
Vous pouvez librement réutiliser, modifier et distribuer le code à des fins éducatives ou non commerciales.

Voir le fichier LICENSE pour le texte complet.


