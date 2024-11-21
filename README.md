# Jet Classification with Set2Graph

This project implements a graph neural network (GNN) based on the Set2Graph architecture to classify jets as b, c, or light using data from the Large Hadron Collider (LHC).

## Table of Contents
- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Expected Results](#expected-results)
- [References](#references)

---

## Introduction
Jets are particle streams produced in high-energy collisions, originating from the fragmentation of quarks and gluons. This project focuses on classifying jets into three categories:
- **b-jets**: From bottom (b) quarks.
- **c-jets**: From charm (c) quarks.
- **Light jets**: From lower-mass quarks (u, d, s).

The Set2Graph neural network is used to classify jets by modeling relationships between particle tracks in jets.

---

## Dataset Description
The dataset consists of jets simulated with Pythia8 and Delphes (emulating an ATLAS-like detector). It includes:
- **Training data**: ~450,000 samples (`training_data.root`)
- **Validation data**: ~181,000 samples (`valid_data.root`)
- **Test data**: ~181,000 samples (`test_data.root`)

Each sample includes:
- **Input features**: Six perigee parameters and jet features (e.g., transverse momentum, pseudorapidity).
- **Labels**: Integer indicating jet type (b, c, light).

Download and structure the dataset automatically using the provided scripts.

---

## Project Structure

project/ 
├── data/                   # Dataset folder 
├── models/                 # Neural network models 
│   ├── __init__.py
|   └── set2graph_model.py 
├── scripts/                # Pipeline scripts 
│   ├── __init__.py
│   ├── download_data.py    # Downloads the dataset 
│   ├── train.py            # Trains the model 
│   ├── evaluate.py         # Evaluates the model 
├── utils/                  # Utility scripts 
│   ├── __init__.py
│   └── dataset.py 
├── .gitignore              # Ignore files to avoid uploading them to the repository               
├── env_S2G_DV.yml          # Conda environment file 
├── main.py                 # Main controller script 
├── README.md               # Documentation
└── requirements.txt        # Pip dependencies 



---

## Setup

### Prerequisites
- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

### Create the Environment
1. Clone the repository:
   ```bash
   git clone git@github.com:vapdif/Set2Graph_DiegoVasquez.git
   cd jet-classification

    Create the environment from the environment.yml file:
        conda env create -f env_S2G_DV.yml

    Activate the environment:
        conda activate jet-class_DV

    Install additional pip dependencies:

        for linux, CUDA 11.8, PyTorch:
           conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
        
        for PyTorch Geometric 2.5.1
            pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
            pip install torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
            pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
            pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
            pip install torch-geometric

        Check the installation PyTorch y PyTorch Geometric 
            python -c "import torch; import torch_geometric; print(torch.__version__, torch_geometric.__version__)"



    Modification at the Terminal (Temporary)
    Before running main.py, export the root directory to the PYTHONPATH:
        export PYTHONPATH=$(pwd)

## Usage

### Step 1: Download the Dataset

Run the following command to download the dataset:
    python main.py download

### Step 2: Train the Model

Train the Set2Graph model using:
    python main.py train
    
    The trained model will be saved as set2graph_model.pth.

### Step 3: Evaluate the Model

Evaluate the model using the test set:
    python main.py evaluate

## Evaluation Metrics

The model is evaluated using the following metrics:

    F1-Score: Combines precision and recall into a single metric.
    Rand Index (RI): Measures similarity between predicted and actual labels.
    Adjusted Rand Index (ARI): Adjusts RI for chance-level similarity.

## Expected Results

The trained Set2Graph model should achieve:

    F1-Score: ~0.77
    Accuracy: ~0.78

These results demonstrate high accuracy in classifying b, c, and light jets.

## References

    Secondary Vertex Finding in Jets Dataset.
    J. Shlomi et al., "Secondary Vertex Finding in Jets with Neural Networks," arXiv:2008.02831.
    G. Aad et al., "Observation of Jets in High-Energy Proton-Proton Collisions at the LHC," Physical Review Letters, 2010.

## Contact

For questions or suggestions, contact Diego F. Vasquez P. 
    diego.vasquez@upr.edu



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
