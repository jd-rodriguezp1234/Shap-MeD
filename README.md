# Shap-MeD
Code for [Shap-Med article](https://arxiv.org/abs/2503.15562)

## Abstract
We present Shap-MeD, a text-to-3D object generative model specialized in the biomedical domain. The objective of this study is to develop an assistant that facilitates the 3D modeling of medical objects, thereby reducing development time. 3D modeling in medicine has various applications, including surgical procedure simulation and planning, the design of personalized prosthetic implants, medical education, the creation of anatomical models, and the development of research prototypes. To achieve this, we leverage Shap-e, an open-source text-to-3D generative model developed by OpenAI, and fine-tune it using a dataset of biomedical objects. Our model achieved a mean squared error (MSE) of 0.089 in latent generation on the evaluation set, compared to Shap-e's MSE of 0.147. Additionally, we conducted a qualitative evaluation, comparing our model with others in the generation of biomedical objects. Our results indicate that Shap-MeD demonstrates higher structural accuracy in biomedical object generation

## How to cite
Please cite by using the next Bibtex citation:
```
@misc{laverde2025shapmed,
      title={Shap-MeD}, 
      author={Nicolás Laverde and Melissa Robles and Johan Rodríguez},
      year={2025},
      eprint={2503.15562},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2503.15562}, 
}
```

## Team Members
- Nicolas Laverde Manotas
- Melissa Robles Carmona
- Johan Rodriguez Portela

## Project Overview
This project consists of developing a text-to-3D mesh workflow specialized in biomedical meshes. To see details of the initial proposal, it is recommended to read the [research proposal](proposal.pdf). For details about results and complete theoretical framework, it is recommended to read the [research article](article.pdf).

## Repository Structure
The repository is divided into 2 execution segments:

### 1. Dataset Generation
Contains the csv, scripts and notebooks necessary to generate the training dataset. This can be found in the [dataset](dataset) folder.

### 2. Model Training
Contains the scripts and notebooks necessary to train the text-to-mesh model, as well as deploy the model to generate meshes not seen in training. All of this is found in the [project](project) folder.

## General Requirements
The project was made on a machine compatible with CUDA 12.2 and pytorch >= 2.2.0. In terms of hardware, a GPU with at least 6 GB of VRAM is required.
