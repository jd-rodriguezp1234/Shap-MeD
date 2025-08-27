import gc
import torch
import json

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.models.configs import model_from_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget, decode_latent_mesh
import os
import time
import argparse
import random
from IPython import embed
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import pickle


# CONFIGS
LATENTS_FOLDER = '/home/estudiante/dataset/dataset_obj/'

# Change model path if empty string pretrained weights are used
MODEL_PATH = "" #'./model_ckpts/med_shape_e_epoch24.pth'


# CODE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class shapE_val_dataset(Dataset):
    def __init__(self, latent_code_path):
        self.captions = pd.read_csv('/home/estudiante/dataset/organs_selected_complete.csv', header=None)
        self.valid_uid = list(pickle.load(open('/home/estudiante/dataset/test_organs.pickle','rb')))
        self.final_uid = self.valid_uid
        self.n2idx = {}
        for i in range(len(self.captions)):
            self.n2idx[self.captions[0][i]] = i
        self.latent_code_path = latent_code_path

    def __len__(self):
        return len(self.final_uid)

    def __getitem__(self, i):
        idx = self.n2idx[self.final_uid[i]]
        assert self.final_uid[i] == self.captions[0][idx]
        latent = torch.load(os.path.join(self.latent_code_path,self.captions[0][idx]+'.pt')).squeeze()

        return {'caption': self.captions[1][idx], 'latent': latent}

print("Creating dataset")
my_dataset_test = shapE_val_dataset(LATENTS_FOLDER)
data_loader = DataLoader(my_dataset_test, batch_size=1, shuffle=False, drop_last=True)

print("Loading transmitter")
xm = load_model('transmitter', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

print("Loading model")
model = load_model('text300M', device=device)
if not os.path.exists("test_metrics"):
    os.mkdir("test_metrics")
if MODEL_PATH != "":
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model_state_dict'])
    saving_file = MODEL_PATH.split("/")[-1].split(".")[0]
    saving_file = f"test_metrics/{saving_file}.json"
else:
    saving_file = f"test_metrics/pretrained.json"

print(f"Evaluating {MODEL_PATH} on test")
with torch.no_grad():                
    test_losses = []
    for j, datatest in tqdm(enumerate(data_loader)):
        prompt = datatest['caption']
        model_kwargs=dict(texts=prompt)
        t = torch.randint(0, load_config('diffusion')['timesteps'], size=(1,), device=device)
        test_x_start = datatest['latent'].cuda()
        test_loss = diffusion.training_losses(model, test_x_start, t, model_kwargs=model_kwargs)
        test_final_loss = torch.mean(test_loss['loss'])
        test_losses.append(test_final_loss.item())
        gc.collect()
        torch.cuda.empty_cache()
    test_mean_loss = torch.mean(torch.Tensor(test_losses)).item()

metrics = {"test_loss": test_mean_loss}
with open(saving_file, "w") as f:
    json.dump(metrics, f)