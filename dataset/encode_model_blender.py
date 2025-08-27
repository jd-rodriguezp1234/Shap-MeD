import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import signal
import functools

import logging

import torch
import shutil

from shap_e.models.download import load_model
from shap_e.util.data_util import load_or_create_multimodal_batch
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget

import pandas as pd
import numpy as np
import requests
import tempfile
from mpl_toolkits import mplot3d
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import open3d as o3d
from getpass import getpass


# CONSTANTS
SAMPLE_SIZE = 2500
RANDOM_STATE = 42
DATASET_PATH = "./dataset_obj"
CACHE_FOLDER = "temp_data_images"

## Blender env
os.environ["BLENDER_PATH"] = "./blender-3.3.1-linux-x64/blender"

# Set torch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Torch is using {device}")

# Load lantent model
xm = load_model('transmitter', device=device)

# Read dataset and downsample
organs_selected = pd.read_csv('./organs_selected.csv')
_, organs_selected = train_test_split(
    organs_selected,
    test_size=SAMPLE_SIZE,
    stratify=organs_selected["Category"],
    random_state=RANDOM_STATE
)
print("Class distribution\n", organs_selected["Category"].value_counts())
organs_selected["id"] = (
    [str(i) for i in range(len(organs_selected))]
    + organs_selected["Name"].apply(lambda x: x[:-4])
)
organs_selected.to_csv("./organs_selected_downsampled.csv")

# Methods
def read_stl(url):
    """Load stl as temp file"""
    response = requests.get(url)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name

    return temp_file_path

def convert_stl_to_obj(path_stl, path_obj):
  mesh = o3d.io.read_triangle_mesh(path_stl)
  o3d.io.write_triangle_mesh(path_obj, mesh)
  try:
    os.remove(path_stl)
  except:
    pass
  return path_obj

def timeout(seconds=5, default=None):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            def handle_timeout(signum, frame):
                raise TimeoutError()

            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(seconds)

            result = func(*args, **kwargs)

            signal.alarm(0)

            return result

        return wrapper

    return decorator

def create_latent_representation(path_obj, path_latent):
    batch = load_or_create_multimodal_batch(
        device,
        model_path=path_obj,
        mv_light_mode="basic",
        mv_image_size=256,
        cache_dir=CACHE_FOLDER,
        verbose=False, # this will show Blender output during renders
    )
    with torch.no_grad():
        latent = xm.encoder.encode_to_bottleneck(batch)
        torch.save(latent, path_latent)
    try:
        os.remove(path_obj)
    except:
        pass
    return latent

@timeout(seconds=300, default=None) # Object 817 or 818 is stuck forever!
def str_to_latent(path_stl, path_obj, path_latente):
    convert_stl_to_obj(path_stl, path_obj)
    create_latent_representation(path_obj, path_latente)

if not os.path.exists(DATASET_PATH):
    os.mkdir(DATASET_PATH)

#password = getpass("Enter password: ")
for index, row in tqdm(organs_selected.iterrows()):
    try:
        name = row.Name[:-4]
        url = row.URL
        path_latente = os.path.join(DATASET_PATH, f"{index}_{name}.pt")
        if os.path.exists(path_latente): # Skip ready files
            continue
        path_stl = read_stl(url)
        path_obj = path_stl.replace('.stl', '.obj')
        str_to_latent(path_stl, path_obj, path_latente)
        torch.cuda.empty_cache()
        shutil.rmtree(CACHE_FOLDER)
        #os.system(f"echo '{password}' | sudo -S rm -r /tmp/*")
    except Exception as e:
        print(f"Failed index: {index} with error message:")
        logging.exception(e)
        continue
