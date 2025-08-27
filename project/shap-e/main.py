import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import torch
import streamlit as st

from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.util.notebooks import decode_latent_mesh
from shap_e.diffusion.sample import sample_latents

import pyvista as pv
from stpyvista import stpyvista

MODEL_FOLDER = './model_ckpts'
MESH_FILE = './mesh.ply'
RAW_STATE_DICT = "./raw_state.pt"

pv.start_xvfb()

def reset():
    del st.session_state["input_model"]
    del st.session_state["input_prompt"]
    st.cache_data.clear()
    st.cache_resource.clear()

if __name__ == "__main__": 
    st.title('Biomedical mesh generator')
    st.markdown(
        "This biomedical mesh generator allows the creation of a biomedical mesh starting from a Natural Language Text Description"
    )
    st.subheader(
        "Instructions"
    )
    st.markdown(
        """
        1. Choose the model version to predict and write your prompt. Model versions are determined by the number of epochs from the fine-tuning process.
        2. View the mesh interactively.
        3. If you like the result, download the object in .ply format by clicking the "Download" button.
        4. If you want to generate a new mesh, press the "New experiment" button and set the new configuration.
        """
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with st.spinner('Loading base model'):
        if 'transmitter' not in st.session_state:
            st.session_state['transmitter'] = load_model('transmitter', device=device)
        if 'model' not in st.session_state:
            st.session_state['model'] = load_model('text300M', device=device)
            torch.save(st.session_state['model'].state_dict(), RAW_STATE_DICT)
        if 'diffusion' not in st.session_state:
            st.session_state['diffusion'] = diffusion_from_config(load_config('diffusion'))
        if 'batch_size' not in st.session_state:
            st.session_state['batch_size'] = 1
        if 'guidance_scale' not in st.session_state:
            st.session_state['guidance_scale'] = 15.0

    model_choices = [
        model
        for model in os.listdir(MODEL_FOLDER)
    ]

    st.selectbox(
        label="Choose a model version",
        options= model_choices + ["Pretrained model"],
        index=None,
        key="input_model"
    )
    
    prompt = st.text_input(
        "Write your prompt here",
        value="",
        key="input_prompt"
    )

    if st.session_state["input_model"]:
        with st.spinner('Loading weights'):
            if st.session_state["input_model"] in model_choices:
                weights = st.session_state['model'].load_state_dict(
                    torch.load(
                        os.path.join(MODEL_FOLDER, st.session_state["input_model"]),
                        map_location=device
                    )['model_state_dict']
                )
            elif st.session_state["input_model"] in ["Pretrained model"]:
                weights = st.session_state['model'].load_state_dict(
                    torch.load(
                        RAW_STATE_DICT
                    )
                )

        if st.session_state["input_prompt"]:
            with st.spinner('Generating mesh'):
                latents = sample_latents(
                    batch_size=st.session_state['batch_size'],
                    model=st.session_state['model'],
                    diffusion=st.session_state['diffusion'],
                    guidance_scale=st.session_state['guidance_scale'],
                    model_kwargs=dict(texts=[prompt] * st.session_state['batch_size']),
                    progress=True,
                    clip_denoised=True,
                    use_fp16=True,
                    use_karras=True,
                    karras_steps=64,
                    sigma_min=1e-3,
                    sigma_max=160,
                    s_churn=0
                )
                with torch.no_grad():
                    gen_mesh = decode_latent_mesh(st.session_state['transmitter'], latents).tri_mesh()
                    with open(os.path.join(MESH_FILE), 'wb') as f:
                        gen_mesh.write_ply(f)
            
                plotter = pv.Plotter(window_size=[400, 400])
                mesh = pv.read(MESH_FILE)
                plotter.add_mesh(mesh, name='mesh', cmap='bwr')
                plotter.view_isometric()
                plotter.background_color = 'black'
                stpyvista(plotter, key="mesh")

                with open(MESH_FILE, 'rb') as f:
                    col1, col2, col3 = st.columns([1,1,1])
                    with col1:
                        down_btn = st.download_button(
                            label="Download",
                            data=f,
                            file_name=f"{prompt}.ply",
                            mime="application/octet-stream"
                          )
                    with col3:
                        new_btn = st.button(
                            label="New experiment",
                            on_click=reset
                        )
