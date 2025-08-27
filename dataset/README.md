# Dataset Generation
The files in this folder allow generating the .pt, .pkl and .csv files necessary to train the text-to-3D mesh model.

## Original Dataset
The original dataset from which we started to generate the dataset used to train the model is [MedShapeNet](https://medshapenet-ikim.streamlit.app/). This dataset contains more than 100 thousand biomedical meshes in STL format, each with its description in plain text. Due to its large size, only a portion of 3,589 meshes was taken for training corresponding to the organ categories:

- aorta
- liver
- kidney

This sampling is found in the [organ csv file](organs_selected.csv), where for each example the following fields are available:
- **Name:** Name of the STL file.
- **Category:** Textual description of what the mesh contains.
- **URL**: Download link for the STL file.
- **Subcategory:** Textual subcategory of what the mesh contains, although it says the same as Category.

## Prerequisites Installation
To generate the training meshes it is required to install blender and shap-e, the first being a 3D object manipulation program and the second the model that will be trained in future steps, but which is used in this case to generate the input representation required for training, i.e., the latent states.

### Environment Generation
It is recommended to generate a Python environment within this folder with the command: `python -m venv datasetvenv` and activate it with the command `source datasetenv/bin/activate` to avoid dependency problems.

### Requirements Installation
Once the environment is activated, various prerequisites are installed with the command `pip install -r requirements.txt`

### Blender Installation
To install blender it is necessary to download the executable script from the official page, in version 3.3.1, with the following shell commands:
 ```
sudo apt-get install xvfb
wget https://download.blender.org/release/Blender3.3/blender-3.3.1-linux-x64.tar.xz
tar -xf blender-3.3.1-linux-x64.tar.xz
 ```
The path where the `blender-3.3.1-linux-x64` folder remains is of vital importance because the other scripts depend on finding blender to generate the dataset. For example, if the steps were executed from the downloads folder, the blender path will be as follows: `~/Downloads/blender-3.3.1-linux-x64/blender`

### Shap-e Installation
To install Shap-e the following command is used that installs from the OpenAI repository: `pip install "git+https://github.com/openai/shap-e"`.

### Possible Errors
If installation errors occur when installing `Shap-e` or the `requirements.txt` file, it is recommended to recreate the environment and install `Shap-e` first before `requirements.txt`.

## Training Mesh Creation
To create the training meshes there is a two-step execution process, in which first the .pt file folder is created, i.e., the latent states, and then these meshes are used to generate train-val-test partitions.

### 1. Creation of .pt latent states
To create the .pt latent states the Python script [`encode_model_blender.py`](encode_model_blender.py) is used, which must be executed from this folder with the command `python encode_model_blender.py`. Before its execution the following steps must be followed:

- Change the blender path in line 28 `os.environ["BLENDER_PATH"] = "./blender-3.3.1-linux-x64/blender"` to the one found in the `Blender Installation` step.
- Change the sample size in line 22 `SAMPLE_SIZE = 600` to the number of examples with which you want to train.

After executing this script, a `dataset_obj` folder will be generated with a `.pt` file for each selected example. Each of these files is generated in a process where the original STL file is downloaded, converted to OBJ format and then converted to a torch tensor with latent states in `.pt` format using the `transmitter` module from `Shap-e`.

A few [latent states](dataset_obj) are attached for review, but it is recommended to run the script again to download all latents, since with 3,589 examples this process took 10 days and culminated in a 15 GB folder.


### 2. Division into train-val-test
To divide into train-val-test the notebook [`train_test_split.ipynb`](train_test_split.ipynb) is used, which can be used from jupyter notebook. To launch jupyter notebook activate the environment and then execute the command `jupyter notebook`. Then, from the IPython graphical interface you can open this notebook, where you can choose, in the second cell, the following configurations:

- **TEST_SIZE:** Portion that will be used for validation and test.
- **VAL_SIZE:** Portion that will be used from validation and test for validation.
- **DATASET_CSV:** CSV file that has the original dataset information and it is recommended not to change.
- **DATASET_FOLDER:** Folder that has the latent states and it is recommended not to change.

Once these variables are configured you can execute the complete notebook, to generate the following files:

- **Train-val-test partitions:** These are three .pkl files that contain the ids of the .pt files that correspond to the train, val and test partitions, being these respectively: [`train_organs.pickle`](train_organs.pickle), [`val_organs.pickle`](val_organs.pickle), [`test_organs.pickle`](test_organs.pickle). 
- **Id and description tuples:** This is the [`organs_selected_complete.csv`](organs_selected_complete.csv) file that contains all pairs of latent id and example description that can be found in the partitions.
