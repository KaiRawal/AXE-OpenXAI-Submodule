# Utils
import time
import torch
import numpy as np
import pickle
import os
import argparse
import warnings

# ML models
from openxai.LoadModel import LoadModel

# Data loaders
from openxai.dataloader import return_loaders

# Explanation models
from openxai.Explainer import Explainer

# Evaluation methods
from openxai.evaluator import Evaluator

# Perturbation methods required for the computation of the relative stability metrics
from openxai.explainers.catalog.perturbation_methods import NormalPerturbation
from openxai.explainers.catalog.perturbation_methods import NewDiscrete_NormalPerturbation
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


parser = argparse.ArgumentParser(description="Generate explanations for ML models")
parser.add_argument("--model_name", type=str, choices=["lr", "ann"], help="Name of the ML model")
parser.add_argument("--data_name", type=str, choices=["heloc", "adult", "german", "compas"], help="Name of the dataset")
args = parser.parse_args()

model_name = args.model_name
data_name = args.data_name

# Load pretrained ml model
model = LoadModel(data_name=data_name,
                  ml_model=model_name,
                  pretrained=True)

# Get training and test loaders
loader_train, loader_test = return_loaders(data_name=data_name,
                                           download=True,
                                           batch_size=1)

data_mode = os.getenv('OPENXAI_DATA_MODE', default='TEST')

if data_mode.upper() == 'TRAIN':
    pass
elif data_mode.upper() == 'TEST':
    loader_test, loader_train = loader_train, loader_test
else:
    raise ValueError("Invalid MODE environment variable. Choose 'TRAIN' or 'TEST'.")



data_all = torch.FloatTensor(loader_train.dataset.data)

shap_init_start = time.time()
shap = Explainer(method='shap',
                 model=model,
                 dataset_tensor=data_all)
shap_init_end = time.time()

lime_init_start = time.time()
lime = Explainer(method='lime',
                 model=model,
                 dataset_tensor=data_all)
lime_init_end = time.time()

smthgrad_init_start = time.time()
smthgrad = Explainer(method='smthgrad',
                     model=model,
                     dataset_tensor=data_all)
smthgrad_init_end = time.time()

vnlagrad_init_start = time.time()
vnlagrad = Explainer(method='vnlagrad',
                     model=model,
                     dataset_tensor=data_all)
vnlagrad_init_end = time.time()

integrad_init_start = time.time()
integrad = Explainer(method='integrad',
                     model=model,
                     dataset_tensor=data_all)
integrad_init_end = time.time()

inputimegrad_init_start = time.time()
inputimegrad = Explainer(method='inputimegrad',
                         model=model,
                         dataset_tensor=data_all)
inputimegrad_init_end = time.time()

output_dir = f"EXPLANATIONS/explanations_{data_name}_{model_name}"
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

total_shap_time = 0
total_lime_time = 0
total_smthgrad_time = 0
total_vnlagrad_time = 0
total_integrad_time = 0
total_inputimegrad_time = 0

for i, data in enumerate(tqdm(loader_train, desc=f"Generating ALL Explanations {data_name} {model_name}", ncols=100)):
    inputs, labels = data
    labels = labels.type(torch.int64)

    # Get SHAP explanation for the current data point
    time_temp = time.time()
    shap_exp = shap.get_explanation(inputs.float(), label=labels)
    total_shap_time += time.time() - time_temp

    # Get LIME explanation for the current data point
    time_temp = time.time()
    lime_exp = lime.get_explanation(inputs.float(), label=labels)
    total_lime_time += time.time() - time_temp

    # Get SmoothGrad explanation for the current data point
    time_temp = time.time()
    smthgrad_exp = smthgrad.get_explanation(inputs.float(), label=labels)
    total_smthgrad_time += time.time() - time_temp

    # Get VNLAGrad explanation for the current data point
    time_temp = time.time()
    vnlagrad_exp = vnlagrad.get_explanation(inputs.float(), label=labels)
    total_vnlagrad_time += time.time() - time_temp

    # Get IntegratedGradients explanation for the current data point
    time_temp = time.time()
    integrad_exp = integrad.get_explanation(inputs.float(), label=labels)
    total_integrad_time += time.time() - time_temp

    # Get InputTimesGradient explanation for the current data point
    time_temp = time.time()
    inputimegrad_exp = inputimegrad.get_explanation(inputs.float(), label=labels)
    total_inputimegrad_time += time.time() - time_temp

    # Save the explanation to a file (e.g., as a PyTorch tensor)
    output_file_s = os.path.join(output_dir, f"shap_{i}.pt")
    torch.save(shap_exp, output_file_s)

    output_file_l = os.path.join(output_dir, f"lime_{i}.pt")
    torch.save(lime_exp, output_file_l)

    output_file_sm = os.path.join(output_dir, f"smthgrad_{i}.pt")
    torch.save(smthgrad_exp, output_file_sm)

    output_file_v = os.path.join(output_dir, f"vnlagrad_{i}.pt")
    torch.save(vnlagrad_exp, output_file_v)

    output_file_i = os.path.join(output_dir, f"integrad_{i}.pt")
    torch.save(integrad_exp, output_file_i)

    output_file_itg = os.path.join(output_dir, f"inputimegrad_{i}.pt")
    torch.save(inputimegrad_exp, output_file_itg)

shap_init_time = shap_init_end - shap_init_start
lime_init_time = lime_init_end - lime_init_start
smthgrad_init_time = smthgrad_init_end - smthgrad_init_start
vnlagrad_init_time = vnlagrad_init_end - vnlagrad_init_start
integrad_init_time = integrad_init_end - integrad_init_start
inputimegrad_init_time = inputimegrad_init_end - inputimegrad_init_start

with open(f"EXPLANATIONS/[META]times_{data_name}_{model_name}.txt", "w") as file:
    file.write(f"Init SHAP Time: {shap_init_time}\n")
    file.write(f"Init LIME Time: {lime_init_time}\n")
    file.write(f"Init SmoothGrad Time: {smthgrad_init_time}\n")
    file.write(f"Init VNLAGrad Time: {vnlagrad_init_time}\n")
    file.write(f"Init IntegratedGradients Time: {integrad_init_time}\n")
    file.write(f"Init InputTimesGradient Time: {inputimegrad_init_time}\n")
    
    file.write(f"SHAP Time: {total_shap_time}\n")
    file.write(f"LIME Time: {total_lime_time}\n")
    file.write(f"SmoothGrad Time: {total_smthgrad_time}\n")
    file.write(f"VNLAGrad Time: {total_vnlagrad_time}\n")
    file.write(f"IntegratedGradients Time: {total_integrad_time}\n")
    file.write(f"InputTimesGradient Time: {total_inputimegrad_time}\n")
    
    file.write(f"Total SHAP Time: {total_shap_time + shap_init_time}\n")
    file.write(f"Total LIME Time: {total_lime_time + lime_init_time}\n")
    file.write(f"Total SmoothGrad Time: {total_smthgrad_time + smthgrad_init_time}\n")
    file.write(f"Total VNLAGrad Time: {total_vnlagrad_time + vnlagrad_init_time}\n")
    file.write(f"Total IntegratedGradients Time: {total_integrad_time + integrad_init_time}\n")
    file.write(f"Total InputTimesGradient Time: {total_inputimegrad_time + inputimegrad_init_time}\n")



