import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import os
import warnings
from sklearn.metrics import auc

# ML models
from openxai.LoadModel import LoadModel

# Data loaders
from openxai.dataloader import return_loaders
from openxai.evaluator import Evaluator
from openxai.explainers.catalog.perturbation_methods import NormalPerturbation
from openxai.explainers.catalog.perturbation_methods import NewDiscrete_NormalPerturbation



parser = argparse.ArgumentParser(description="Evaluate explanations for ML models")
parser.add_argument("--data_name", type=str, choices=["heloc", "adult", "german", "compas"], help="Name of the dataset")
parser.add_argument("--model_name", type=str, choices=["lr", "ann", "adv1", "adv2", "adv3"], help="Name of the model")
parser.add_argument("--fa_only", type=str, default="false", help="Flag to indicate if only FA should be used")

args = parser.parse_args()


data_loader_batch_size = 1

model_name = args.model_name
data_name = args.data_name

fa_only = str(args.fa_only).lower() != "false"


output_dir = f"EVALUATIONS/openxai_{args.data_name}_{args.model_name}"
os.makedirs(f"{output_dir}/PLOTS", exist_ok=True)

loader_train, loader_test = return_loaders(data_name=data_name,
                                           download=True,
                                           batch_size=data_loader_batch_size)
data_mode = os.getenv('OPENXAI_DATA_MODE', 'TEST')

if data_mode.upper() == 'TRAIN':
    pass
elif data_mode.upper() == 'TEST':
    loader_test, loader_train = loader_train, loader_test
else:
    raise ValueError("Invalid MODE environment variable. Choose 'TRAIN' or 'TEST'.")



X = loader_train.dataset.data.astype('float32')
y = np.array(loader_train.dataset.targets.values).flatten().astype('int64')
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    model = LoadModel(data_name=data_name,
                ml_model=model_name,
                pretrained=True)

x_tensor = torch.FloatTensor(X)
y_pred = np.argmax(model(x_tensor).detach().numpy(), axis=1).flatten()





def _get_features(dataset):
    if dataset == 'compas':
        feature_types = ['c', 'd', 'c', 'c', 'd', 'd', 'd']
    # Adult feature types
    elif dataset == 'adult':
        feature_types = ['c'] * 6 + ['d'] * 7
    # Gaussian feature types
    elif dataset == 'synthetic':
        feature_types = ['c'] * 20
    # Heloc feature types
    elif dataset == 'heloc':
        feature_types = ['c'] * 23
    # German Credit Data feature metadata
    elif dataset == 'german':
        feature_types = ['c'] * 8 + ['d'] * 12
        feature_metadata = dict()
        feature_metadata['feature_n_cols'] = [1, 1, 1, 1, 1, 1, 1, 1, 4, 5, 10, 5, 5, 4, 3, 4, 3, 3, 4, 2]
        feature_metadata['feature_types'] = feature_types
        feature_types = feature_metadata

    return feature_types

def _get_perturbation(dataset):
    perturbation_mean = 0.0
    perturbation_std = 0.5
    perturbation_flip_percentage = 0.3
    if dataset == 'german':
        # use special perturbation class
        perturbation = NewDiscrete_NormalPerturbation("tabular",
                                                    mean=perturbation_mean,
                                                    std_dev=perturbation_std,
                                                    flip_percentage=perturbation_flip_percentage)

    else:
        perturbation = NormalPerturbation("tabular",
                                        mean=perturbation_mean,
                                        std_dev=perturbation_std,
                                        flip_percentage=perturbation_flip_percentage)
    return perturbation



def _generate_mask(explanation, top_k):
    mask_indices = torch.topk(explanation, top_k).indices
    mask = torch.zeros(explanation.shape) > 10
    for i in mask_indices:
        mask[i] = True
    return mask


def get_input_dict(data_row, all_data, explanation, model, top_k, y_pred):
    return {
        'x': torch.from_numpy(data_row).float(),
        'input_data': torch.from_numpy(all_data).float(),
        # explainer
        'explanation_x': explanation,
        'model': model,
        # perturbation
        'perturb_method': _get_perturbation(data_name),
        'perturb_max_distance': 0.4,
        'feature_metadata': _get_features(data_name),
        'top_k': top_k,
        'mask': _generate_mask(explanation.flatten(), top_k),
        'y_pred': torch.tensor(y_pred),
    }
#         p_norm
#         eval_metric
#         y
#         y_pred
#         L_map
        

# x
# model
# top_k
# perturb_method
# feature_metadata
# input_data
# max_perturb_distance

    pass

def get_predictive_metrics(k=3, mode='shap'):
    all_PGI = []
    all_PGU = []
    for i, point in tqdm(enumerate(X), desc=f'PRED_{data_name}_{k}_{mode}_{X.shape}'):
        arr = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            if mode == 'random':
                arr = torch.from_numpy(np.random.random(X.shape[1]))
            else:
                arr = torch.load(f'EXPLANATIONS/explanations_{data_name}_{model_name}/{mode}_{i}.pt').float()
        input_dict = get_input_dict(X[i].flatten(), X, arr, model, k, y_pred[i])
        evaluator = Evaluator(input_dict,
                              inputs=X[i].flatten(),
                              labels=None, 
                              model=model, 
                              explainer=None)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            all_PGI.append(evaluator.evaluate(metric='PGI'))
            all_PGU.append(evaluator.evaluate(metric='PGU'))
    
    np.save(f'{output_dir}/PGI_{data_name}_{model_name}_{mode}_{k}.npy', np.array(all_PGI))
    np.save(f'{output_dir}/PGU_{data_name}_{model_name}_{mode}_{k}.npy', np.array(all_PGU))
    return {'PGI': all_PGI, 'PGU': all_PGU}




def get_legacy_metrics(k=3, mode='shap'):
    all_FA = []
    all_SA = []
    all_RA = []
    all_RC = []
    all_SRA = []
    all_PRA = []
    # all_PGI = []
    # all_PGU = []
    # print(f'{data_name}_{k}_{mode} {X.shape}')
    for i, point in tqdm(enumerate(X), desc=f'GT_{data_name}_{k}_{mode}_{X.shape}'):
        arr = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            if mode == 'random':
                arr = torch.from_numpy(np.random.random(X.shape[1]))
            else:
                arr = torch.load(f'EXPLANATIONS/explanations_{data_name}_{model_name}/{mode}_{i}.pt').float()
        input_dict = {'y_pred': None, 'top_k': k, 'model': model, 'explanation_x': arr}
        evaluator = Evaluator(input_dict,
                              inputs=X[i].flatten(),
                              labels=None, 
                              model=model, 
                              explainer=None)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            all_FA.append(evaluator.evaluate(metric='FA')[0][0])
            if not fa_only:
                all_SA.append(evaluator.evaluate(metric='SA')[0][0])
                all_RA.append(evaluator.evaluate(metric='RA')[0][0])
                all_RC.append(evaluator.evaluate(metric='RC')[0][0])
                all_SRA.append(evaluator.evaluate(metric='SRA')[0][0])
                all_PRA.append(evaluator.evaluate(metric='PRA')[0][0])
                # all_PGI.append(evaluator.evaluate(metric='PGI')[0][0])
                # all_PGU.append(evaluator.evaluate(metric='PGU')[0][0])
        # break
    
    np.save(f'{output_dir}/FA_{data_name}_{model_name}_{mode}_{k}.npy', np.array(all_FA))
    if not fa_only:
        np.save(f'{output_dir}/SA_{data_name}_{model_name}_{mode}_{k}.npy', np.array(all_SA))
        np.save(f'{output_dir}/RA_{data_name}_{model_name}_{mode}_{k}.npy', np.array(all_RA))
        np.save(f'{output_dir}/RC_{data_name}_{model_name}_{mode}_{k}.npy', np.array(all_RC))
        np.save(f'{output_dir}/SRA_{data_name}_{model_name}_{mode}_{k}.npy', np.array(all_SRA))
        np.save(f'{output_dir}/PRA_{data_name}_{model_name}_{mode}_{k}.npy', np.array(all_PRA))
        # np.save(f'{output_dir}/PGI_{data_name}_{model_name}_{mode}_{k}.npy', np.array(all_PGI))
        # np.save(f'{output_dir}/PGU_{data_name}_{model_name}_{mode}_{k}.npy', np.array(all_PGU))
    return {'FA': all_FA, 'SA': all_SA, 'RA': all_RA, 'RC': all_RC, 'SRA': all_SRA, 'PRA': all_PRA}  # 'PGI': all_PGI, 'PGU': all_PGU}

random_means = {
    'FA': [],
    'SA': [],
    'RA': [],
    'RC': [],
    'SRA': [],
    'PRA': [],
    'PGI': [],
    'PGU': [],
}

shap_means = {str(key): [val for val in value] for key, value in random_means.items()}
lime_means = {str(key): [val for val in value] for key, value in random_means.items()}
vnlagrad_means = {str(key): [val for val in value] for key, value in random_means.items()}
smthgrad_means = {str(key): [val for val in value] for key, value in random_means.items()}
inputimegrad_means = {str(key): [val for val in value] for key, value in random_means.items()}
integrad_means = {str(key): [val for val in value] for key, value in random_means.items()}

si = 1
step = 2
if data_name == 'compas':
    step = 1

def process_mode(mode, k, means_dict):
    if not os.path.exists(f'{output_dir}/PGI_{data_name}_{model_name}_{mode}_{k}.npy'):
        get_predictive_metrics(k=k, mode=mode)
    if model_name == 'lr' and fa_only and not os.path.exists(f'{output_dir}/FA_{data_name}_{model_name}_{mode}_{k}.npy'):
        get_legacy_metrics(k=k, mode=mode)
    elif model_name == 'lr' and not fa_only and not os.path.exists(f'{output_dir}/SA_{data_name}_{model_name}_{mode}_{k}.npy'):
        get_legacy_metrics(k=k, mode=mode)
    else:
        pass
        # raise NotImplementedError
    means_dict['PGI'].append(np.mean(np.load(f'{output_dir}/PGI_{data_name}_{model_name}_{mode}_{k}.npy')))
    means_dict['PGU'].append(np.mean(np.load(f'{output_dir}/PGU_{data_name}_{model_name}_{mode}_{k}.npy')))
    if  model_name == 'lr':
        means_dict['FA'].append(np.mean(np.load(f'{output_dir}/FA_{data_name}_{model_name}_{mode}_{k}.npy')))
        if not fa_only:
            means_dict['SA'].append(np.mean(np.load(f'{output_dir}/SA_{data_name}_{model_name}_{mode}_{k}.npy')))
            means_dict['RA'].append(np.mean(np.load(f'{output_dir}/RA_{data_name}_{model_name}_{mode}_{k}.npy')))
            means_dict['RC'].append(np.mean(np.load(f'{output_dir}/RC_{data_name}_{model_name}_{mode}_{k}.npy')))
            means_dict['SRA'].append(np.mean(np.load(f'{output_dir}/SRA_{data_name}_{model_name}_{mode}_{k}.npy')))
            means_dict['PRA'].append(np.mean(np.load(f'{output_dir}/PRA_{data_name}_{model_name}_{mode}_{k}.npy')))

for i in range(si, X.shape[1], step):
    k = i
    process_mode('shap', k, shap_means)
    process_mode('lime', k, lime_means)
    process_mode('vnlagrad', k, vnlagrad_means)
    process_mode('smthgrad', k, smthgrad_means)
    process_mode('inputimegrad', k, inputimegrad_means)
    process_mode('integrad', k, integrad_means)

    np.random.seed(1)
    mode = 'random'
    k = i
    if not os.path.exists(f'{output_dir}/PGI_{data_name}_{model_name}_{mode}_{k}.npy'):
        get_predictive_metrics(k=k, mode=mode)
    if model_name == 'lr' and not os.path.exists(f'{output_dir}/FA_{data_name}_{model_name}_{mode}_{k}.npy'):
        get_legacy_metrics(k=k, mode=mode)
    elif model_name == 'lr' and not fa_only and not os.path.exists(f'{output_dir}/SA_{data_name}_{model_name}_{mode}_{k}.npy'):
        get_legacy_metrics(k=k, mode=mode)
    else:
        pass
        # raise NotImplementedError
    random_means['PGI'].append(np.mean(np.load(f'{output_dir}/PGI_{data_name}_{model_name}_{mode}_{k}.npy')))
    random_means['PGU'].append(np.mean(np.load(f'{output_dir}/PGU_{data_name}_{model_name}_{mode}_{k}.npy')))
    if model_name == 'lr':
        random_means['FA'].append(np.mean(np.load(f'{output_dir}/FA_{data_name}_{model_name}_{mode}_{k}.npy')))
        if not fa_only:
            random_means['SA'].append(np.mean(np.load(f'{output_dir}/SA_{data_name}_{model_name}_{mode}_{k}.npy')))
            random_means['RA'].append(np.mean(np.load(f'{output_dir}/RA_{data_name}_{model_name}_{mode}_{k}.npy')))
            random_means['RC'].append(np.mean(np.load(f'{output_dir}/RC_{data_name}_{model_name}_{mode}_{k}.npy')))
            random_means['SRA'].append(np.mean(np.load(f'{output_dir}/SRA_{data_name}_{model_name}_{mode}_{k}.npy')))
            random_means['PRA'].append(np.mean(np.load(f'{output_dir}/PRA_{data_name}_{model_name}_{mode}_{k}.npy')))
    

def make_plots(metric):
    long_metric = None
    if metric == 'FA':
        long_metric = 'Feature Agreement'
    elif metric == 'SA':
        long_metric = 'Sign Agreement'
    elif metric == 'RA':
        long_metric = 'Rank Agreement'
    elif metric == 'RC':
        long_metric = 'Rank Correlation'
    elif metric == 'SRA':
        long_metric = 'Signed Rank Agreement'
    elif metric == 'PRA':
        long_metric = 'Pairwise Rank Agreement'
    elif metric == 'PGI':
        long_metric = 'Prediction Gap on Important'
    elif metric == 'PGU':
        long_metric = 'Prediction Gap on Unimportant'
    else:
        pass

    # print(f'{X.shape=}')
    # print(f'{len(shap_means[metric])=}')
    # print(f'{len(random_means[metric])=}')
    # print(f'{len(lime_means[metric])=}')
    # print(f'{len(vnlagrad_means[metric])=}')
    # print(f'{len(smthgrad_means[metric])=}')
    # print(f'{len(inputimegrad_means[metric])=}')
    # print(f'{len(integrad_means[metric])=}')

    sns.lineplot(y=shap_means[metric], x=[i for i in range(si, X.shape[1], step)], label='Shap')
    sns.lineplot(y=random_means[metric], x=[i for i in range(si, X.shape[1], step)], label='random')
    sns.lineplot(y=lime_means[metric], x=[i for i in range(si, X.shape[1], step)], label='LIME')
    sns.lineplot(y=vnlagrad_means[metric], x=[i for i in range(si, X.shape[1], step)], label='VNLA Grad')
    sns.lineplot(y=smthgrad_means[metric], x=[i for i in range(si, X.shape[1], step)], label='SmoothGrad')
    sns.lineplot(y=inputimegrad_means[metric], x=[i for i in range(si, X.shape[1], step)], label='Input Times Grad')
    sns.lineplot(y=integrad_means[metric], x=[i for i in range(si, X.shape[1], step)], label='Integrated Gradients')
    
    print()
    print(f'SHAP {long_metric} MEAN = {np.mean(shap_means[metric])} AUC = {auc(np.array([i for i in range(si,X.shape[1],step)]) / X.shape[1], shap_means[metric])}')
    print(f'Random {long_metric} MEAN = {np.mean(random_means[metric])} AUC = {auc(np.array([i for i in range(si,X.shape[1],step)]) / X.shape[1], random_means[metric])}')
    print(f'LIME {long_metric} MEAN = {np.mean(lime_means[metric])} AUC = {auc(np.array([i for i in range(si,X.shape[1],step)]) / X.shape[1], lime_means[metric])}')
    print(f'VNLA Grad {long_metric} MEAN = {np.mean(vnlagrad_means[metric])} AUC = {auc(np.array([i for i in range(si,X.shape[1],step)]) / X.shape[1], vnlagrad_means[metric])}')
    print(f'SmoothGrad {long_metric} MEAN = {np.mean(smthgrad_means[metric])} AUC = {auc(np.array([i for i in range(si,X.shape[1],step)]) / X.shape[1], smthgrad_means[metric])}')
    print(f'Input Times Grad {long_metric} MEAN = {np.mean(inputimegrad_means[metric])} AUC = {auc(np.array([i for i in range(si,X.shape[1],step)]) / X.shape[1], inputimegrad_means[metric])}')
    print(f'Integrated Gradients {long_metric} MEAN = {np.mean(integrad_means[metric])} AUC = {auc(np.array([i for i in range(si,X.shape[1],step)]) / X.shape[1], integrad_means[metric])}')
    
    plt.xlabel("Number of (Important) Features Selected")
    plt.ylabel(long_metric)
    plt.legend(title='Explanation Algorithm')
    plt.title(f'{long_metric} wrt Ground Truth ({model_name} on {data_name})')
    plt.ylim(0,1)
    plt.savefig(f"{output_dir}/PLOTS/{metric}_{data_name}_{model_name}.png")
    plt.clf()
    print(f"{output_dir}/PLOTS/{metric}_{data_name}_{model_name}.png")

if model_name == 'lr':
    make_plots('FA')
    make_plots('PGI')
    make_plots('PGU')
    if not fa_only:
        make_plots('SA')
        make_plots('RA')
        make_plots('RC')
        make_plots('SRA')
        make_plots('PRA')
elif model_name == 'ann':
    make_plots('PGI')
    make_plots('PGU')
else:
    raise NotImplementedError
