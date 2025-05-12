import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import faiss
from tqdm import tqdm
import argparse
import os
from sklearn.metrics import auc

# ML models
from openxai.LoadModel import LoadModel

# Data loaders
from openxai.dataloader import return_loaders

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)



parser = argparse.ArgumentParser(description="Evaluate explanations for ML models")
parser.add_argument("--model_name", type=str, choices=["lr", "ann"], help="Name of the ML model")
parser.add_argument("--data_name", type=str, choices=["heloc", "adult", "german", "compas"], help="Name of the dataset")
args = parser.parse_args()


output_dir = f"EVALUATIONS/axe_{args.data_name}_{args.model_name}"
os.makedirs(f"{output_dir}/PLOTS", exist_ok=True)


def top_k_indices(arr, n):
    arr_copy = arr.copy().flatten()
    absolute_values = np.abs(arr_copy)
    sorted_indices = np.argsort(absolute_values)[::-1]
    top_n_indices = sorted_indices[:n]
    return top_n_indices


def build_knn_index(X, included_columns):

    excluded_columns = [i for i in range(X.shape[1]) if i not in included_columns]

    data_with_exclusions = np.delete(X, excluded_columns, axis=1)

    d = data_with_exclusions.shape[1]
    index = faiss.IndexFlatL2(d)

    index.add(data_with_exclusions)

    return index

def query_subset(point, included_columns):
    return point.flatten()[np.isin(np.arange(len(point)), sorted(int(item) for item in included_columns))]


def query_knn_index(index, labels, query_point, k):
    query_point = np.array(query_point).flatten().astype('float32')

    # Search for k-NN neighbors
    D, I = index.search(query_point.reshape(1, -1), k+1)

    # Calculate the predicted label (e.g., majority vote)
    k_nearest_labels = labels[I[0][1:]]
    predicted_label = np.bincount(k_nearest_labels).argmax()

    return predicted_label



data_loader_batch_size = 1

model_name = args.model_name
data_name = args.data_name


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
model = LoadModel(data_name=data_name,
              ml_model=model_name,
              pretrained=True)

x_tensor = torch.FloatTensor(X)
y_pred = np.argmax(model(x_tensor).detach().numpy(), axis=1).flatten()


np.save(f'{output_dir}/{data_name}_{model_name}_y_pred', np.array(y_pred).flatten())
np.save(f'{output_dir}/{data_name}_{model_name}_y_true', np.array(y).flatten())

def run_exp_random_importances(NUM_FEATURES, k):
    HYPERPARAM = k

    arr = np.random.random(X.shape[1])
    selected_columns = tuple(sorted(top_k_indices(arr, NUM_FEATURES)))
    index = build_knn_index(X, selected_columns)

    results = []
    for i, point in enumerate(X):
        results.append(
            query_knn_index(index, y_pred, query_subset(point, selected_columns), HYPERPARAM))

    np.save(
        f'{output_dir}/{data_name}_{model_name}_y_pred_top_{NUM_FEATURES}_RANDOM_features_{HYPERPARAM}-NN',
        np.array(results))
    return results

def run_exp_crandom_importances(NUM_FEATURES, k):
    HYPERPARAM = k
    indices = {}

    results = []
    for i, point in enumerate(X):
        arr = np.random.random(X.shape[1])
        selected_columns = tuple(sorted(top_k_indices(arr, NUM_FEATURES)))
        if selected_columns not in indices:
            indices[selected_columns] = build_knn_index(X, selected_columns)
        index = indices[selected_columns]
        results.append(
            query_knn_index(index, y_pred, query_subset(point, selected_columns), HYPERPARAM))

    np.save(
        f'{output_dir}/{data_name}_{model_name}_y_pred_top_{NUM_FEATURES}_cRANDOM_features_{HYPERPARAM}-NN',
        np.array(results))
    return results

def run_exp_logistic_ground_truth(NUM_FEATURES, k):
    HYPERPARAM = k

    params = dict(model.named_parameters())['linear.weight']
    arr = params[0].float().detach() - params[1].float().detach()
    arr = np.array(arr).flatten()
    selected_columns = tuple(sorted(top_k_indices(arr, NUM_FEATURES)))
    index = build_knn_index(X, selected_columns)

    results = []
    for i, point in enumerate(X):
        results.append(
            query_knn_index(index, y_pred, query_subset(point, selected_columns), HYPERPARAM))

    np.save(
        f'{output_dir}/{data_name}_{model_name}_y_pred_top_{NUM_FEATURES}_COEFF_features_{HYPERPARAM}-NN',
        np.array(results))
    return results

def run_exp_logistic_cground_truth(NUM_FEATURES, k):
    HYPERPARAM = k
    indices = {}
    params = dict(model.named_parameters())['linear.weight']
    arr = params[0].float().detach() - params[1].float().detach()
    coeff = np.array(arr).flatten()
    


    results = []
    for i, point in enumerate(X):
        arr = coeff * point
        selected_columns = tuple(sorted(top_k_indices(arr, NUM_FEATURES)))
        if selected_columns not in indices:
            indices[selected_columns] = build_knn_index(X, selected_columns)
        index = indices[selected_columns]
        results.append(
            query_knn_index(index, y_pred, query_subset(point, selected_columns), HYPERPARAM))

    np.save(
        f'{output_dir}/{data_name}_{model_name}_y_pred_top_{NUM_FEATURES}_cCOEFF_features_{HYPERPARAM}-NN',
        np.array(results))
    return results

def _process_explanation(mode, i, indices, NUM_FEATURES):
    arr = torch.load(f'EXPLANATIONS/explanations_{data_name}_{model_name}/{mode}_{i}.pt').float().detach().numpy().flatten()
    selected_columns = tuple(sorted(top_k_indices(arr, NUM_FEATURES)))
    if selected_columns not in indices:
        indices[selected_columns] = build_knn_index(X, selected_columns)

def _process_and_append(mode, i, point, indices, NUM_FEATURES, results_list, HYPERPARAM):
    arr = torch.load(f'EXPLANATIONS/explanations_{data_name}_{model_name}/{mode}_{i}.pt').float().detach().numpy().flatten()
    selected_columns = tuple(sorted(top_k_indices(arr, NUM_FEATURES)))
    results_list.append(
        query_knn_index(indices[selected_columns], y_pred, query_subset(point, selected_columns), HYPERPARAM))

def run_exp(NUM_FEATURES, k):
    HYPERPARAM = k
    indices = {}

    shaps = []
    limes = []
    vnlagrads = []
    smthgrads = []
    inputimegrads = []
    integrads = []


    explanations = ['shap', 'lime', 'vnlagrad', 'smthgrad', 'inputimegrad', 'integrad']
    results_dict = {
        'shap': shaps,
        'lime': limes,
        'vnlagrad': vnlagrads,
        'smthgrad': smthgrads,
        'inputimegrad': inputimegrads,
        'integrad': integrads
    }

    for explanation in explanations:
        indices = {}
        for i, point in enumerate(tqdm(X, desc=f'Processing {explanation} explanations {NUM_FEATURES=} {k=}', disable=True)):
            _process_explanation(explanation, i, indices, NUM_FEATURES)
            _process_and_append(explanation, i, point, indices, NUM_FEATURES, results_dict[explanation], k)

        np.save(
            f'{output_dir}/{data_name}_{model_name}_y_pred_top_{NUM_FEATURES}_{explanation.upper()}_features_{HYPERPARAM}-NN',
            np.array(results_dict[explanation]))
    return shaps, limes, vnlagrads, smthgrads, inputimegrads, integrads

def get_plot(k):
    all_shaps = {}
    all_limes = {}
    all_vnlagrads = {}
    all_smthgrads = {}
    all_inputimegrads = {}
    all_integrads = {}
    
    all_rands = {}
    all_crands = {}
    print(f'Accuracy_{data_name}_{model_name}_{k}NN_preds')
    for nf in tqdm(range(1,X.shape[1]), desc=f'Accuracy_{data_name}_{model_name}_{k}NN_preds'):
        shaps, limes, vnlagrads, smthgrads, inputimegrads, integrads = None, None, None, None, None, None
        if not os.path.exists(f'{output_dir}/{data_name}_{model_name}_y_pred_top_{nf}_INTEGRAD_features_{k}-NN.npy'):
            shaps, limes, vnlagrads, smthgrads, inputimegrads, integrads = run_exp(NUM_FEATURES=nf, k=k)
        else:
            shaps = np.load(f'{output_dir}/{data_name}_{model_name}_y_pred_top_{nf}_SHAP_features_{k}-NN.npy')
            limes = np.load(f'{output_dir}/{data_name}_{model_name}_y_pred_top_{nf}_LIME_features_{k}-NN.npy')
            vnlagrads = np.load(f'{output_dir}/{data_name}_{model_name}_y_pred_top_{nf}_VNLAGRAD_features_{k}-NN.npy')
            smthgrads = np.load(f'{output_dir}/{data_name}_{model_name}_y_pred_top_{nf}_SMTHGRAD_features_{k}-NN.npy')
            inputimegrads = np.load(f'{output_dir}/{data_name}_{model_name}_y_pred_top_{nf}_INPUTIMEGRAD_features_{k}-NN.npy')
            integrads = np.load(f'{output_dir}/{data_name}_{model_name}_y_pred_top_{nf}_INTEGRAD_features_{k}-NN.npy')
        assert shaps is not None
        assert limes is not None
        assert vnlagrads is not None
        assert smthgrads is not None
        assert inputimegrads is not None
        assert integrads is not None

        all_shaps[nf] = shaps
        all_limes[nf] = limes
        all_vnlagrads[nf] = vnlagrads
        all_smthgrads[nf] = smthgrads
        all_inputimegrads[nf] = inputimegrads
        all_integrads[nf] = integrads

        results = None
        if not os.path.exists(
                f'{output_dir}/{data_name}_{model_name}_y_pred_top_{nf}_RANDOM_features_{k}-NN.npy'):
            results = run_exp_random_importances(NUM_FEATURES=nf, k=k)
        else:
            results = np.load(
                f'{output_dir}/{data_name}_{model_name}_y_pred_top_{nf}_RANDOM_features_{k}-NN.npy')
        assert results is not None
        all_rands[nf] = results

        results = None
        if not os.path.exists(
                f'{output_dir}/{data_name}_{model_name}_y_pred_top_{nf}_cRANDOM_features_{k}-NN.npy'):
            results = run_exp_crandom_importances(NUM_FEATURES=nf, k=k)
        else:
            results = np.load(
                f'{output_dir}/{data_name}_{model_name}_y_pred_top_{nf}_cRANDOM_features_{k}-NN.npy')
        assert results is not None
        all_crands[nf] = results


    all_imps = {}
    all_cimps = {}
    if model_name != 'ann':
        for nf in tqdm(range(1, X.shape[1]), desc=f'Accuracy_COEFF_{data_name}_{model_name}_{k}NN_preds'):
            results = None
            if not os.path.exists(
                    f'{output_dir}/{data_name}_{model_name}_y_pred_top_{nf}_COEFF_features_{k}-NN.npy'):
                results = run_exp_logistic_ground_truth(NUM_FEATURES=nf, k=k)
            else:
                results = np.load(
                    f'{output_dir}/{data_name}_{model_name}_y_pred_top_{nf}_COEFF_features_{k}-NN.npy')
            assert results is not None
            all_imps[nf] = results
            results = None
            if not os.path.exists(
                    f'{output_dir}/{data_name}_{model_name}_y_pred_top_{nf}_cCOEFF_features_{k}-NN.npy'):
                results = run_exp_logistic_cground_truth(NUM_FEATURES=nf, k=k)
            else:
                results = np.load(
                    f'{output_dir}/{data_name}_{model_name}_y_pred_top_{nf}_cCOEFF_features_{k}-NN.npy')
            assert results is not None
            all_cimps[nf] = results
    
    shap_accuracies = []
    lime_accuracies = []
    vnlagrad_accuracies = []
    smthgrad_accuracies = []
    inputimegrad_accuracies = []
    integrad_accuracies = []
    
    coeff_accuracies = []
    ccoeff_accuracies = []
    rand_accuracies = []
    crand_accuracies = []
    for nf in all_shaps:
        shap_accuracies.append((np.array(all_shaps[nf]) == y_pred).mean())
        lime_accuracies.append((np.array(all_limes[nf]) == y_pred).mean())
        vnlagrad_accuracies.append((np.array(all_vnlagrads[nf]) == y_pred).mean())
        smthgrad_accuracies.append((np.array(all_smthgrads[nf]) == y_pred).mean())
        inputimegrad_accuracies.append((np.array(all_inputimegrads[nf]) == y_pred).mean())
        integrad_accuracies.append((np.array(all_integrads[nf]) == y_pred).mean())
        
        rand_accuracies.append((np.array(all_rands[nf]) == y_pred).mean())
        crand_accuracies.append((np.array(all_crands[nf]) == y_pred).mean())
        if nf in all_imps:
            coeff_accuracies.append((np.array(all_imps[nf]) == y_pred).mean())
        if nf in all_cimps:
            ccoeff_accuracies.append((np.array(all_cimps[nf]) == y_pred).mean())
    
    sns.lineplot(y=shap_accuracies, x=[i for i in range(1, X.shape[1])], label='Shap')
    sns.lineplot(y=lime_accuracies, x=[i for i in range(1, X.shape[1])], label='LIME')
    sns.lineplot(y=vnlagrad_accuracies, x=[i for i in range(1, X.shape[1])], label='VNLA Grad')
    sns.lineplot(y=smthgrad_accuracies, x=[i for i in range(1, X.shape[1])], label='SmoothGrad')
    sns.lineplot(y=inputimegrad_accuracies, x=[i for i in range(1, X.shape[1])], label='Input Times Grad')
    sns.lineplot(y=integrad_accuracies, x=[i for i in range(1, X.shape[1])], label='Integrated Gradients')
    sns.lineplot(y=crand_accuracies, x=[i for i in range(1, X.shape[1])], label='corrected_random')
    # sns.lineplot(y=rand_accuracies, x=[i for i in range(1, X.shape[1])], label='random', linestyle=':')
    
    if len(coeff_accuracies) > 1:
        sns.lineplot(y=coeff_accuracies, x=[i for i in range(1, X.shape[1])], label='Model Weights')
    if len(ccoeff_accuracies) > 1:
        sns.lineplot(y=ccoeff_accuracies, x=[i for i in range(1, X.shape[1])], label='Weights * Features', linestyle='--')
    
    plt.xlabel("Number of (Important) Features Selected")
    plt.ylabel("Accuracy (wrt Initial Model Prediction)")
    plt.legend(title='Feature Selection Method')
    plt.title(f'Accuracies for {k}-NN* models, for an {model_name.upper()} model on {data_name.upper()} data')
    plt.savefig(f"{output_dir}/PLOTS/axe_{data_name}_{model_name}_{k}NN_preds.png")
    plt.clf()
    
    # Print the mean and AUC for each line
    metrics = {
        'Shap': shap_accuracies,
        'LIME': lime_accuracies,
        'VNLA Grad': vnlagrad_accuracies,
        'SmoothGrad': smthgrad_accuracies,
        'Input Times Grad': inputimegrad_accuracies,
        'Integrated Gradients': integrad_accuracies,
        'random': rand_accuracies,
        'corrected_random': crand_accuracies,
        'Model Weights': coeff_accuracies,
        'Weights * Features': ccoeff_accuracies
    }
    
    for label, accuracies in metrics.items():
        if len(accuracies) > 1:
            mean_accuracy = np.mean(accuracies)
            auc_value = auc(np.array([i for i in range(1, X.shape[1])]) / X.shape[1], accuracies)
            print(f'{label} MEAN = {mean_accuracy} AUC = {auc_value}')
    

import warnings
warnings.filterwarnings('ignore')

np.random.seed(1)

get_plot(1)
get_plot(3)
get_plot(5)
get_plot(9)
get_plot(15)
get_plot(31)
get_plot(65)



