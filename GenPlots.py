import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import torch
import os
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)



def parse_axe(infile='PLOTS/axe_aucs.txt', outfile='PLOTS/axe_aucs.csv'):
    """
    reads the input file, which has lines formatted as:
    (example snippet)
    ```
    Accuracy_adult_lr_15NN_preds
    Shap MEAN = 0.9422977704072232 AUC = 0.798545732874091
    LIME MEAN = 0.9397733554449972 AUC = 0.7975124378109454
    VNLA Grad MEAN = 0.9433665008291875 AUC = 0.8008036739380022
    SmoothGrad MEAN = 0.9433665008291875 AUC = 0.8008036739380022
    Input Times Grad MEAN = 0.9330385111479638 AUC = 0.7928817451205512
    Integrated Gradients MEAN = 0.943430993182237 AUC = 0.8008547008547008
    random MEAN = 0.8877464529205822 AUC = 0.75209422970617
    corrected_random MEAN = 0.8879215035931454 AUC = 0.751775311476804
    Model Weights MEAN = 0.943430993182237 AUC = 0.8008547008547008
    Weights * Features MEAN = 0.933075363921135 AUC = 0.792915763065017
    Accuracy_german_ann_1NN_preds
    Shap MEAN = 0.9619491525423728 AUC = 0.9298333333333331
    LIME MEAN = 0.9654237288135594 AUC = 0.9332499999999998
    VNLA Grad MEAN = 0.9652542372881356 AUC = 0.9330833333333332
    SmoothGrad MEAN = 0.9650000000000001 AUC = 0.9328333333333333
    Input Times Grad MEAN = 0.9639830508474575 AUC = 0.9318333333333333
    Integrated Gradients MEAN = 0.9646610169491525 AUC = 0.9325
    random MEAN = 0.9282203389830509 AUC = 0.8997083333333332
    corrected_random MEAN = 0.9433050847457627 AUC = 0.9115833333333332
    ```

    and converts into a pandas dataframe (eventually written as csv to the outfile), 
    where each corresponding row is

    {
        'metric': 'Accuracy',
        'dataset': 'adult',
        'model': 'lr',
        'NN_preds': '15NN_preds',
        'Shap': 0.798545732874091,
        'LIME': 0.7975124378109454,
        'VNLA Grad': 0.8008036739380022,
        'SmoothGrad': 0.8008036739380022,
        'Input Times Grad': 0.7928817451205512,
        'Integrated Gradients': 0.8008547008547008,
        'random': 0.75209422970617,
        'corrected_random': 0.751775311476804,
        'Model Weights': 0.8008547008547008,
        'Weights * Features': 0.792915763065017,
    }

    """
    # Initialize an empty list to store the rows
    data = []

    # Read the input file
    with open(infile, 'r') as f:
        lines = f.readlines()

    # Process the lines
    i = 0
    while i < len(lines):
        # Extract metric, dataset, model, and NN_preds from the first line
        first_line = lines[i].strip()
        parts = first_line.split('_')
        metric = parts[0]
        dataset = parts[1]
        model = parts[2]
        
        nn_preds = str(parts[3])

        row = {
            # 'metric': metric,
            'dataset': dataset,
            'model': model,
            'NN_preds': nn_preds
        }

        # Process the next 10 lines for the AUC values
        num_lines = 11 if model == 'lr' else 9
        for j in range(1, num_lines):
            if i + j >= len(lines):
                break
            line = lines[i + j].strip()
            if line:
                match = re.search(r'(\w+(?: \* \w+)?) MEAN = [\d.]+ AUC = ([\d.]+)', line)
                if match:
                    method = line.split()[0]
                    auc = float(match.group(2))
                    row[method] = auc
        data.append(row)
        i += num_lines

    # Convert the list of rows into a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Rename some columns from the df
    df.rename(columns={
        'Integrated': 'ig',
        'Input': 'itg',
        'SmoothGrad': 'sg',
        'VNLA': 'vnla',
        'LIME': 'lime',
        'random': 'random_error',
        'corrected_random': 'random',
        'Model': 'coeffs',
        'Weights': 'wts_time_feats',
        'Shap': 'shap',
        'NN_preds': 'axe_eval',
    }, inplace=True)
    df = add_ranks(df)
    df['id'] = df['axe_eval']
    # Write the DataFrame to a CSV file
    df.to_csv(outfile, index=False)
    return df


def parse_openxai(infile='PLOTS/openxai_aucs.txt', outfile='PLOTS/openxai_aucs.csv'):
    """
    reads the input file, which has 800 lines formated as:
    (example 10 line snippet)
    ```
            SHAP Prediction Gap on Unimportant MEAN = 0.04941311851143837 AUC = 0.048059125430881984
            Random Prediction Gap on Unimportant MEAN = 0.047454580664634705 AUC = 0.046134878012041254
            LIME Prediction Gap on Unimportant MEAN = 0.037427403032779694 AUC = 0.03611012507850925
            VNLA Grad Prediction Gap on Unimportant MEAN = 0.06255792826414108 AUC = 0.06123557719402015
            SmoothGrad Prediction Gap on Unimportant MEAN = 0.06244437396526337 AUC = 0.06113789368731279
            Input Times Grad Prediction Gap on Unimportant MEAN = 0.04731715843081474 AUC = 0.045973391272127614
            Integrated Gradients Prediction Gap on Unimportant MEAN = 0.04401472583413124 AUC = 0.04269631635397672
            ../EVALUATION_RESULTS/LEGACY/PLOTS/PGU_german_lr.png
    ```

    and converts into a pandas dataframe (eventually written as csv to the outfile), 
    where the corresponding row for these 10 lines is

    {
        'metric': 'PGU',
        'dataset': 'german',
        'model': 'lr',
        'SHAP': 0.048059125430881984,
        'Random': 0.046134878012041254,
        'LIME': 0.03611012507850925,
        'VNLA Grad': 0.06123557719402015
        'SmoothGrad': 0.0611378936873127,
        'Input Times Grad': 0.045973391272127614,
        'Integrated Gradients': 0.04269631635397672,
    }

    """
    # Initialize an empty list to store the rows
    data = []

    # Read the input file
    with open(infile, 'r') as f:
        lines = f.readlines()

    # Process every 9 lines as a single record
    for i in range(0, len(lines), 9):
        record = lines[i:i+9]
        row = {}
        
        # Extract metric, dataset, and model from the last line
        last_line = record[-1].strip()
        match = re.search(r'EVALUATIONS/openxai_(\w+)_(\w+)/PLOTS/(\w+)_(\w+)_(\w+).png', last_line)
        if match:
            row['metric'] = match.group(3)
            row['dataset'] = match.group(4)
            row['model'] = match.group(5)
        
        # Extract AUC values from the first 7 lines
        for line in record[:-1]:
            parts = line.split('AUC = ')
            if len(parts) == 2:
                
                key = parts[0].split()[0].strip()
                
                value = float(parts[1].strip())
                row[key] = value
        
        data.append(row)

    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(data)
    # Rename some columns from the df
    df.rename(columns={
        'Integrated': 'ig',
        'Input': 'itg',
        'SmoothGrad': 'sg',
        'VNLA': 'vnla',
        'LIME': 'lime',
        'Random': 'random',
        'SHAP': 'shap',
        'metric': 'openxai_eval',
    }, inplace=True)
    # df['ground_truth'] = ['legacy'] * 40 + ['corrected'] * 40
    df['ground_truth'] = 'legacy'
    df = add_ranks(df)
    df.loc[df['openxai_eval'] == 'PGU', 'ranked_columns'] = df.loc[df['openxai_eval'] == 'PGU', 'ranked_columns'].apply(lambda x: x[::-1])
    df['id'] = [f'{oeid}_{gt}' for gt, oeid in zip(list(df['ground_truth']), list(df['openxai_eval']))]
    # Save the DataFrame to a CSV file
    df.to_csv(outfile, index=False)
    return df
    pass


def add_ranks(df):
    # subset = ['ig', 'itg', 'sg', 'vnla', 'lime', 'random', 'shap']
    subset = ['ig', 'itg', 'sg', 'vnla', 'lime', 'shap', 'random']
    def rank_columns(row):
        return [col for col, _ in sorted(row[subset].items(), key=lambda item: item[1], reverse=True)]

    df['ranked_columns'] = df.apply(rank_columns, axis=1)
    return df
    pass

# def get_id(df):
#     pass

# def corr_matrix(df_X, df_Y, outfile):
#     """
#     both dfs have 2 columns: 'id' and 'ranked_columns'.
#     the output plot is saved to outfile

#     the plot consists of a correlation matrix,
#     the elements on the X and Y axes are the id's from the df's
#     the values in the matrix are rank correlations between the ranked_column's
#     (ranked_columns are lists of equal length that always contain the same elements)
#     """

#     # Create a DataFrame to store the correlation values
#     ids_X = df_X['id'].tolist()
#     ids_Y = df_Y['id'].tolist()
#     corr_matrix = pd.DataFrame(index=ids_X, columns=ids_Y)

#     # Calculate the rank correlations
#     for id_X in ids_X:
#         for id_Y in ids_Y:
#             ranks_X = df_X[df_X['id'] == id_X]['ranked_columns'].values[0]
#             ranks_Y = df_Y[df_Y['id'] == id_Y]['ranked_columns'].values[0]
#             corr, _ = spearmanr(ranks_X, ranks_Y)
#             corr_matrix.at[id_X, id_Y] = corr

#     # Convert the correlation matrix to numeric values
#     corr_matrix = corr_matrix.astype(float)

#     # Plot the correlation matrix
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
#     plt.title('Rank Correlation Matrix')
#     plt.xlabel('Y IDs')
#     plt.ylabel('X IDs')
#     plt.savefig(outfile)
#     plt.close()

# def _get_exps(dataset, model, explainer):
#     e2f = {
#         'vnla' : 'vnlagrad',
#         'ig': 'integrad',
#         'itg': 'inputimegrad',
#         'sg': 'smthgrad',
#         'shap': 'shap',
#         'lime': 'lime',
#     }
#     prefix = e2f[explainer]

#     explanations = []
#     i = 0
#     while True:
#         file_path = f'../EXPERIMENT_OUTPUTS/explanations_{dataset}_{model}/{prefix}_{i}.pt'
#         if not os.path.exists(file_path):
#             break
#         explanation = torch.load(file_path)
#         explanation = explanation.detach().numpy().flatten()
#         explanations.append(explanation)
#         i += 1

#     if explanations:
#         df = pd.DataFrame(explanations)
#         df.index = range(len(explanations))
#         output_path = f'../CORI/explanations/{dataset}_{model}_{prefix}.csv'
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         df.to_csv(output_path, index_label='index')
#         return df
#     else:
#         raise ValueError(f'No explanations stored for {dataset=} {model=} {explainer=}')

#     pass

# def corri_matrix(df_X, df_Y, dataset, model):
#     outfile = f'../CORI/PLOTS/{dataset}_{model}.png'
#     explainers = ['ig', 'itg', 'sg', 'vnla', 'lime', 'shap']
#     all_explanations = {}
#     for explainer in explainers:
#         all_explanations[explainer] = _get_exps(dataset, model, explainer)
    
    
#     pass

def plot_dataset_model(data, dataset, model, outfile):
    ds2disp = {
        'german': 'German Credit',
        'adult': 'Adult Income',
        'heloc': 'HELOC',
        'compas': 'COMPAS',
    }
    model_to_title = {
        'ann': f'Neural Net on {ds2disp[dataset]} Dataset',
        'lr': f'Log. Reg. on {ds2disp[dataset]} Dataset'
    }
    col2lbl = {
        'lime': 'LIME',
        'shap': 'SHAP',
        'ig': 'Integrated Grad',
        'itg': r'Input $\times$ Grad',
        'sg': 'Smooth Grad',
        'vnla': 'Grad',
        'random': 'Random',
    }
    plt.figure(figsize=(10, 6))
    
    
    for idx, column in enumerate(['shap', 'vnla', 'ig', 'lime', 'sg', 'itg', 'random']):
        
        if column != 'eval_metric':
            offset = idx * 0.05  # Add a small offset to each line

            markers = ['s', 'o', '^', 'v', 'x', '1', '+']
            linestyles = ['-', ':', ':', '-', ':', ':', '--']
            plt.plot(data['eval_metric'], data[column] + offset, label=col2lbl[column], marker=markers[idx % len(markers)], linestyle=linestyles[idx % len(linestyles)], linewidth=2.5, markersize=10, markeredgewidth=2)
            
    plt.xlabel('Evaluation Metric', fontsize=16, fontweight='bold')
    plt.ylabel('Avg. Evaluation Scores \n (Standardized: $\mu = 0$, $\sigma = 1$)', fontsize=16, fontweight='bold', multialignment='center')
    plt.title(model_to_title[model], fontsize=22, fontweight='bold')
    
    if dataset == 'adult':
        plt.legend(loc='lower left', fontsize=14, title='Explainer', title_fontsize='14', framealpha=0.2)
    
    plt.axvline(x=3.5, color='gray', linestyle='--')
    
    plt.xticks(fontsize=15, fontweight='bold')  # Make xticks font larger and bolder
    plt.xticks(rotation=45)
    
    plt.yticks([-2, -1, 0, 1, 2], fontsize=15)  # Set yticks to fixed values and make font larger and bolder

    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    
    plt.close()

def merge_and_plot(inaxe='PLOTS/axe_aucs.csv', inopenx='PLOTS/openxai_aucs.csv', outdir='./PLOTS/'):
    df_axe = pd.read_csv(inaxe)
    df_openx = pd.read_csv(inopenx)
    df_axe.rename(columns={'axe_eval': 'eval_metric'}, inplace=True)
    df_openx.rename(columns={'openxai_eval': 'eval_metric'}, inplace=True)
    df_openx = df_openx[df_openx['ground_truth'] == 'legacy']
    df_axe = df_axe[df_axe['eval_metric'].isin(['1NN', '3NN', '5NN', '9NN'])]
    subset = ['eval_metric', 'dataset', 'model', 'shap', 'random', 'lime', 'vnla', 'sg', 'itg', 'ig']
    df_axe_subset = df_axe[subset]
    df_openx_subset = df_openx[subset]
    
    df_combined = pd.concat([df_axe_subset, df_openx_subset], ignore_index=True)
    output_file = os.path.join(outdir, 'combined_aucs.csv')
    df_combined.to_csv(outdir+'combined_data_unscaled.csv', index=False)
    

    # Define the columns to scale
    columns_to_scale = ['shap', 'random', 'lime', 'vnla', 'sg', 'itg', 'ig']
    # Scale each row for the specified columns using a new scaler per row
    
    def scale_row(row):
        scaler = StandardScaler()
        row[columns_to_scale] = scaler.fit_transform(row[columns_to_scale].values.reshape(-1, 1)).flatten()
        
        if row['eval_metric'] == 'PGU':
            row[columns_to_scale] = -row[columns_to_scale]
            row['eval_metric'] = '(-)PGU'
        return row
    df_combined = df_combined.apply(scale_row, axis=1)
    df_combined.to_csv(outdir+'combined_data_scaled.csv', index=False)
    
    # Define the custom sort order
    sort_order = ['1NN', '3NN', '5NN', '9NN', 'FA', 'SA', 'RA', 'SRA', 'RC', 'PRA', 'PGI', '(-)PGU']
    
    # Convert the 'eval_metric' column to a categorical type with the defined order
    df_combined['eval_metric'] = pd.Categorical(df_combined['eval_metric'], categories=sort_order, ordered=True)
    
    # Sort the dataframe by the 'eval_metric' column
    df_combined = df_combined.sort_values('eval_metric')

    df_combined['eval_metric'] = df_combined['eval_metric'].replace({
        '1NN': 'AXE_1',
        '3NN': 'AXE_3',
        '5NN': 'AXE_5',
        '9NN': 'AXE_9',
        # '1NN': '$AXE_1^{auc}$',
        # '3NN': '$AXE_3^{auc}$',
        # '5NN': '$AXE_5^{auc}$',
        # '9NN': '$AXE_9^{auc}$',
    })

    # os.makedirs(outdir, exist_ok=True)
    datasets = ['compas', 'adult', 'heloc', 'german']
    models = ['lr', 'ann']
    for model in models:
        for dataset in datasets:
            df_filtered = df_combined[(df_combined['model'] == model) & (df_combined['dataset'] == dataset)]
            df_filtered = df_filtered.drop(columns=['dataset', 'model'])
            plot_dataset_model(df_filtered, dataset, model, outdir+f'{model}_{dataset}_comparison.pdf')

    pass

def main():

    df_openxai = parse_openxai()
    df_axe = parse_axe()
    datasets = ['heloc', 'compas', 'adult', 'german']
    models = ['lr', 'ann']
    for model in models:
        for dataset in datasets:
            df_X = df_axe[(df_axe['dataset'] == dataset) & (df_axe['model'] == model)]
            df_Y = df_openxai[(df_openxai['dataset'] == dataset) & (df_openxai['model'] == model)]
    merge_and_plot()
    pass

if __name__ == '__main__':
    main()