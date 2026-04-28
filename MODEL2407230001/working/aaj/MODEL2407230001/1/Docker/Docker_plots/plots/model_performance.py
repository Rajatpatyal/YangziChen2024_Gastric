import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import os
import gc

# Get the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths to the data files and figures directory
test_data_path = os.path.join(current_dir, 'data', 'pred_score_for_test.xlsx')
cohort_info_path = os.path.join(current_dir, 'data', 'cohort_information.xlsx')
external_test_data_path = os.path.join(current_dir, 'data', 'pred_score_for_external_test.xlsx')
figures_dir = os.path.join(current_dir, 'figures', 'Model_performance')

# Create the figures directory if it doesn't exist
os.makedirs(figures_dir, exist_ok=True)

########## model performance for test dataset ##########
test = pd.read_excel(test_data_path, sheet_name=0, index_col=0)
cohort_information = pd.read_excel(cohort_info_path, sheet_name=0)
cohort_information = cohort_information[['sample_id', 'stage']]

# Remove duplicate indexes if present
test = test[~test.index.duplicated(keep='first')]
cohort_information = cohort_information[~cohort_information.index.duplicated(keep='first')]

patients = test.merge(cohort_information, on="sample_id")
normal = test[test['state'] == 0].copy()
normal['stage'] = "N"
test = pd.concat([patients, normal]).reset_index(drop=True)

# PCA analysis
df_PCA = test.iloc[:, :147]
pca = PCA(n_components=2)
PCA_results = pca.fit_transform(df_PCA)
summ = pca.explained_variance_ratio_
df1 = pd.DataFrame(PCA_results, columns=['PC1', 'PC2'])
df1 = df1.reset_index(drop=True)  # Ensure df1 has a clean index
df1['state'] = test['state'].astype(int).astype('category').reset_index(drop=True)
df1['sample_id'] = test['sample_id'].reset_index(drop=True)

xlab = f"PC1({summ[0]*100:.2f}%)"
ylab = f"PC2({summ[1]*100:.2f}%)"

plt.figure(figsize=(6.31, 3.96))
sns.scatterplot(data=df1, x='PC1', y='PC2', hue='state', palette='viridis')
for state in df1['state'].unique():
    subset = df1[df1['state'] == state]
    ellipse = Ellipse(xy=(subset['PC1'].mean(), subset['PC2'].mean()), 
                      width=subset['PC1'].std()*2, height=subset['PC2'].std()*2, 
                      alpha=0.2)
    plt.gca().add_patch(ellipse)
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.title('PCA Analysis')
plt.legend(title='')
plt.grid(False)
plt.savefig(os.path.join(figures_dir, 'PCA_Test.pdf'))
plt.close()

df2 = df1[['PC1', 'PC2', 'state', 'sample_id']]
df3 = test[['sample_id', 'pred_score', 'stage']]
df4 = df2.merge(df3, on='sample_id')

model_performance_prediction = df4[['pred_score', 'PC1', 'state', 'sample_id', 'stage']]
model_performance_prediction.loc[:,'state'] = model_performance_prediction['state'].astype(int)
model_performance_prediction['clinical_state'] = np.where(model_performance_prediction['state'] == 1, "GC", "Healthy")
model_performance_prediction = model_performance_prediction.drop(columns=['state'])

ylab = f"PC1({summ[0]*100:.2f}%)"
xlab = "predicted value for GC"

plt.figure(figsize=(9, 6))
sns.scatterplot(data=model_performance_prediction, x='pred_score', y='PC1', hue='clinical_state', s=100, palette='viridis')
plt.axvline(x=0.5, linestyle='dotted')
plt.xlabel(xlab, fontsize=16)
plt.ylabel(ylab, fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(title='', fontsize=16)
plt.grid(False)
plt.savefig(os.path.join(figures_dir, 'Model_performance_prediction_test_all_stage.pdf'))
plt.close()

########## model performance for external test dataset ##########
gc.collect()
external_test = pd.read_excel(external_test_data_path, sheet_name=0, index_col=0)
cohort_information = pd.read_excel(cohort_info_path, sheet_name=0)
cohort_information = cohort_information[['sample_id', 'stage']]

# Remove duplicate indexes if present
external_test = external_test[~external_test.index.duplicated(keep='first')]
cohort_information = cohort_information[~cohort_information.index.duplicated(keep='first')]

patients = external_test.merge(cohort_information, on="sample_id")
normal = external_test[external_test['state'] == 0].copy()
normal['stage'] = "N"
external_test = pd.concat([patients, normal]).reset_index(drop=True)

# PCA analysis
df_PCA = external_test.iloc[:, :147]
pca = PCA(n_components=2)
PCA_results = pca.fit_transform(df_PCA)
summ = pca.explained_variance_ratio_
df1 = pd.DataFrame(PCA_results, columns=['PC1', 'PC2'])
df1 = df1.reset_index(drop=True)  # Ensure df1 has a clean index
df1['state'] = external_test['state'].astype(int).astype('category').reset_index(drop=True)
df1['sample_id'] = external_test['sample_id'].reset_index(drop=True)

xlab = f"PC1({summ[0]*100:.2f}%)"
ylab = f"PC2({summ[1]*100:.2f}%)"

plt.figure(figsize=(6.31, 3.96))
sns.scatterplot(data=df1, x='PC1', y='PC2', hue='state', palette='viridis')
for state in df1['state'].unique():
    subset = df1[df1['state'] == state]
    ellipse = Ellipse(xy=(subset['PC1'].mean(), subset['PC2'].mean()), 
                      width=subset['PC1'].std()*2, height=subset['PC2'].std()*2, 
                      alpha=0.2)
    plt.gca().add_patch(ellipse)
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.title('PCA Analysis')
plt.legend(title='')
plt.grid(False)
plt.savefig(os.path.join(figures_dir, 'PCA_External_test.pdf'))
plt.close()

df2 = df1[['PC1', 'PC2', 'state', 'sample_id']]
df3 = external_test[['sample_id', 'pred_score', 'stage']]
df4 = df2.merge(df3, on='sample_id')

model_performance_prediction = df4[['pred_score', 'PC1', 'state', 'sample_id', 'stage']]
model_performance_prediction.loc[:,'state'] = model_performance_prediction['state'].astype(int)
model_performance_prediction['clinical_state'] = np.where(model_performance_prediction['state'] == 1, "GC", "Healthy")
model_performance_prediction = model_performance_prediction.drop(columns=['state'])

ylab = f"PC1({summ[0]*100:.2f}%)"
xlab = "predicted value for GC"

plt.figure(figsize=(9, 6))
sns.scatterplot(data=model_performance_prediction, x='pred_score', y='PC1', hue='clinical_state', s=100, palette='viridis')
plt.axvline(x=0.5, linestyle='dotted')
plt.xlabel(xlab, fontsize=16)
plt.ylabel(ylab, fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(title='', fontsize=16)
plt.grid(False)
plt.savefig(os.path.join(figures_dir, 'Model_performance_prediction_external_test_all_stage.pdf'))
plt.close()

print('Done')