import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths to the data files
biomarker_clinical_path = os.path.join(current_dir, 'data', 'Biomarker_clinical.xlsx')
figures_dir = os.path.join(current_dir, 'figures', 'Clinical_Model_comparison')

# Create the figures directory if it doesn't exist
os.makedirs(figures_dir, exist_ok=True)

# Load the data
biomarker_clinical = pd.read_excel(biomarker_clinical_path, sheet_name="Sheet1")
predict_risk_score = pd.read_excel(biomarker_clinical_path, sheet_name="Sheet2")

# Merge datasets
biomarker_clinical = pd.merge(biomarker_clinical, predict_risk_score, on="orinumber")

# Select and rename columns
biomarker_clinical = biomarker_clinical[['orinumber', 'CA199（0-27）', 'CA724（0-6.9）', 'CEA,（0-5）', 'cohort', 'pred_score', 'state']]
biomarker_clinical.columns = ['orinumber', 'CA199', 'CA724', 'CEA', 'cohort', 'pred_score', 'state']

# Convert to numeric
biomarker_clinical['CA199'] = pd.to_numeric(biomarker_clinical['CA199'], errors='coerce')
biomarker_clinical['CA724'] = pd.to_numeric(biomarker_clinical['CA724'], errors='coerce')
biomarker_clinical['CEA'] = pd.to_numeric(biomarker_clinical['CEA'], errors='coerce')

# Filter data
CA199 = biomarker_clinical[['orinumber', 'CA199', 'pred_score', 'state']].dropna(subset=['CA199'])
CA724 = biomarker_clinical[['orinumber', 'CA724', 'pred_score', 'state']].dropna(subset=['CA724'])
CEA = biomarker_clinical[['orinumber', 'CEA', 'pred_score', 'state']].dropna(subset=['CEA'])

# Plot CA199
plt.figure(figsize=(9, 8))
sns.scatterplot(data=CA199, x=np.log2(CA199['CA199']), y=CA199['pred_score'], hue=CA199['state'], palette='tab10', s=100, alpha=0.8)
plt.axvline(x=np.log2(27), linestyle='--', color='blue')
plt.axhline(y=0.5, linestyle='--', color='red')
plt.title("Comparison of Prediction Accuracy Between CA199 and RF model", fontsize=15, fontweight='bold', loc='center')
plt.xlabel("log2(CA199)", fontsize=16, fontweight='bold')
plt.ylabel("Metabolite Biomarkers Risk Score", fontsize=16, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.legend(title='', fontsize=16)
plt.grid(False)
plt.tight_layout(pad=4.0) 
plt.savefig(os.path.join(figures_dir, "Comparison_of_Prediction_Accuracy_Between_CA199_and_RF_Model.pdf"))
plt.close()

# Plot CA724
plt.figure(figsize=(9, 8))
sns.scatterplot(data=CA724, x=np.log2(CA724['CA724']), y=CA724['pred_score'], hue=CA724['state'], palette='tab10', s=100, alpha=0.8)
plt.axvline(x=np.log2(6.9), linestyle='--', color='blue')
plt.axhline(y=0.5, linestyle='--', color='red')
plt.title("Comparison of Prediction Accuracy Between CA724 and RF model", fontsize= 15, fontweight='bold', loc='center')
plt.xlabel("log2(CA724)", fontsize=16, fontweight='bold')
plt.ylabel("Metabolite Biomarkers Risk Score", fontsize=16, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.legend(title='', fontsize=16)
plt.grid(False)
plt.tight_layout(pad=4.0) 
plt.savefig(os.path.join(figures_dir, "Comparison_of_Prediction_Accuracy_Between_CA724_and_RF_Model.pdf"))
plt.close()

# Plot CEA
plt.figure(figsize=(9, 8))
sns.scatterplot(data=CEA, x=np.log2(CEA['CEA']), y=CEA['pred_score'], hue=CEA['state'], palette='tab10', s=100, alpha=0.8)
plt.axvline(x=np.log2(5), linestyle='--', color='blue')
plt.axhline(y=0.5, linestyle='--', color='red')
plt.title("Comparison of Prediction Accuracy Between CEA and RF model", fontsize=15, fontweight='bold', loc='center')
plt.xlabel("log2(CEA)", fontsize=16, fontweight='bold')
plt.ylabel("Metabolite Biomarkers Risk Score", fontsize=16, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.legend(title='', fontsize=16)
plt.grid(False)
plt.tight_layout(pad=4.0) 
plt.savefig(os.path.join(figures_dir, "Comparison_of_Prediction_Accuracy_Between_CEA_and_RF_Model.pdf"))
plt.close()

print('Done')