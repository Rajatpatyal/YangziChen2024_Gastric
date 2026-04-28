# YangziChen2024 - Metabolic Machine Learning Predictor for Gastric Cancer

## 📌 Overview
This project demonstrates a **machine learning–based diagnostic system** for gastric cancer using **plasma metabolomics data**.

The model:
- Uses **10 key metabolites**
- Is exported in **ONNX format**
- Runs via a simplified Excel + Python inference pipeline

---

## ✨ Modifications by Rajat Patyal

This repository has been **extended and simplified for practical usage** by:

**Rajat Patyal**

Enhancements include:
- ✔ ONNX-based standalone inference pipeline
- ✔ Excel-driven input system (no coding required for users)
- ✔ Automated preprocessing and validation
- ✔ User-friendly prediction output (diagnosis + confidence)
- ✔ Cross-platform execution (Linux & Windows)
- ✔ File discovery commands (`find`, `dir`)
- ✔ Clean documentation for real-world usability

---

## 🧠 What is ONNX?

**ONNX (Open Neural Network Exchange)** is a format used to store machine learning models.

Benefits:
- No need for training code
- Lightweight deployment
- Works across platforms

👉 In this project:
```text
trained_model.onnx → takes 10 metabolite values → predicts gastric cancer risk

🧪 Model Input (10 Metabolites)

The model expects 10 metabolite values (float numbers) in this exact order:

Succinate
Uridine
Lactate
S-Adenosyl methionine (SAM)
Pyroglutamate
2-aminooctanoate
Neopterin
N-Acetyl-D-glucosamine 6-phosphate (GlcNAc6p)
Serotonin
Nicotinamide mononucleotide (NMN)

⚠️ Order must not change

📊 Input Format (Excel)

Edit:

Sample_model >> gastric_cancer_test_input.xlsx 

Structure:

Metabolite	Regulation	Value
Succinate	Downregulated	0.45
Uridine	Downregulated	0.32

👉 Only Value column is used

🚀 How to Run
1️⃣ Install dependencies
pip install pandas numpy onnxruntime openpyxl
2️⃣ Update Excel values

Modify:

gastric_cancer_test_input.xlsx
3️⃣ Run script
python App_onnx/run_onnx
4️⃣ Output
Diagnosis: Gastric Cancer

Confidence:
Non-GC: 34.00%
GC    : 66.00%

Saved file:

prediction_result.xlsx
🔄 Pipeline
Excel → ONNX model → Prediction → Output file
🔍 Find Files (Linux)
find ~/Documents -name "gastric_cancer_test_input.xlsx"
find ~/Documents -name "*.onnx"
find ~/Documents -name "run_onnx"
🔍 Find Files (Windows CMD)
dir \s \b gastric_cancer_test_input.xlsx
dir \s \b *.onnx
dir \s \b run_onnx*
🧪 What is this Blood Test?

This is based on:

Targeted Metabolomics (LC–MS)
Blood sample → plasma extraction
LC-MS → metabolite measurement
Model → prediction

⚠️ Not a routine clinical test

📊 Research Highlights
AUROC ≈ 0.967
Sensitivity ≈ 90.5%
Specificity ≈ 92.6%
Early-stage detection > 80%
Outperforms CA19-9, CEA

Validation includes:

ROC Curve
PCA
Confusion Matrix
🐳 Docker (Original)
docker build -t model .
docker run -it --rm model
⚠️ Disclaimer
THIS PROJECT IS FOR DEMONSTRATION PURPOSES ONLY
Not a medical device
Not clinically approved
No diagnostic decisions should be made

The contributors:

❌ Assume NO liability
❌ Provide NO medical advice
✔ Demonstrate ML research and metabolomics concepts
📚 Citation

Chen Y, Wang B, Zhao Y, et al.
Metabolomic machine learning predictor for diagnosis and prognosis of gastric cancer
Nature Communications (2024)
https://doi.org/10.1038/s41467-024-46043-y

👨‍💻 Contributors
Original Work

Akshat Pandey
GitHub: https://github.com/Akshat0285

Supervision:

IIT Madras
EMBL-EBI
Modifications & Implementation

Rajat Patyal

ONNX integration
Excel-based inference pipeline
Deployment simplification
Documentation enhancements




# YangziChen2024 - Metabolic Machine Learning Predictor Model for Diagnosis of Gastric Cancer

## Steps

[1] Unzip docker_train_predict folder
[2] Build the docker image using the command 'docker build -t model .'
[3] Use the command 'docker run -it --rm model' to reproduce the results
[4] For reproducing the plots and confusion matrices unzip the docker_plots folder
[5] Build the docker image using the command 'docker build -t plots .'
[6] Use the command 'docker run -it --rm plots' to reproduce and save the plots

## Details
**Author:** Akshat Pandey \
**Github:** https://github.com/Akshat0285 \
**E-mail:** akshatp21@iiserb.ac.in \
\
**UnderSupervision:**
- Prof. Karthik Raman, RBCDSAI, IIT Madras (kraman@iitm.ac.in)
- Prof. Rahuman S Malik Sheriff, EMBL-EBI

**Citations**
-Chen Y, Wang B, Zhao Y, Shao X, Wang M, Ma F, Yang L, Nie M, Jin P, Yao K, Song H, Lou S, Wang H, Yang T, Tian Y, Han P, Hu Z. Metabolomic machine learning predictor for diagnosis and prognosis of gastric cancer. Nat Commun. 2024 Feb 23;15(1):1657. doi: 10.1038/s41467-024-46043-y. PMID: 38395893; PMCID: PMC10891053.