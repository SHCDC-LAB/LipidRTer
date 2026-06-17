# 🧪 LipidRTer

A transferable machine-learning framework for lipid retention time prediction and RT-assisted lipid annotation in LC–MS lipidomics.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![GUI](https://img.shields.io/badge/GUI-Streamlit-red)

---

## Overview

**LipidRTer** is a Streamlit-based platform for lipid retention time (RT) prediction and MSP library RT updating.

It supports:

- lipid RT prediction from SMILES structures
- transfer learning using small RT calibration datasets
- model fine-tuning and prediction-only workflows
- model evaluation and interpretation
- MSP library preparation and RT updating

---

## Key features

- Predict lipid RT from SMILES
- Fine-tune pretrained models for new chromatographic systems
- Generate PaDEL molecular descriptors
- Use precomputed descriptor files
- Evaluate models using R², Q², RMSE, MAE, MedAE, MRE, and MedRE
- Perform permutation testing
- Generate Williams plots and applicability domain analysis
- Run SHAP-based feature interpretation
- Update predicted RT values into MSP libraries
- Provide an end-to-end Streamlit dashboard

---

## Installation

```bash
git clone https://github.com/your-username/LipidRTer.git
cd LipidRTer

conda create -n lipidrter python=3.10
conda activate lipidrter

pip install -r requirements.txt
```

## Quick start

```bash
streamlit run app.py
```

## Citation

LipidRTer: A Transferable Retention Time Prediction Framework for Improved Lipid Annotation.

## License

MIT License
