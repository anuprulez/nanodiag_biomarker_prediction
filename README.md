### GraphMeXplain: Prioritising DNA methylation biomarkers using graph neural networks and explainable AI

CpG site prioritisation from the unknown sites using already available markers by propagating information through a CpG-CpG site interaction network

### Environment

Install (via conda) the following:

```
conda env create -f env.yml
```

### Steps to run

### Breast cancer

#### Using raw datasets 

- Set `p_raw_data` to `true` in `config.yaml` (All required datasets will be downloaded from [Zenodo](https://zenodo.org/records/17191564/files/bc_raw_datasets_to_preprocess.zip?download=1)
- Run `breast_cancer_analysis/preprocess_raw_data/process_raw_illumina_arrays_bc.py` to preprocess arrays and soft label assignment
- Run `breast_cancer_analysis/train_gnn/main.py` for training Graph neural network
- Run `breast_cancer_analysis/train_gnn/run_xai_explainer.py` for explainable AI subgraph analysis

#### Using preprocessed datasets 

- Set `p_raw_data` to `false` and `download_preprocessed_data` to `true` in `config.yaml`. Preprocessed datasets will be downloaded from [Zenodo](https://zenodo.org/records/17201552/files/bc_preprocessed_to_train_p_val_corrected.zip?download=1)
- Run `breast_cancer_analysis/train_gnn/main.py` for training Graph neural network
- Run `breast_cancer_analysis/train_gnn/run_xai_explainer.py` for explainable AI subgraph analysis


### AML

- Set `p_raw_data` to `true` in `config.yaml` (All required datasets will be downloaded from [Zenodo](https://zenodo.org/records/17202627/files/aml_raw_datasets.zip?download=1)
- Run `aml_analysis/preprocess_raw_data/preprocess_HumanMethylation450_dataset.py` to preprocess arrays and soft label assignment
- Run `aml_analysis/train_gnn/main.py` for training Graph neural network
- Run `aml_analysis/train_gnn/run_xai_explainer.py` for explainable AI subgraph analysis

#### Using preprocessed datasets (faster)

- Set `p_raw_data` to `false` and `download_preprocessed_data` to `true` in `config.yaml`.  Preprocessed datasets will be downloaded from [Zenodo](https://zenodo.org/records/17205435/files/aml_processed_datasets_c_0.5_te_10000.zip?download=1)
- Run `aml_analysis/train_gnn/main.py` for training Graph neural network
- Run `aml_analysis/train_gnn/run_xai_explainer.py` for explainable AI subgraph analysis


### Preprint

[Prioritizing DNA methylation biomarkers using graph neural networks and explainable AI](https://www.biorxiv.org/content/10.64898/2026.01.26.701692v1.abstract)



