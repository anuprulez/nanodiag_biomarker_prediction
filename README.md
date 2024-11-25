### Gene and probe (CpG context) prioritisation using Graph neural networks

![nanodiag_poster](https://github.com/user-attachments/assets/482e1633-c429-43f9-8068-0c0849d9005e)

Gene and probe prioritisation from the unknown sets using already available markers by propagating information through a gene-gene interaction network

### Environment

Install (via conda) the following:

```
conda env create -f running_env_16_08_24.yml
```

### Steps to run

1. Acquire datasets from:
  - Use DNA methylation arrays from: [In vivo kinetics of early, non-random methylome and transcriptome changes induced by DNA-hypomethylating treatment in primary AML blasts](https://www.nature.com/articles/s41375-023-01876-2)
  - Execute [preprocess HumanMethylation450_dataset](https://github.com/anuprulez/nanodiag_biomarker_prediction/blob/main/process_illumina_arrays/src/preprocess_HumanMethylation450_dataset.py) and [create gene-gene network](https://github.com/anuprulez/nanodiag_biomarker_prediction/blob/main/process_illumina_arrays/src/create_probe_gene_network.py)

2.  Propagate soft labels
  - Execute [soft label assignment](https://github.com/anuprulez/nanodiag_biomarker_prediction/blob/main/pu_label_propagation/src/assign_pre_labels.py)

3.  Graph neural network for representation refinement
  - Finally, execute [train graph neural network](https://github.com/anuprulez/nanodiag_biomarker_prediction/blob/main/graph_neural_networks/src/main.py)

### Results

UMAP for AML:

![umap_aml](https://github.com/user-attachments/assets/36da952e-a631-4403-95d1-69835a442595)

UMAP for Breast cancer:

![umap_bc](https://github.com/user-attachments/assets/72eeab68-1a3f-4ba1-b3c3-1cb3bb3f8563)

Novel gene/probe prioritisation

![aml_pr](https://github.com/user-attachments/assets/ceb4acef-1d5c-4fbb-aa7d-48772c5c01de)

![bc_pr](https://github.com/user-attachments/assets/f4ae60bf-f5b6-4dd0-9f3d-0386b7b219e2)



