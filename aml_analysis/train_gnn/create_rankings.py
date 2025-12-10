import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from omegaconf.omegaconf import OmegaConf


def rank_CpGs(config):

    base_path = config.p_base
    file_path = "pred_likely_pos_no_training_genes_probes_aml_PNA_20.csv"
    pred_cpgs_path = base_path + file_path

    pred_CpGs = pd.read_csv(pred_cpgs_path, sep="\t")
    print(pred_CpGs.head())

    naipu_ranked_CpGs = pd.read_csv(base_path + "out_gene_rankings_aml.csv", sep=" ", header=None)
    print(naipu_ranked_CpGs.head())

    CpG_names = list()
    updated_ranks = list()
    gnn_weight = 0.05
    for index, row in pred_CpGs.iterrows():
        cpg = row["test_gene_names"]
        pred_prob = row["pred_probs"]
        naipu_score = naipu_ranked_CpGs[naipu_ranked_CpGs.iloc[:, 0] == cpg].iloc[0, 1]
        new_rank = gnn_weight * pred_prob +  (1 - gnn_weight) * naipu_score
        CpG_names.append(cpg)
        updated_ranks.append(new_rank)
    
    ranked_CpGs_df = pd.DataFrame({"CpG": CpG_names, "Updated_Rank": updated_ranks})

    ranked_CpGs_df = ranked_CpGs_df.sort_values(by="Updated_Rank", ascending=False)

    ranked_CpGs_df.to_csv(base_path + "ranked_CpGs_gnn_niapu.csv", sep="\t", index=False)
    
    


if __name__ == "__main__":
    config = OmegaConf.load("../config/config.yaml")
    rank_CpGs(config)