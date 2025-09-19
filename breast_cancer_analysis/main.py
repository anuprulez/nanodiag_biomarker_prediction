import preprocess_data
import train_model
#import xai_explainer

base_path = "data_19_Sept_25/outputs/"

config = {
    "SEED": 32,
    "n_edges": 40000,
    "n_epo": 4,
    "k_folds": 5,
    "batch_size": 128,
    "num_classes": 5,
    "gene_dim": 86,
    "hidden_dim": 128,
    "learning_rate": 0.0001,
    "scale_features": "0,1,3",  #"degree,ring,NetShort",
    "out_links": f"{base_path}/out_links_bc.csv",
    "out_genes": f"{base_path}/out_genes_bc.csv",
    "out_gene_rankings": f"{base_path}/out_gene_rankings_bc.csv",
    "merged_signals": f"{base_path}/combined_pos_neg_signals_bc.csv",
    "nedbit_features": f"{base_path}/nedbit_features_bc.csv",
    "dnam_features": f"{base_path}/dnam_features_bc.csv",
    "nedbit_dnam_features": f"{base_path}/df_nebit_dnam_features_bc.csv",
    "nedbit_dnam_features_norm": "data/df_nebit_dnam_features_norm.csv",
    "plot_local_path": f"{base_path}",
    "data_local_path": f"{base_path}",
    "model_local_path": "model/"
}


def run_training():
    preprocess_data.read_files(config)
    train_model.train_gnn_model(config)
    #xai_explainer.gnn_explainer(trained_model, data, config)


if __name__ == "__main__":
    run_training()