import preprocess_data
import train_model
#import xai_explainer


config = {
    "SEED": 32,
    "n_edges": 1576515,
    "n_epo": 4,
    "k_folds": 5,
    "batch_size": 128,
    "num_classes": 5,
    "gene_dim": 86,
    "hidden_dim": 128,
    "learning_rate": 0.0001,
    "scale_features": "0,1,3",  #"degree,ring,NetShort",
    "out_links": "data/out_links_bc.csv",
    "out_genes": "data/out_genes_bc.csv",
    "out_gene_rankings": "data/out_gene_rankings_bc.csv",
    "merged_signals": "data/combined_pos_neg_signals_bc.csv",
    "nedbit_features": "data/nedbit_features_bc.csv",
    "dnam_features": "data/dnam_features_bc.csv",
    "nedbit_dnam_features": "data/df_nebit_dnam_features_bc.csv",
    "nedbit_dnam_features_norm": "data/df_nebit_dnam_features_norm.csv",
    "plot_local_path": "data/",
    "data_local_path": "data/",
    "model_local_path": "model/"
}


def run_training():
    preprocess_data.read_files(config)
    train_model.train_gnn_model(config)
    #xai_explainer.gnn_explainer(trained_model, data, config)


if __name__ == "__main__":
    run_training()