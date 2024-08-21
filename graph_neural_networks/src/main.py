import preprocess_data
import train_model
import xai_explainer


config = {
    "SEED": 32,
    "n_edges": 2000000,
    "n_epo": 4,
    "k_folds": 5,
    "batch_size": 128,
    "num_classes": 5,
    "gene_dim": 40,
    "hidden_dim": 128,
    "learning_rate": 0.0001,
    "scale_features": "degree,ring,NetShort",
    "out_links": "../../pu_label_propagation/data/output/out_links.csv",
    "out_genes": "../../pu_label_propagation/data/output/out_genes.csv",
    "out_gene_rankings": "../../pu_label_propagation/data/output/out_gene_rankings.csv",
    "merged_signals": "../../process_illumina_arrays/data/output/merged_signals.csv",
    "nedbit_features": "../../pu_label_propagation/data/output/nedbit_features.csv",
    "dnam_features": "../../pu_label_propagation/data/output/dnam_features.csv",
    "nedbit_dnam_features": "../data/output/df_nebit_dnam_features.csv",
    "plot_local_path": "../data/output/",
    "data_local_path": "../data/output/",
    "model_local_path": "../model/"
}


def run_training():
    compact_data, feature_n, mapped_f_name, out_genes = preprocess_data.read_files(config)
    trained_model, data = train_model.create_training_proc(compact_data, feature_n, mapped_f_name, out_genes, config)
    xai_explainer.gnn_explainer(trained_model, data, config)


if __name__ == "__main__":
    run_training()
