import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, roc_curve, roc_auc_score, \
RocCurveDisplay, average_precision_score, confusion_matrix

import numpy as np
import pandas as pd


def plot_loss_acc(n_epo, tr_loss, val_acc, te_acc, config):
    plot_local_path = config["plot_local_path"]
    k_folds = config["k_folds"]
    n_edges = config["n_edges"]
    n_epo = config["n_epo"]
    
    plt.figure()
    x_val = np.arange(n_epo)
    plt.plot(x_val, tr_loss)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.grid(True)
    plt.title("Training loss")
    plt.savefig(plot_local_path + "{}_folds_CV_{}_links_{}_epochs_training_loss.pdf".format(k_folds, n_edges, n_epo), dpi=200)

    plt.figure()
    # plot accuracy on validation data
    x_val = np.arange(n_epo)
    plt.plot(x_val, val_acc)
    plt.plot(x_val, te_acc)
    plt.ylabel("Validation and Test accuracy")
    plt.xlabel("Epochs")
    plt.grid(True)
    plt.legend(["Validation", "Test"])
    plt.title("Validation and Test accuracy")
    plt.savefig(plot_local_path + "{}_folds_CV_{}_links_{}_epochs_validation_test_accuracy.pdf".format(k_folds, n_edges, n_epo), dpi=200)
    plt.show()


def plot_confusion_matrix(true_labels, predicted_labels, config, classes=[1, 2, 3, 4, 5]):
    plot_local_path = config["plot_local_path"]
    n_edges = config["n_edges"]
    n_epo = config["n_epo"]
    # Calculate confusion matrix
    true_labels = [int(item) + 1 for item in true_labels]
    predicted_labels = [int(item) + 1 for item in predicted_labels]
    cm = confusion_matrix(true_labels, predicted_labels)
    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  # Adjust font scale for better readability
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    
    # Add labels and title
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Show plot
    plt.tight_layout()
    #plt.show()
    plt.grid(True)
    plt.savefig(plot_local_path + "Confusion_matrix_NPPI_{}_NEpochs_{}.pdf".format(n_edges, n_epo), dpi=200)


def analyse_ground_truth_pos(model, compact_data, out_genes, all_pred, config):
    plot_local_path = config["plot_local_path"]
    n_edges = config["n_edges"]
    n_epo = config["n_epo"]
    ground_truth_pos_genes = out_genes[out_genes.iloc[:, 2] > 0]
    ground_truth_pos_gene_ids = ground_truth_pos_genes.iloc[:, 0].tolist()
    test_index = [index for index, item in enumerate(compact_data.test_mask) if item == True]
    print()
    masked_pos_genes_ids = list(set(ground_truth_pos_gene_ids).intersection(set(test_index)))
    print(len(ground_truth_pos_gene_ids), len(test_index), len(masked_pos_genes_ids))
    model.eval()
    out = model(compact_data.x, compact_data.edge_index)
    all_pred = out.argmax(dim=1)
    masked_p_pos_labels = all_pred[masked_pos_genes_ids]
    masked_p_pos_labels = masked_p_pos_labels.cpu().detach()
    df_p_labels = pd.DataFrame(masked_p_pos_labels, columns=["pred_labels"])
    # plot histogram
    plt.figure(figsize=(8, 6))
    g = sns.histplot(data=df_p_labels, x="pred_labels")
    plt.xlabel('Predicted classes')
    plt.ylabel('Count')
    plt.title('Masked positive genes predicted into different classes.')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    # set the ticks first 
    g.set_xticks(range(5))
    # set the labels 
    g.set_xticklabels(['1', '2', '3', '4', '5']) 
    # Show plot
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(plot_local_path + "Histogram_positive__NPPI_{}_NEpochs_{}.pdf".format(n_edges, n_epo), dpi=200)

    plt.figure(figsize=(8, 6))
    g = sns.kdeplot(data=df_p_labels, x="pred_labels")
    plt.xlabel('Predicted classes')
    plt.ylabel('Density')
    plt.title('Masked positive genes predicted into different classes.')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    # set the ticks first 
    g.set_xticks(range(5))
    # set the labels 
    g.set_xticklabels(['1', '2', '3', '4', '5']) 
    # Show plot
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(plot_local_path + "KDE_positive__NPPI_{}_NEpochs_{}.pdf".format(n_edges, n_epo), dpi=200)


def plot_features(features, labels, config):
    plot_local_path = config["plot_local_path"]
    n_neighbors=10 #10 #5
    min_dist=0.99 #0.99 #0.3
    metric='correlation'
    labels = [int(item) for item in labels]
    embeddings = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='correlation').fit_transform(features)
    data = {"UMAP1": embeddings[:, 0], "UMAP2": embeddings[:, 1], "Label": labels}
    df = pd.DataFrame(data)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="UMAP1", y="UMAP2", hue="Label", data=df, palette="viridis", s=50, alpha=1.0)
    plt.title("UMAP Visualization of NedBit + DNA Methylation features")
    plt.savefig(plot_local_path + "umap_nedbit_dnam_features.pdf")


def plot_node_embed(features, labels, config, feature_type):
    plot_local_path = config["plot_local_path"]
    n_neighbors=20 #10 #5
    min_dist=0.8 #0.99 #0.3
    metric='correlation'
    labels = [int(item) + 1 for item in labels]
    embeddings = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='correlation').fit_transform(features)
    print("Embeddings shape: ", embeddings.shape)
    data = {"UMAP1": embeddings[:, 0], "UMAP2": embeddings[:, 1], "Label": labels}
    df = pd.DataFrame(data)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="UMAP1", y="UMAP2", hue="Label", data=df, palette="viridis", s=50, alpha=1.0)
    plt.title("UMAP Visualization of node embeddings from last {} layer".format(feature_type))
    plt.savefig(plot_local_path + "umap_node_embeddings_{}_{}_{}.pdf".format(n_neighbors, min_dist, feature_type))
