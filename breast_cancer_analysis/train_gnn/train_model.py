import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import gnn_network
import plot_gnn
import utils

detach = utils.detach_from_gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_activation = {}
    
# Define the hook function
def hook_fn(module, input, output):
    model_activation[module.__class__.__name__] = output.detach()


def create_masks(mapped_node_ids, mask_list):
    mask = mapped_node_ids.index.isin(mask_list)
    return torch.tensor(mask, dtype=torch.bool)


def train(data, optimizer, model, criterion):
    # Clear gradients
    optimizer.zero_grad()
    # forward pass
    out = model(data.x, data.edge_index)
    # compute error using training mask
    loss = criterion(out[data.batch_train_mask], data.y[data.batch_train_mask])
    # compute gradients
    loss.backward()
    # optimize weights
    optimizer.step()
    return loss


def predict_data_val(model, data):
    """
    Predict using trained model and test data
    """
    # predict on test fold
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    val_correct = pred[data.val_mask] == data.y[data.val_mask]
    val_acc = int(val_correct.sum()) / float(int(data.val_mask.sum()))
    return val_acc


def predict_data_test(model, data):
    model.eval()
    model = model.cuda()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    probs = F.softmax(out, dim=1)
    te_probs = detach(probs[data.test_mask])
    pred_max_probs = probs[data.test_mask].max(dim=1).values
    pred_labels = pred[data.test_mask]
    true_labels = data.y[data.test_mask]
    test_correct = pred_labels == true_labels
    test_acc = int(test_correct.sum()) / float(int(data.test_mask.sum()))
    return test_acc, detach(pred_labels), detach(true_labels), pred, te_probs, detach(pred_max_probs)


def extract_node_embeddings(model, data, model_activation, config):
    data_local_path = config.p_data
    conv_name = "PNAConv"
    activation_name = "BatchNorm1d"
    bn4_activation = model_activation[activation_name]
    conv4_activation = model_activation[conv_name]
    pred_embeddings_conv4 = conv4_activation[data.test_mask]
    pred_embeddings_batch_norm4 = bn4_activation[data.test_mask]
    true_labels = data.y[data.test_mask]
    pred_embeddings_conv4 = detach(pred_embeddings_conv4)
    pred_embeddings_batch_norm4 = detach(pred_embeddings_batch_norm4)
    true_labels = detach(true_labels)
    torch.save(pred_embeddings_conv4, config.p_torch_embed)
    torch.save(pred_embeddings_batch_norm4, config.p_torch_embed_batch_norm)
    torch.save(true_labels, config.p_true_labels)
    print("Plot UMAP embeddings")    
    plot_gnn.plot_node_embed(pred_embeddings_conv4, true_labels, config, conv_name)
    plot_gnn.plot_node_embed(pred_embeddings_batch_norm4, true_labels, config, activation_name)


def save_model(model, config):
    model_local_path = config.p_model
    model_path = "{}/trained_model_edges_{}_epo_{}.ptm".format(model_local_path, config.n_edges, config.n_epo)
    torch.save(model.state_dict(), model_path)
    return model_path


def load_model(config, model_path, data):
    model = gnn_network.GPNA(config, data)
    print(model)
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    return model


def train_gnn_model(config):
    """
    Create network architecture and assign loss, optimizers ...
    """
    learning_rate = config.learning_rate
    k_folds = config.k_folds
    n_epo = config.n_epo
    batch_size = config.batch_size
    data_local_path = config.p_data
    out_genes = pd.read_csv(config.p_out_genes, sep=" ", header=None)
    mapped_f_name = out_genes.loc[:, 0]
    
    
    print(f"Used device: {device}")

    data = torch.load(config.p_torch_data, weights_only=False)
    tr_nodes = pd.read_csv(config.p_train_probe_genes, sep=",")
    tr_node_ids = tr_nodes["tr_gene_ids"].tolist()
    tr_node_ids = np.array(tr_node_ids)

    print("Initialize model")
    model = gnn_network.GPNA(config, data)
    model = model.cuda()

    layer_pnaconv4 = model.pnaconv4
    layer_pnaconv4.register_forward_hook(hook_fn)

    layer_batch_norm4 = model.batch_norm4
    layer_batch_norm4.register_forward_hook(hook_fn)
    
    data = data.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #st_kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)
    tr_loss_epo = list()
    te_acc_epo = list()
    val_acc_epo = list()
    
    # loop over epochs
    print("Start epoch training...")
    for epoch in range(n_epo):
        tr_loss_fold = list()
        val_acc_fold = list()
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        #skfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        for fold, (train_index, val_index) in enumerate(kfold.split(tr_node_ids)):
            val_node_ids = tr_node_ids[val_index]
            train_nodes_ids = tr_node_ids[train_index]
            print(f"Epoch {epoch+1}, Fold {fold+1}: train nodes: {len(train_nodes_ids)}, val nodes: {len(val_node_ids)}")
            data.val_mask, _, _ = utils.create_test_masks(mapped_f_name, val_node_ids, out_genes)
            n_batches = int((len(train_index) + 1) / float(batch_size))
            batch_tr_loss = list()
            # loop over batches
            print("Start fold training for epoch: {}, fold: {}...".format(epoch+1, fold+1))
            for bat in range(n_batches):
                batch_tr_node_ids = train_nodes_ids[bat * batch_size: (bat+1) * batch_size]
                data.batch_train_mask = create_masks(mapped_f_name, batch_tr_node_ids)
                tr_loss = train(data, optimizer, model, criterion)
                tr_loss = detach(tr_loss)
                batch_tr_loss.append(np.round(np.mean(tr_loss), 2))
            tr_loss_fold.append(np.round(np.mean(batch_tr_loss), 2))
            # predict using trained model
            val_acc = predict_data_val(model, data)
            print("Epoch {}/{}, fold {}/{} average training loss: {}".format(epoch+1, n_epo, fold+1, k_folds, np.round(np.mean(batch_tr_loss), 2)))
            print("Epoch: {}/{}, Fold: {}/{}, val accuracy: {}".format(epoch+1, n_epo, fold+1, k_folds, np.round(val_acc), 2))
            val_acc_fold.append(val_acc)

        print("-------------------")
        te_acc, *_ = predict_data_test(model, data)
        te_acc_epo.append(te_acc)
        tr_loss_epo.append(np.round(np.mean(tr_loss_fold), 2))
        val_acc_epo.append(np.round(np.mean(val_acc_fold), 2))
        print()
        print("Epoch {}: Training Loss: {}".format(epoch+1, np.round(np.mean(tr_loss_fold), 2)))
        print("Epoch {}: Val accuracy: {}".format(epoch+1, np.round(np.mean(val_acc_fold), 2)))
        print("Epoch {}: Test accuracy: {}".format(epoch+1, np.round(np.mean(te_acc), 2)))
        print()
    print("==============")
    plot_gnn.plot_loss_acc(n_epo, tr_loss_epo, val_acc_epo, te_acc_epo, config)
    print("CV Training Loss after {} epochs: {}".format(n_epo, np.round(np.mean(tr_loss_epo), 2)))
    print("CV Val acc after {} epochs: {}".format(n_epo, np.round(np.mean(val_acc_epo), 2)))
    final_test_acc, pred_labels, true_labels, all_pred, all_probs, all_pred_prob = predict_data_test(model, data)
    torch.save(pred_labels, config.p_pred_labels)
    torch.save(all_pred_prob, config.p_pred_probs)
    torch.save(model, config.p_torch_model)
    _ = save_model(model, config)
    print("CV Test acc after {} epochs: {}".format(n_epo, np.round(final_test_acc, 2)))
    extract_node_embeddings(model, data, model_activation, config)
    plot_gnn.plot_confusion_matrix(true_labels, pred_labels, config)
    plot_gnn.plot_precision_recall(true_labels, all_probs, all_pred_prob, config)