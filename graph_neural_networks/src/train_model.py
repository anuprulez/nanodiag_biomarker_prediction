import torch
from torch.nn import Linear, BatchNorm1d, ReLU
import torch.nn.functional as F

import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

import gnn_network
import plot_gnn
import utils

model_activation = {}

# Define the hook function
#def hook_fn(module, input, output):
#    print(f"Inside {module.__class__.__name__} forward hook")
#    print(f"Input: {input}")
#    print(f"Output: {output}")
    
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
    '''
    Predict using trained model and test data
    '''
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
    pred_labels = pred[data.test_mask]
    true_labels = data.y[data.test_mask]
    test_correct = pred_labels == true_labels
    test_acc = int(test_correct.sum()) / float(int(data.test_mask.sum()))
    return test_acc, pred_labels.cpu().detach().numpy(), true_labels.cpu().detach().numpy(), pred


def extract_node_embeddings(model, data, model_activation, config):
    data_local_path = config["data_local_path"]
    conv_name = "PNAConv"
    activation_name = "BatchNorm1d"
    bn4_activation = model_activation[activation_name]
    conv4_activation = model_activation[conv_name]
    pred_embeddings_conv4 = conv4_activation[data.test_mask]
    pred_embeddings_batch_norm4 = bn4_activation[data.test_mask]
    true_labels = data.y[data.test_mask]
    pred_embeddings_conv4 = pred_embeddings_conv4.cpu().detach().numpy()
    pred_embeddings_batch_norm4 = pred_embeddings_batch_norm4.cpu().detach().numpy()
    true_labels = true_labels.cpu().detach().numpy()

    torch.save(pred_embeddings_conv4, data_local_path + 'embed_conv.pt')
    torch.save(pred_embeddings_batch_norm4, data_local_path + 'embed_batch_norm.pt')
    torch.save(true_labels, data_local_path + 'true_labels.pt')
    print(bn4_activation.shape, conv4_activation.shape, pred_embeddings_batch_norm4.shape)
    
    plot_gnn.plot_node_embed(pred_embeddings_conv4, true_labels, config, conv_name)
    plot_gnn.plot_node_embed(pred_embeddings_batch_norm4, true_labels, config, activation_name)

    print("----------------")


def save_model(model, config):
    model_local_path = config["model_local_path"]
    model_path = "{}/trained_model_edges_{}_epo_{}.ptm".format(model_local_path, config["n_edges"], config["n_epo"])
    torch.save(model.state_dict(), model_path)
    return model_path


def load_model(config, model_path, data):
    #model = gnn_network.GCN(config)
    model = gnn_network.GPNA(config, data)
    print(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    return model


def train_gnn_model(config):
    '''
    Create network architecture and assign loss, optimizers ...
    '''
    learning_rate = config["learning_rate"]
    k_folds = config["k_folds"]
    n_epo = config["n_epo"]
    batch_size = config["batch_size"]
    plot_local_path = config["plot_local_path"]
    data_local_path = config["data_local_path"]
    out_genes = pd.read_csv(config["out_genes"], sep=" ", header=None)
    mapped_f_name = out_genes.loc[:, 0]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    data = torch.load(config["data_local_path"] + 'data.pt')
    tr_nodes = pd.read_csv(data_local_path + "training_node_ids.csv", sep="\t")
    print(tr_nodes)
    tr_node_ids = tr_nodes["training_node_ids"].tolist()
    tr_node_ids = np.array(tr_node_ids)

    print("Initialize model")
    model = gnn_network.GPNA(config, data)
    print(torch.cuda.is_available())
    model = model.cuda()

    layer_pnaconv4 = model.pnaconv4
    layer_pnaconv4.register_forward_hook(hook_fn)

    layer_batch_norm4 = model.batch_norm4
    layer_batch_norm4.register_forward_hook(hook_fn)
    
    data = data.cuda()
    print(model)
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    st_kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)
    tr_loss_epo = list()
    te_acc_epo = list()
    val_acc_epo = list()

    # loop over epochs
    print("Start epoch training...")
    for epoch in range(n_epo):
        tr_loss_fold = list()
        val_acc_fold = list()
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        for fold, (train_index, val_index) in enumerate(kfold.split(tr_node_ids)):
            val_node_ids = tr_node_ids[val_index]
            train_nodes_ids = tr_node_ids[train_index]
            data.val_mask, _, _ = utils.create_test_masks(mapped_f_name, val_node_ids, out_genes)
            n_batches = int((len(train_index) + 1) / float(batch_size))
            batch_tr_loss = list()
            # loop over batches
            print("Start fold training for epoch: {}, fold: {}...".format(epoch+1, fold+1))
            for bat in range(n_batches):
                batch_tr_node_ids = train_nodes_ids[bat * batch_size: (bat+1) * batch_size]
                data.batch_train_mask = create_masks(mapped_f_name, batch_tr_node_ids)
                tr_loss = train(data, optimizer, model, criterion)
                batch_tr_loss.append(tr_loss.cpu().detach().numpy())
            tr_loss_fold.append(np.mean(batch_tr_loss))
            # predict using trained model
            val_acc = predict_data_val(model, data)
            print("Epoch {}/{}, fold {}/{} average training loss: {}".format(str(epoch+1), str(n_epo), str(fold+1), str(k_folds), str(np.mean(batch_tr_loss))))
            print("Epoch: {}/{}, Fold: {}/{}, val accuracy: {}".format(str(epoch+1), str(n_epo), str(fold+1), str(k_folds), str(val_acc)))
            val_acc_fold.append(val_acc)

        print("-------------------")
        te_acc, _, _, _ = predict_data_test(model, data)
        te_acc_epo.append(te_acc)
        tr_loss_epo.append(np.mean(tr_loss_fold))
        val_acc_epo.append(np.mean(val_acc_fold))
        print()
        print("Epoch {}: Training Loss: {}".format(str(epoch+1), str(np.mean(tr_loss_fold))))
        print("Epoch {}: Val accuracy: {}".format(str(epoch+1), str(np.mean(val_acc_fold))))
        print("Epoch {}: Test accuracy: {}".format(str(epoch+1), str(np.mean(te_acc))))
        print()
    #saved_model_path = save_model(model, config)
    print("==============")
    plot_gnn.plot_loss_acc(n_epo, tr_loss_epo, val_acc_epo, te_acc_epo, config)
    print("CV Training Loss after {} epochs: {}".format(str(n_epo), str(np.mean(tr_loss_epo))))
    print("CV Val acc after {} epochs: {}".format(str(n_epo), str(np.mean(val_acc_epo))))
    loaded_model = model #load_model(config, saved_model_path, data)
    final_test_acc, pred_labels, true_labels, all_pred = predict_data_test(loaded_model, data)
    torch.save(pred_labels, data_local_path + 'pred_labels.pt')
    torch.save(model, data_local_path + "model.pt")
    print("CV Test acc after {} epochs: {}".format(n_epo, final_test_acc))
    print("==============")
    extract_node_embeddings(model, data, model_activation, config)
    plot_gnn.plot_confusion_matrix(true_labels, pred_labels, config)
    plot_gnn.analyse_ground_truth_pos(loaded_model, data, out_genes, all_pred, config)
    
    #return model, data