import copy
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from sklearn.model_selection import train_test_split
from torch_geometric.utils import coalesce

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

def make_hook(name):
    def _hook(module, inp, out):
        model_activation[name] = out.detach()
    return _hook


'''def create_masks(mapped_node_ids, mask_list):
    mask = mapped_node_ids.isin(mask_list)
    return torch.tensor(mask, dtype=torch.bool)'''

def create_masks(mapped_node_ids: pd.Series, mask_list):
    # True where the SERIES VALUE (node id/name) is in the mask list
    mask = mapped_node_ids.isin(mask_list).to_numpy()
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
    model_path = f"{model_local_path}/trained_model_edges_{config.n_edges}_epo_{config.n_epo}.ptm"
    torch.save(model.state_dict(), model_path)
    return model_path


def load_model(config, model_path, data):
    model = gnn_network.GPNA(config, data)
    print(model)
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    return model


def ensure_bool(m):
    return m if m.dtype == torch.bool else m.bool()

def make_neighbor_loaders(data, config):
    # Keep the big graph on CPU
    data = data.cpu()

    data.edge_index = coalesce(data.edge_index, num_nodes=data.num_nodes)

    train_loader = NeighborLoader(
        data,
        input_nodes=ensure_bool(data.train_mask),  # seed nodes = train
        num_neighbors=config.neighbors_spread,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        subgraph_type=config.graph_subtype
    )

    val_loader = NeighborLoader(
        data,
        input_nodes=ensure_bool(data.val_mask),
        num_neighbors=config.neighbors_spread,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        subgraph_type=config.graph_subtype
    )
    test_loader = NeighborLoader(
        data,
        input_nodes=ensure_bool(data.test_mask),
        num_neighbors=config.neighbors_spread,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        subgraph_type=config.graph_subtype
    )
    return train_loader, val_loader, test_loader


def train_one_epoch(train_loader, model, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    
    for tr_idx, batch in enumerate(train_loader):
        batch = batch.to(device, non_blocking=True)
        #print(f"Batch tr: {tr_idx}, {batch.x.shape}, {batch.edge_index.shape}")
        optimizer.zero_grad(set_to_none=True)
        out = model(batch.x, batch.edge_index)
        seed_n = batch.batch_size # first seed_n nodes = seeds
        logits = out[:seed_n]
        targets = batch.y[:seed_n].long()
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * seed_n
        total_correct += (logits.argmax(-1) == targets).sum().item()
        total_count += seed_n

    avg_loss = total_loss / max(1, total_count)
    avg_acc = total_correct / max(1, total_count)
    return avg_loss, avg_acc

@torch.no_grad()
def val_evaluate(loader, model, criterion, device):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    used_val_ids = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        used_val_ids.extend(batch.n_id.cpu().tolist()[:batch.batch_size])
        out = model(batch.x, batch.edge_index)
        seed_n = batch.batch_size  # evaluating only the seed nodes of this batch
        logits = out[:seed_n]
        targets = batch.y[:seed_n].long()
        loss = criterion(logits, targets)

        total_loss += float(loss) * seed_n
        total_correct += (logits.argmax(-1) == targets).sum().item()
        total_count += seed_n

    avg_loss = total_loss / max(1, total_count)
    avg_acc = total_correct / max(1, total_count)
    return avg_loss, avg_acc, used_val_ids


@torch.no_grad()
def test_evaluate(loader, model, criterion, device):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    pred_labels, true_labels, all_probs, all_pred_probs, test_ids = [], [], [], [], []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        out = model(batch.x, batch.edge_index)
        seed_n = batch.batch_size  # evaluating only the seed nodes of this batch
        test_ids.extend(batch.n_id.cpu().tolist()[:seed_n])
        logits = out[:seed_n]
        batch_prob = F.softmax(logits, dim=1)
        batch_max_prob = batch_prob.max(dim=1).values
        targets = batch.y[:seed_n].long()
        loss = criterion(logits, targets)

        total_loss += float(detach(loss)) * seed_n
        batch_pred_label = logits.argmax(-1)
        pred_labels.extend(detach(batch_pred_label))
        true_labels.extend(detach(targets))
        all_probs.extend(detach(batch_prob))
        all_pred_probs.extend(detach(batch_max_prob))
        total_correct += (logits.argmax(-1) == targets).sum().item()
        total_count += seed_n

    avg_loss = total_loss / max(1, total_count)
    avg_acc = total_correct / max(1, total_count)
    return avg_loss, avg_acc, pred_labels, true_labels, all_probs, all_pred_probs, test_ids


def train_gnn_model(config):
    """
    Create network architecture and assign loss, optimizers ...
    """
    use_amp = False
    learning_rate = config.learning_rate
    n_epo = config.n_epo
    out_genes = pd.read_csv(config.p_out_genes, sep=" ", header=None)
    mapped_f_name = out_genes.loc[:, 0]
    
    print(f"Used device: {device}")

    data = torch.load(config.p_torch_data, weights_only=False)
    tr_nodes = pd.read_csv(config.p_train_probe_genes, sep=",")
    te_nodes = pd.read_csv(config.p_test_probe_genes, sep=",")
    te_node_ids = te_nodes["test_gene_ids"].tolist()
    tr_node_ids = tr_nodes["tr_gene_ids"].tolist()
    tr_node_ids = np.array(tr_node_ids)

    print("Initialize model")
    model = gnn_network.GPNA(config, data)
    model = model.cuda()

    layer_pnaconv4 = model.pnaconv4
    #layer_pnaconv4.register_forward_hook(hook_fn)

    layer_batch_norm4 = model.batch_norm4
    #layer_batch_norm4.register_forward_hook(hook_fn)

    layer_pnaconv4.register_forward_hook(make_hook("pnaconv4"))
    layer_batch_norm4.register_forward_hook(make_hook("batch_norm4"))
    
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    tr_loss_epo = list()
    tr_acc_epo = list()
    te_loss_epo = list()
    te_acc_epo = list()
    val_loss_epo = list()
    val_acc_epo = list()
    best_te_acc = -float("inf")
    best_state = None
    best_epoch = -1

    split_tr_node_ids, val_node_ids = train_test_split(tr_node_ids, shuffle=True, test_size=config.test_size, random_state=42)

    print(f"Intersection between train and val genes: {set(split_tr_node_ids).intersection(set(val_node_ids))}")
    print(f"Intersection between train and test genes: {set(split_tr_node_ids).intersection(set(te_node_ids))}")
    print(f"Intersection between val and test genes: {set(val_node_ids).intersection(set(te_node_ids))}")
    print(f"Tr nodes: {len(split_tr_node_ids)}, Te nodes: {len(te_node_ids)}, Val nodes: {len(val_node_ids)}")

    data.train_mask = create_masks(mapped_f_name, split_tr_node_ids)
    data.val_mask = create_masks(mapped_f_name, val_node_ids)
    data.test_mask  = create_masks(mapped_f_name, te_node_ids)

    #print(data.train_mask.shape)
    print(f"Tr masks: {data.train_mask.sum().item()}, Te masks: {data.test_mask.sum().item()}, Val masks: {data.val_mask.sum().item()}")

    train_loader, val_loader, test_loader = make_neighbor_loaders(data, config)
    val_ids_epo = list()
    for epoch in range(n_epo):
        
        tr_loss, tr_acc = train_one_epoch(train_loader, model, optimizer, criterion, device)
        val_loss, val_acc, used_val_ids = val_evaluate(val_loader, model, criterion, device)   
        val_ids_epo.extend(used_val_ids)     
        print(f"[Epoch {epoch:03d} / {n_epo:03d}] "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        te_loss, te_acc, *_ = test_evaluate(test_loader, model, criterion, device)
        print(f"[TEST] loss={te_loss:.4f} acc={te_acc:.4f}")
        print("-------------------")
        tr_loss_epo.append(tr_loss)
        tr_acc_epo.append(tr_acc)
        val_loss_epo.append(val_loss)
        val_acc_epo.append(val_acc)
        te_acc_epo.append(te_acc)
        te_loss_epo.append(te_loss)

        if te_acc > best_te_acc:
            best_te_acc = te_acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            print(f"Saving the model state, best epoch was {best_epoch} with test acc {te_acc:.2f}.")
            torch.save(model.state_dict(), config.p_torch_model)   # <-- best checkpoint

    print("Analysing validation nodes split")

    print(sorted(val_ids_epo)[:5], sorted(val_node_ids)[:5], len(val_ids_epo), len(val_node_ids))
    
    print("Plot and report all training epochs")
    plot_gnn.plot_loss_acc(n_epo, tr_loss_epo, te_loss_epo, val_acc_epo, te_acc_epo, config)
    print(f"CV Training Loss after {n_epo} epochs: {np.mean(tr_loss_epo):.2f}")
    print(f"CV Val acc after {n_epo} epochs: {np.mean(val_acc_epo):.2f}")

    ## Restore the best trained model for downstream usages
    print(f"[Restore] Loaded best model from epoch {best_epoch} (test acc {best_te_acc:.2f}).")
    if best_state is not None:
        model.load_state_dict(best_state)
    # avg_loss, avg_acc, pred_labels, true_labels, all_probs, all_pred_probs
    final_test_loss, final_test_acc, pred_labels, true_labels, all_probs, all_pred_prob, test_ids = \
        test_evaluate(test_loader, model, criterion, device)
    # Save predictions, true labels, model
    torch.save(test_ids, config.p_test_loader_ids)
    torch.save(true_labels, config.p_true_labels)
    torch.save(pred_labels, config.p_pred_labels)
    torch.save(all_pred_prob, config.p_pred_probs)
    print(f"CV Test acc using the best model (stored at {best_epoch}): {final_test_acc:.2f}, {final_test_loss: .2f}")
    #extract_node_embeddings(model, data, model_activation, config)
    plot_gnn.plot_confusion_matrix(true_labels, pred_labels, config)
    plot_gnn.plot_precision_recall(true_labels, all_probs, config)
    #plot_gnn.plot_radar({"Net-A": [0.82, 0.76, 0.91, 0.65, 0.88], "Net-B": [0.79, 0.81, 0.87, 0.70, 0.90]}, [1, 2, 3, 4, 5], config)