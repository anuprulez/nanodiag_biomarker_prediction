import copy
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from sklearn.model_selection import train_test_split
from torch_geometric.utils import coalesce
from sklearn.metrics import f1_score, precision_recall_fscore_support
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import gnn_network
import plot_gnn
import utils

detach = utils.detach_from_gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_masks(mapped_node_ids: pd.Series, mask_list):
    # True where the SERIES VALUE (node id/name) is in the mask list
    mask = mapped_node_ids.isin(mask_list).to_numpy()
    return torch.tensor(mask, dtype=torch.bool)


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
        out, *_ = model(batch.x, batch.edge_index)
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
        out, out_pna4, out_bn4 = model(batch.x, batch.edge_index)
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
    pred_labels, true_labels, all_probs, best_class_pred_probs, test_ids, \
        emb_pna4, emb_bn4 = [], [], [], [], [], [], []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        out, out_pna4, out_bn4 = model(batch.x, batch.edge_index)
        seed_n = batch.batch_size  # evaluating only the seed nodes of this batch
        test_ids.extend(batch.n_id.cpu().tolist()[:seed_n])
        logits = out[:seed_n]
        emb_pna4.extend(detach(out_pna4[:seed_n]))
        emb_bn4.extend(detach(out_bn4[:seed_n]))
        batch_prob = F.softmax(logits, dim=1)
        batch_max_prob = batch_prob.max(dim=1).values
        targets = batch.y[:seed_n].long()
        loss = criterion(logits, targets)

        total_loss += float(detach(loss)) * seed_n
        batch_pred_label = logits.argmax(-1)
        pred_labels.extend(detach(batch_pred_label))
        true_labels.extend(detach(targets))
        all_probs.extend(detach(batch_prob))
        best_class_pred_probs.extend(detach(batch_max_prob))
        total_correct += (logits.argmax(-1) == targets).sum().item()
        total_count += seed_n

    avg_loss = total_loss / max(1, total_count)
    avg_acc = total_correct / max(1, total_count)
    return avg_loss, avg_acc, pred_labels, true_labels, all_probs, best_class_pred_probs, test_ids, emb_pna4, emb_bn4


def choose_model(config, data):
    if config.model_type == "pna":
        model = gnn_network.GPNA(config, data)
    elif config.model_type == "gcn":
        model = gnn_network.GCN(config)
    elif config.model_type == "gsage":
        model = gnn_network.GraphSAGE(config)
    elif config.model_type == "gin":
        model = gnn_network.GraphSAGE(config)
    elif config.model_type == "gatv2":
        model = gnn_network.GATv2(config)
    elif config.model_type == "gcn2":
        model = gnn_network.GCNII(config)
    elif config.model_type == "appnet":
        model = gnn_network.APPNPNet(config)
    elif config.model_type == "gtran":
        model = gnn_network.GraphTransformer(config)
    else:
        model = gnn_network.GPNA(config, data)
    return model


def train_gnn_model(config):
    """
    Create network architecture and assign loss, optimizers ...
    """
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

    print(f"Initialize model: {config.model_type}")
    model = choose_model(config, data)
    model = model.cuda()

    # loss fn
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
    print(f"Tr masks: {data.train_mask.sum().item()}, Te masks: {data.test_mask.sum().item()}, Val masks: {data.val_mask.sum().item()}")
    print("Plotting UMAP using raw features")
    test_x = data.x[data.test_mask == 1]
    test_y = data.y[data.test_mask == 1]
    plot_gnn.plot_features(test_x, test_y, config, "UMAP of raw NedBit + DNA Methylation features", "test_before_GNN")

    train_loader, val_loader, test_loader = make_neighbor_loaders(data, config)
    val_ids_epo = list()
    for epoch in range(n_epo):
        
        tr_loss, tr_acc = train_one_epoch(train_loader, model, optimizer, criterion, device)
        val_loss, val_acc, used_val_ids = val_evaluate(val_loader, model, criterion, device)   
        val_ids_epo.extend(used_val_ids)     
        print(f"[Epoch {epoch:03d} / {n_epo:03d}] "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        te_loss, te_acc, *_  = test_evaluate(test_loader, model, criterion, device)
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
    
    print("Plot and report all training epochs")
    
    plot_gnn.plot_loss_acc(n_epo, tr_loss_epo, te_loss_epo, val_acc_epo, te_acc_epo, config)
    print(f"CV Training Loss after {n_epo} epochs: {np.mean(tr_loss_epo):.2f}")
    print(f"CV Val acc after {n_epo} epochs: {np.mean(val_acc_epo):.2f}")

    ## Restore the best trained model for downstream usages
    print(f"[Restore] Loaded best model from epoch {best_epoch} (test acc {best_te_acc:.2f}).")
    if best_state is not None:
        model.load_state_dict(best_state)
    # avg_loss, avg_acc, pred_labels, true_labels, all_probs, all_pred_probs
    final_test_loss, final_test_acc, pred_labels, true_labels, all_class_pred_probs, best_class_pred_probs, test_ids, embs_pna4, embs_bn4 = \
        test_evaluate(test_loader, model, criterion, device)

    # Save predictions, true labels, model
    torch.save(test_ids, config.p_test_loader_ids)
    torch.save(true_labels, config.p_true_labels)
    torch.save(pred_labels, config.p_pred_labels)
    torch.save(best_class_pred_probs, config.p_best_class_pred_probs)
    torch.save(all_class_pred_probs, config.p_all_class_pred_probs)

    print(f"CV Test acc using the best model (stored at {best_epoch}): {final_test_acc:.2f}, {final_test_loss: .2f}")
    plot_gnn.plot_confusion_matrix(true_labels, pred_labels, config)
    plot_gnn.plot_precision_recall(true_labels, all_class_pred_probs, config)

    te_f1_macro = f1_score(true_labels, pred_labels, average='macro')
    te_f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
    te_f1_micro = f1_score(true_labels, pred_labels, average="micro")
    te_prec, te_recall, *_ = precision_recall_fscore_support(true_labels, pred_labels, average="weighted")

    metrics = {
        "te_f1_macro": te_f1_macro,
        "te_f1_weighted": te_f1_weighted,
        "te_f1_micro": te_f1_micro,
        "te_precision": te_prec,
        "te_recall":te_recall,
        "tr_loss": tr_loss_epo,
        "te_loss": te_loss_epo,
        "val_acc": val_acc_epo,
        "te_acc": te_acc_epo
    }

    print(f"All metrics: {metrics}")
    utils.save_accuracy_scores(metrics, f"{config.p_plot}all_metrics_{config.model_type}.json")
    plot_gnn.plot_node_embed(embs_pna4, true_labels, pred_labels, config, "PNAConv4")
    plot_gnn.plot_node_embed(embs_bn4, true_labels, pred_labels, config, "BatchNorm1d")