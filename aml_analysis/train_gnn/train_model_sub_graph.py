import copy
from pathlib import Path
import torch
import math
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, degree
from sklearn.metrics import f1_score, precision_recall_fscore_support

import joblib

import numpy as np
import pandas as pd

import gnn_network
import plot_gnn
import utils


detach = utils.detach_from_gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_masks(mapped_node_ids, mask_list):
    # True where the SERIES VALUE (node id/name) is in the mask list
    mask = torch.tensor([x in mask_list for x in mapped_node_ids], dtype=torch.bool)
    return mask


def ensure_bool(m):
    return m if m.dtype == torch.bool else m.bool()


def filter_edges(edge_index, allowed_nodes):
    allowed = allowed_nodes  # set(allowed_nodes.tolist())
    src, dst = edge_index

    mask = [(u.item() in allowed and v.item() in allowed) for u, v in zip(src, dst)]
    mask = torch.tensor(mask, dtype=torch.bool, device=edge_index.device)
    return edge_index[:, mask]


def make_neighbor_loaders(data, train_nodes, val_nodes, te_nodes, config):
    # Keep the big graph on CPU
    data = data.cpu()

    data.edge_index = coalesce(data.edge_index, num_nodes=data.num_nodes)

    print(
        f"Intersection between train and val genes: {set(train_nodes).intersection(set(val_nodes))}"
    )
    print(
        f"Intersection between train and test genes: {set(train_nodes).intersection(set(te_nodes))}"
    )
    print(
        f"Intersection between val and test genes: {set(val_nodes).intersection(set(te_nodes))}"
    )
    print(
        f"Tr nodes: {len(train_nodes)}, Te nodes: {len(te_nodes)}, Val nodes: {len(val_nodes)}"
    )

    train_loader = NeighborLoader(
        data,
        input_nodes=ensure_bool(data.train_mask),
        num_neighbors=config.neighbors_spread,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=int(config.get("num_workers", 8)),
        pin_memory=True,
        subgraph_type=config.graph_subtype,
    )

    val_loader = NeighborLoader(
        data,
        input_nodes=ensure_bool(data.val_mask),
        num_neighbors=config.neighbors_spread,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=int(config.get("num_workers", 8)),
        pin_memory=True,
        subgraph_type=config.graph_subtype,
    )
    test_loader = NeighborLoader(
        data,
        input_nodes=ensure_bool(data.test_mask),
        num_neighbors=config.neighbors_spread,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=int(config.get("num_workers", 8)),
        pin_memory=True,
        subgraph_type=config.graph_subtype,
    )
    return train_loader, val_loader, test_loader


def train_one_epoch(train_loader, model, optimizer, criterion, device, grad_clip_norm=None):
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0

    for _, batch in enumerate(train_loader):
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out, *_ = model(batch.x, batch.edge_index)
        seed_n = batch.batch_size  # first seed_n nodes = seeds
        logits = out[:seed_n]
        targets = batch.y[:seed_n].long()
        loss = criterion(logits, targets)
        loss.backward()
        if grad_clip_norm is not None:
            clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        optimizer.step()

        total_loss += float(loss.detach()) * seed_n
        total_correct += (logits.argmax(-1) == targets).sum().item()
        total_count += seed_n

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count
    return avg_loss, avg_acc


@torch.no_grad()
def val_evaluate(loader, model, criterion, device):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    used_val_ids = []
    val_targets, val_predictions = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            seed_n = batch.batch_size
            used_val_ids.extend(batch.n_id.cpu().tolist()[:seed_n])
            out, *_ = model(batch.x, batch.edge_index)
            # evaluating only the seed nodes of this batch
            logits = out[:seed_n]
            targets = batch.y[:seed_n].long()
            val_targets.extend(targets.cpu().tolist())
            val_predictions.extend(logits.argmax(-1).cpu().tolist())
            loss = criterion(logits, targets)

            total_loss += float(loss.detach()) * seed_n
            total_correct += (logits.argmax(-1) == targets).sum().item()
            total_count += seed_n

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count
    macro_f1 = f1_score(val_targets, val_predictions, labels=np.arange(5), average="macro", zero_division=0)
    return avg_loss, avg_acc, used_val_ids, float(macro_f1)


@torch.no_grad()
def test_evaluate(loader, model, criterion, device):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    with torch.no_grad():
        (
            pred_labels,
            true_labels,
            all_probs,
            best_class_pred_probs,
            test_ids,
            emb_pna4,
            emb_bn4,
        ) = ([], [], [], [], [], [], [])
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

            batch_pred_label = logits.argmax(-1)
            total_loss += float(detach(loss)) * seed_n
            pred_labels.extend(detach(batch_pred_label))
            true_labels.extend(detach(targets))
            all_probs.extend(detach(batch_prob))
            best_class_pred_probs.extend(detach(batch_max_prob))
            total_correct += (batch_pred_label == targets).sum().item()
            total_count += seed_n

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count
    return (
        avg_loss,
        avg_acc,
        pred_labels,
        true_labels,
        all_probs,
        best_class_pred_probs,
        test_ids,
        emb_pna4,
        emb_bn4,
    )


def validate_predefined_splits(data, out_genes, config):
    """Require the original node partitions throughout training and evaluation."""
    if out_genes[0].tolist() != list(range(data.num_nodes)):
        raise ValueError("Graph node IDs must match tensor row order")
    manifest = pd.read_csv(config.p_split_manifest)
    if not manifest["feature_name"].is_unique:
        raise ValueError("Split manifest contains duplicate node names")
    split_by_name = manifest.set_index("feature_name")["split"]
    assignments = split_by_name.reindex(out_genes[1].tolist())
    if not assignments.isin(["train", "val", "test"]).all():
        raise ValueError("Every graph node must have a predefined train, val or test split")
    split_codes = torch.full((data.num_nodes,), -1, dtype=torch.long)
    for code, split in enumerate(("train", "val", "test")):
        expected = torch.tensor(assignments.eq(split).to_numpy(), dtype=torch.bool)
        actual = getattr(data, f"{split}_mask", None)
        if actual is None or actual.dtype != torch.bool or not torch.equal(actual.cpu(), expected):
            raise ValueError(f"{split} mask does not match the original split manifest")
        if not expected.any():
            raise ValueError(f"The {split} partition is empty")
        split_codes[expected] = code
    src, dst = data.edge_index.cpu()
    if not torch.equal(split_codes[src], split_codes[dst]):
        raise ValueError("Graph edges must not connect different dataset partitions")


def train_gnn_model(config, labels, chosen_model):
    """
    Create network architecture and assign loss, optimizers ...
    """
    n_epo = config.n_epo
    out_genes = pd.read_csv(config.p_out_genes, sep=" ", header=None)
    node_ids = out_genes.loc[:, 0].tolist()

    print(f"Used device: {device}")

    data = torch.load(config.p_torch_data, weights_only=False)
    validate_predefined_splits(data, out_genes, config)
    if hasattr(data, "x"):
        data.x = data.x.to(torch.float32)
    deg = degree(
        data.edge_index[0], num_nodes=data.num_nodes
    )  # in-degree or out-degree
    print(f"Mean degree: {deg.mean().item()}")
    print(f"Max degree: {deg.max().item()}")
    print(f"Std dev: {deg.std().item()}")
    print(f"Median degree: {deg.median().item()}")
    ratio = deg.max().item() / (deg.mean().item() + 1e-9)
    print(f"Max / Mean ratio: {ratio}")
    tr_nodes = pd.read_csv(config.p_train_probe_genes)
    val_nodes = pd.read_csv(config.p_val_probe_genes)
    te_nodes = pd.read_csv(config.p_test_probe_genes)
    te_node_ids = te_nodes["test_gene_ids"].tolist()
    tr_node_ids = tr_nodes["train_gene_ids"].tolist()
    val_node_ids = val_nodes["val_gene_ids"].tolist()

    print(f"tr_node_ids: {tr_node_ids[-10:]}")
    if tr_node_ids:
        print(f"Example training label: {data.y[tr_node_ids[0]]}")
        print(f"Example training features: {data.x[tr_node_ids[0]]}")

    tr_loss_epo = list()
    tr_acc_epo = list()
    te_loss_epo = list()
    te_acc_epo = list()
    val_loss_epo = list()
    val_acc_epo = list()
    best_val_acc = -float("inf")
    best_val_macro_f1 = -float("inf")
    best_selection_key = (-float("inf"), -float("inf"))
    selection_metric = str(config.get("selection_metric", "accuracy"))
    if selection_metric not in ("accuracy", "macro_f1"):
        raise ValueError("selection_metric must be accuracy or macro_f1")
    val_macro_f1_epo = []
    best_state = None
    best_epoch = -1

    print(
        f"Intersection between train and val genes: {set(tr_node_ids).intersection(set(val_node_ids))}"
    )
    print(
        f"Intersection between train and test genes: {set(tr_node_ids).intersection(set(te_node_ids))}"
    )
    print(
        f"Intersection between val and test genes: {set(val_node_ids).intersection(set(te_node_ids))}"
    )
    print(
        f"Tr nodes: {len(tr_node_ids)}, Te nodes: {len(te_node_ids)}, Val nodes: {len(val_node_ids)}"
    )

    # Masks were established before NeDBIT calculation and are stored with the
    # merged disconnected graph. Refuse inconsistent auxiliary split files.
    for split, ids in (("train", tr_node_ids), ("val", val_node_ids), ("test", te_node_ids)):
        if len(ids) != len(set(ids)) or not torch.equal(
            getattr(data, f"{split}_mask").cpu(), create_masks(node_ids, set(ids))
        ):
            raise ValueError(f"{split} node file does not match the predefined partition")

    print(f"Initialize model: {chosen_model}")
    model = utils.choose_model(config, data, chosen_model, True)
    for name, module in model.named_modules():
        print(f"{name}: {module}")
    model = model.to(device)

    # loss fn
    class_weights = None
    weight_power = float(config.get("class_weight_power", 0.0))
    if weight_power:
        counts = torch.bincount(data.y[data.train_mask], minlength=config.num_classes).float()
        if (counts == 0).any():
            raise ValueError("Class weighting requires every class in training")
        class_weights = (counts.sum() / (config.num_classes * counts)).pow(weight_power)
        class_weights /= class_weights.mean()
        class_weights = class_weights.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    base_lr = config.learning_rate
    weight_decay = config.weight_decay
    optimizer = AdamW(
        model.parameters(), lr=base_lr, weight_decay=weight_decay
    )  # lr=base_lr, weight_decay=weight_decay

    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=int(config.get("lr_patience", 1)),
        threshold=0.0, threshold_mode="abs", min_lr=1e-6,
    )
    early_stopping_patience = int(config.get("early_stopping_patience", 5))
    if early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive")
    epochs_without_improvement = 0

    print(
        f"Tr masks: {data.train_mask.sum().item()}, Te masks: {data.test_mask.sum().item()}, Val masks: {data.val_mask.sum().item()}"
    )
    print("Plotting UMAP using raw features ...")
    test_x = data.x[data.test_mask == 1]
    test_y = data.y[data.test_mask == 1]
    plot_gnn.plot_features(
        test_x,
        test_y,
        chosen_model,
        config,
        f"UMAP of raw features (NeDBIT + DNA Methylation): {chosen_model}",
        "test_before_GNN",
    )
    print("Creating neighbor loaders ...")
    train_loader, val_loader, test_loader = make_neighbor_loaders(
        data, tr_node_ids, val_node_ids, te_node_ids, config
    )
    print("Start training ...")
    for epoch in range(n_epo):
        tr_loss, tr_acc = train_one_epoch(
            train_loader, model, optimizer, criterion, device,
            grad_clip_norm=config.get("grad_clip_norm", None),
        )
        # Repeat the same sampled validation neighborhoods each epoch without
        # changing the random stream used for training or dropout.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(config.SEED))
            val_loss, val_acc, used_val_ids, val_macro_f1 = val_evaluate(
                val_loader, model, criterion, device
            )
        selection_score = val_macro_f1 if selection_metric == "macro_f1" else val_acc
        selection_key = (selection_score, val_acc if selection_metric == "macro_f1" else -val_loss)
        scheduler.step(selection_score)
        print(
            f"[Epoch {epoch+1:03d} / {n_epo:03d}] "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f}"
        )
        print("-------------------")
        tr_loss_epo.append(tr_loss)
        tr_acc_epo.append(tr_acc)
        val_loss_epo.append(val_loss)
        val_acc_epo.append(val_acc)
        val_macro_f1_epo.append(val_macro_f1)

        if selection_key > best_selection_key:
            best_selection_key = selection_key
            best_val_macro_f1 = val_macro_f1
            epochs_without_improvement = 0
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            print(
                f"Saving epoch {best_epoch}: validation {selection_metric}={selection_score:.4f}"
            )
            Path(config.p_torch_model).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.p_torch_model)  # best validation checkpoint

        else:
            epochs_without_improvement += 1
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6g}")
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping: no validation {selection_metric} improvement for {early_stopping_patience} epochs")
            break

    completed_epochs = len(tr_loss_epo)
    print("Plot and report all completed training epochs")

    plot_gnn.plot_loss_acc(
        completed_epochs, tr_loss_epo, val_loss_epo, tr_acc_epo, val_acc_epo, chosen_model, config
    )
    print(f"Mean training loss over {completed_epochs} epochs: {np.mean(tr_loss_epo):.2f}")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    ## Restore the best trained model for downstream usages
    print(
        f"[Restore] Loaded best model from epoch {best_epoch} (val acc {best_val_acc:.2f})."
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    # avg_loss, avg_acc, pred_labels, true_labels, all_probs, all_pred_probs
    (
        final_test_loss,
        final_test_acc,
        pred_labels,
        true_labels,
        all_class_pred_probs,
        best_class_pred_probs,
        test_ids,
        embs_pna4,
        embs_bn4,
    ) = test_evaluate(test_loader, model, criterion, device)
    te_acc_epo.append(final_test_acc)
    te_loss_epo.append(final_test_loss)

    # Save predictions, true labels, model
    torch.save(embs_pna4, config.p_torch_embed)
    torch.save(embs_bn4, config.p_torch_embed_batch_norm)
    torch.save(test_ids, config.p_test_loader_ids)
    torch.save(true_labels, config.p_true_labels)
    torch.save(pred_labels, config.p_pred_labels)
    torch.save(best_class_pred_probs, config.p_best_class_pred_probs)
    torch.save(all_class_pred_probs, config.p_all_class_pred_probs)

    print(
        f"Test acc using the best model (stored at {best_epoch}): acc {final_test_acc:.2f}, loss {final_test_loss: .2f}"
    )

    te_f1_macro = f1_score(true_labels, pred_labels, average="macro")
    te_f1_weighted = f1_score(true_labels, pred_labels, average="weighted")
    te_f1_micro = f1_score(true_labels, pred_labels, average="micro")
    te_prec, te_recall, *_ = precision_recall_fscore_support(
        true_labels, pred_labels, average="weighted"
    )

    metrics = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_macro_f1": best_val_macro_f1,
        "selection_metric": selection_metric,
        "val_macro_f1": val_macro_f1_epo,
        "completed_epochs": completed_epochs,
        "te_f1_macro": te_f1_macro,
        "te_f1_weighted": te_f1_weighted,
        "te_f1_micro": te_f1_micro,
        "te_precision": te_prec,
        "te_recall": te_recall,
        "tr_loss": tr_loss_epo,
        "te_loss": te_loss_epo,
        "val_loss": val_loss_epo,
        "tr_acc": tr_acc_epo,
        "te_acc": te_acc_epo,
        "val_acc": val_acc_epo,
    }

    print(f"All metrics: {metrics}")
    utils.save_accuracy_scores(
        metrics, Path(config.p_plot) / f"all_metrics_{chosen_model}.json"
    )
    print("Plotting metrics and UMAP plots for final embeddings")
    plot_gnn.plot_node_embed(
        embs_pna4, true_labels, pred_labels, chosen_model, config, "PNAConv4"
    )
    plot_gnn.plot_node_embed(
        embs_bn4, true_labels, pred_labels, chosen_model, config, "BatchNorm1d"
    )
    plot_gnn.plot_confusion_matrix(true_labels, pred_labels, chosen_model, config)
    # plot_gnn.plot_precision_recall(true_labels, all_class_pred_probs, chosen_model, config)
    print("Finished.")
    return metrics
