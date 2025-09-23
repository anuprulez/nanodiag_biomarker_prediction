from torch_geometric.explain import Explainer, GNNExplainer, GraphMaskExplainer, CaptumExplainer
from torch_geometric.data import Data
import matplotlib.pyplot as plt


def gnn_explainer(model, data, config):
    plot_local_path = config["plot_local_path"]
    print("Running GNN explanation...")
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
        threshold_config=dict(
            threshold_type='topk',
            value=30,
        )
    )
    '''explainer = Explainer(
        model=model,
        algorithm=CaptumExplainer('IntegratedGradients'),
        explanation_type='model',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
        node_mask_type='attributes',
        edge_mask_type='object',
        threshold_config=dict(
            threshold_type='topk',
            value=200,
        ),
    )'''
    # list of ids to explore
    explore_test_ids = [10841]
    plot_local_path += "explainer_plots/" 
    for node_i in explore_test_ids: #10841
        node_index = node_i
        explanation = explainer(data.x, data.edge_index, index=node_index)
        print(data.x[node_index])
        print()
        plt.figure(figsize=(8, 6))
        print(f'Generated explanations in {explanation.available_explanations}')
        #path = plot_local_path + 'feature_importance_{}.png'.format(node_index)
        #explanation.visualize_feature_importance(path, top_k=10)
        #print(f"Feature importance plot has been saved to '{path}'")
        plt.figure(figsize=(8, 6))
        path = plot_local_path + 'subgraph_{}.pdf'.format(node_index)
        explanation.visualize_graph(path)
        print(f"Subgraph visualization plot has been saved to '{path}'")
        
        #get_explanation_subgraph = explanation.get_explanation_subgraph()
        #explanation_path = plot_local_path + 'explanation_subgraph_subgraph_{}.pdf'.format(node_index)
        #get_explanation_subgraph.visualize_graph(explanation_path)
        
        #get_complement_subgraph = explanation.get_complement_subgraph()
        #print(get_explanation_subgraph)
        #print(get_complement_subgraph)
        #print(explanation.edge_mask)
        #print(explanation.node_mask)
        print("----")