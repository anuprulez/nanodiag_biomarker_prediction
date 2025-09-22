import preprocess_data
import train_model
#import xai_explainer

from omegaconf.omegaconf import OmegaConf

def run_training():
    config = OmegaConf.load("../breast_cancer_analysis/config/config.yaml")
    preprocess_data.read_files(config)
    train_model.train_gnn_model(config)
    #xai_explainer.gnn_explainer(trained_model, data, config)


if __name__ == "__main__":
    
    run_training()