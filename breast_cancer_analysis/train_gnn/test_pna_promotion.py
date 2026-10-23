"""Exercise frozen-checkpoint evaluation and PNA-only installation on a tiny graph."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from promote_pna_accuracy import choose_model,seed_all,val_evaluate
from seed_inputs import seed_signature,write_seed_manifest

class PnaPromotionTests(unittest.TestCase):
    def test_freeze_evaluate_install_and_reject_repeat(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);gen=root/'generated';study=gen/'optuna/fixture/PNA';prepared=study/'prepared'
            prepared.mkdir(parents=True)
            names=[f'cg{i:08d}_G' for i in range(100)]
            splits=['train']*50+['val']*25+['test']*25
            nodes=root/'nodes.csv';manifest=root/'splits.csv'
            pd.DataFrame({0:range(100),1:names}).to_csv(nodes,sep=' ',index=False,header=False)
            pd.DataFrame({'feature_name':names,'split':splits}).to_csv(manifest,index=False)
            seeds=root/'seeds.tsv';seeds.write_text('probeID\tdeltaBeta\tP.Value\tCHR\ncg00000001\t0.1\t0.01\t1\n')
            c=OmegaConf.create(dict(SEED=32,model_type='PNA',gene_dim=6,num_classes=5,hidden_dim=32,
                gnn_layers=1,dropout=.1,feature_dropout=0.,embedding_dropout=0.,class_weight_power=1.,
                num_workers=0,batch_size=32,neighbors_spread=[2],n_epo=2,lr_patience=2,early_stopping_patience=2,
                grad_clip_norm=1.,learning_rate=.001,weight_decay=.001,checkpoint_selection_metric='accuracy',
                p_out_genes=str(nodes),p_split_manifest=str(manifest),p_seeds=str(seeds),
                p_preprocessing_manifest=str(root/'seed_manifest.json'),p_generated_files=str(study/'final')+'/'))
            write_seed_manifest(c,seed_signature(seeds))
            seed_all(32)
            edges=[]
            for first,last in [(0,50),(50,75),(75,100)]:
                for i in range(first,last-1):edges.extend([(i,i+1),(i+1,i)])
            data=Data(x=torch.rand(100,6),y=torch.arange(100)%5,edge_index=torch.tensor(edges).T.contiguous(),
                      train_mask=torch.tensor(np.array(splits)=='train'),val_mask=torch.tensor(np.array(splits)=='val'),
                      test_mask=torch.tensor(np.array(splits)=='test'))
            torch.save(data,gen/'data_bc_PNA.pt');torch.save(data,prepared/'data_bc_PNA.pt')
            for split in ('train','val','test'):
                ids=[i for i,s in enumerate(splits) if s==split]
                pd.DataFrame({split+'_gene_ids':ids,split+'_gene_names':[names[i] for i in ids]}).to_csv(prepared/f'{split}_probe_genes_bc_PNA.csv',index=False)
            model=choose_model(c,data,'PNA',False)
            torch.save(model.state_dict(),study/'best_model.pt')
            counts=torch.bincount(data.y[data.train_mask],minlength=5).float();weights=counts.sum()/(5*counts);weights/=weights.mean()
            criterion=torch.nn.CrossEntropyLoss(weight=weights)
            seed_all(32)
            loader=NeighborLoader(data,input_nodes=data.val_mask,num_neighbors=[2],batch_size=32,shuffle=False,num_workers=0,subgraph_type='bidirectional')
            loss,accuracy,_,f1=val_evaluate(loader,model,criterion,torch.device('cpu'))
            OmegaConf.save(c,study/'best_config.yaml')
            (study/'best_params.json').write_text(json.dumps({'trial':0,'validation_metric':'accuracy','validation_score':accuracy}))
            trial=study/'trial_00000';trial.mkdir();(trial/'validation_history.json').write_text(json.dumps([{'epoch':1,'accuracy':accuracy,'macro_f1':f1,'loss':loss}]))
            (gen/'all_metrics_PNA.json').write_text(json.dumps({'best_val_acc':0.,'te_acc':[.1]}))
            pd.DataFrame([{'model':'PNA','best_val_acc':0.,'te_acc':.1,'te_f1_macro':.1,'te_f1_weighted':.1,
                           'te_f1_micro':.1,'te_precision':.1,'te_recall':.1},
                          {'model':'GCN','best_val_acc':.2,'te_acc':.2,'te_f1_macro':.2,'te_f1_weighted':.2,
                           'te_f1_micro':.2,'te_precision':.2,'te_recall':.2}]).to_csv(gen/'Heatmap_avg_edges_9913451_epochs_100.csv',index=False)
            source=root/'config.yaml'
            OmegaConf.save(OmegaConf.create({'model_overrides':{'PNA':{'hidden_dim':64},'GCN':{'hidden_dim':16}},'split_ratios':[.7,.15,.15]}),source)
            command=[sys.executable,str(Path(__file__).parent/'promote_pna_accuracy.py'),'--study',str(study),'--config',str(source),'--install']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
            result=subprocess.run(command,capture_output=True,text=True,env=env)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            output=gen/'pna_accuracy_improved'
            self.assertTrue((output/'selection_frozen_before_test.json').is_file())
            self.assertTrue((output/'Heatmap_PNA_improved.pdf').is_file())
            updated=OmegaConf.load(source)
            self.assertEqual(updated.model_overrides.GCN.hidden_dim,16)
            self.assertEqual(updated.model_overrides.PNA.hidden_dim,32)
            self.assertEqual(OmegaConf.to_container(updated,resolve=False)['model_overrides']['PNA']['p_generated_files'],'${p_base}generated/pna_accuracy_improved/')
            repeat=subprocess.run(command,capture_output=True,text=True,env=env)
            self.assertNotEqual(repeat.returncode,0)
            self.assertIn('do not repeat test',repeat.stderr)

if __name__=='__main__':unittest.main()
