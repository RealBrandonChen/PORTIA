import portia as pt
import numpy as np
import matplotlib as plt
import os

parent_dir = "G:\\GeorgiaTech\\OneDrive - Georgia Institute of Technology\\Documents\\Courses\\GT 2023\\BMED-6517_Machine-learning-Biosci\\Project1\\New folder\PORTIA\\"
data_name = "100_mr_50_cond\simulated_noNoise.txt"
data_path = os.path.join(parent_dir, data_name) 
sim_data = np.loadtxt(data_path)
sim_data = sim_data[1:]

# for exp_id, data in enumerate(sim_data):
#     print("exp_id: " + str(exp_id))
#     print("data: " + str(data))

# exp_id from 0, data is the list of single experiment.    
dataset = pt.GeneExpressionDataset()
exp_id = 1
data = [0, 0, ..., 1.03424, 1.28009]

for exp_id, data in enumerate(sim_data):
    dataset.add(pt.Experiment(exp_id, data))
    
tf_idx = np.arange(100)
M_bar, S = pt.run(dataset, tf_idx=tf_idx, method='fast', return_sign=True)

res_path = os.path.join(parent_dir, "results.txt") 
gene_names = np.arange(200)
with open(res_path, 'w') as f:
    for gene_a, gene_b, score in pt.rank_scores(M_bar, gene_names, limit=10000):
        f.write(f'{gene_a}\t{gene_b}\t{score}\n')
        
# tf_mask = np.zeros(n_genes, dtype=bool)
# tf_mask[tf_idx] = True
# res = graph_theoretic_evaluation(tmp_filepath, G_target, G_pred, tf_mask=tf_mask)

