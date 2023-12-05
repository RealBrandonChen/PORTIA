import portia as pt
import numpy as np
import matplotlib as plt
import os
from evalTools import *

parent_dir = "E:\\Courses\\Repo\PORTIA\\"
data_num = [100, 40, 5]
data_name = ["{}_mr_50_cond\simulated_noNoise.txt".format(data_num[0]), "{}_mr_50_cond\simulated_noNoise.txt".format(data_num[1]), "{}_mr_50_cond\simulated_noNoise.txt".format(data_num[2])]
parent_dir = "E:\\Courses\\Repo\PORTIA\\sorted_data\\"


for i in range(len(data_num)):
    
    data_path = os.path.join(parent_dir, data_name[i]) 
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
    ## Use for loop to test different parameters
    method = ['fast', 'end-to-end', 'no-transform']       
    for method in method:
        M_bar, S = pt.run(dataset, tf_idx=tf_idx, method=method, return_sign=True)
        res_path = os.path.join(parent_dir, "sorted_data/results_{}.txt".format(method)) 
        gene_names = np.arange(200)
        with open(res_path, 'w') as f:
            for gene_a, gene_b, score in pt.rank_scores(M_bar, gene_names, limit=10000):
                f.write(f'{gene_a}\t{gene_b}\t{score}\n')
            
        # Call sorting function
        top_n = [100, 200, 300 ,400]
        for top_n in top_n:
            sortResults(res_path, method=method, top_n=top_n, testSetNum=data_num[i])