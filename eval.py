import matplotlib
matplotlib.use('Agg')
from utils.data_reader import Personas
from model.seq2spg import Seq2SPG
from model.common_layer import NoamOpt
from model.common_layer import evaluate
import pickle
from utils import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import time
import numpy as np 
from random import shuffle
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import math

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def do_learning(model, train_iter, val_iter, iterations, task=0):
    logger = {str(i): [] for i in range(iterations)}
    loss, ppl_val, ent_b,bleu_score_b = evaluate(model, val_iter, model_name=config.model,ty="test",verbose=False)
    logger[str(0)] = [loss, ppl_val, ent_b, bleu_score_b]
    for i in range(1,iterations):
        if i < 5:
            m = "select"
        else:
            m = "selective_training"
        for j, d in enumerate(train_iter):
            _, _, _ = model.train_one_batch(d, mode=m, task=task)
        if(i in list(range(1, 26))):#1,3,5,7,
            loss, ppl_val, ent_b, bleu_score_b = evaluate(model, val_iter, model_name=config.model,ty="test",verbose=False, log=False, result_file="results/results_our " + str(i) +".txt",ref_file="results/ref_our" + str(i) + ".txt", case_file="results/case_our" + str(i) + ".txt")
            logger[str(i)] = [loss, ppl_val, ent_b, bleu_score_b]
    return logger

p = Personas()
# Build model, optimizer, and set states
print("Test model",config.model)
model = Seq2SPG(p.vocab,model_file_path=config.save_path,is_eval=False)
fine_tune = []
iter_per_task = []
iterations = 26
weights_original = deepcopy(model.state_dict())
tasks = p.get_personas('test')
for per in tqdm(tasks):
    num_of_dialog = p.get_num_of_dialog(persona=per, split='test')
    for val_dial_index in range(num_of_dialog):
        if config.fix_dialnum_train:
            train_iter, val_iter = p.get_balanced_loader(persona=per,batch_size=config.batch_size, split='test', fold=val_dial_index, dial_num=config.k_shot)

        else:
            train_iter, val_iter = p.get_data_loader(persona=per,batch_size=config.batch_size, split='test', fold=val_dial_index)
        logger = do_learning(model, train_iter, val_iter, iterations=iterations, task=per)
        fine_tune.append(logger)
        model.load_state_dict({ name: weights_original[name] for name in weights_original })

if config.fix_dialnum_train:
    config.save_path = config.save_path+'_fix_dialnum_'+str(config.k_shot)+'_'
pickle.dump( [fine_tune,iterations], open( config.save_path+'evaluation.p', "wb" ) )
measure = ["LOSS","PPL","Entl_b","Bleu_b"]
temp = {m: [[] for i in list(range(0, 26))] for m in measure}
for expe in fine_tune:
    for idx_measure,m in enumerate(measure):
        for j,i in enumerate(list(range(0, 26))):
            temp[m][j].append(expe[str(i)][idx_measure])  ## position 1 is ppl

fig = plt.figure(figsize=(20,80))

log = {}
for id_mes, m in enumerate(measure):
    ax1 = fig.add_subplot(331 + id_mes)
    x = range(len(list(np.array(temp[m]).mean(axis=1))))
    y = np.array(temp[m]).mean(axis=1)
    e = np.array(temp[m]).std(axis=1)
    plt.errorbar(x, y, e)
    plt.title(m)
    log[m] = y

plt.savefig(config.save_path+'epoch_vs_ppl.pdf')
print("----------------------------------------------------------------------")
print("epoch\tloss\tPeplexity\tEntl_b\tBleu_b\n")
for j,i in enumerate(list(range(0, 26))):
    print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(i,log['LOSS'][j],math.exp(log['LOSS'][j]),log['Entl_b'][j],log['Bleu_b'][j]))
print("----------------------------------------------------------------------")
with open(config.save_path+'result.txt', 'w', encoding='utf-8') as f:
    f.write("epoch\tloss\tPeplexity\tEntl_b\tBleu_b\n")
    for j,i in enumerate(list(range(0, 26))):
        f.write("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(i,log['LOSS'][j],math.exp(log['LOSS'][j]),log['Entl_b'][j],log['Bleu_b'][j]))

