import matplotlib
matplotlib.use('Agg')
from utils.data_reader import Personas
from model.seq2spg import Seq2SPG
from model.common_layer import NoamOpt, evaluate
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
import pickle
from tensorboardX import SummaryWriter

def do_learning_fix_step(model, train_iter, val_iter, iterations, test=False, mode="pretrain", task=0):
    val_p = []
    val_p_list = []
    val_loss = 0
    for _ ,_ in enumerate(range(iterations)):
        
        for d in train_iter:
            t_loss, t_ppl, _ = model.train_one_batch(d, mode=mode, task=task)
        if test:
            _, test_ppl = do_evaluation(model, val_iter)
            val_p_list.append(test_ppl)
    #weight = deepcopy(model.state.dict())

    if test:
        return val_p_list
    else:
        for d in val_iter:
            _, t_ppl, t_loss = model.train_one_batch(d,train= False)
            val_loss+=t_loss
            val_p.append(t_ppl)
        return val_loss, np.mean(val_p)

def do_evaluation(model, test_iter):
    p, l = [],[]
    for batch in test_iter:
        loss, ppl, _ = model.train_one_batch(batch, train=False)
        l.append(loss)
        p.append(ppl)
    return np.mean(l), np.mean(p)

#=================================main=================================

p = Personas()
writer = SummaryWriter(log_dir=config.save_path)
# Build model, optimizer, and set states
if not (config.load_frompretrain=='None'): 
    meta_net = Seq2SPG(p.vocab,model_file_path=config.load_frompretrain,is_eval=False)
else: 
    meta_net = Seq2SPG(p.vocab)
if config.meta_optimizer=='sgd':
    meta_optimizer = torch.optim.SGD(meta_net.parameters(), lr=config.meta_lr)
elif config.meta_optimizer=='adam':
    meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=config.meta_lr)
elif config.meta_optimizer=='noam':
    meta_optimizer = NoamOpt(config.hidden_dim, 1, 4000, torch.optim.Adam(meta_net.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
else:
    raise ValueError

meta_batch_size = config.meta_batch_size
tasks = p.get_personas('train')
steps = (len(tasks) // meta_batch_size) + int(len(tasks) % meta_batch_size !=0)


# meta early stop
patience = 10
if config.fix_dialnum_train:
    patience = 100
best_loss = 10000000
stop_count = 0
for meta_iteration in range(config.epochs):
    ## save original weights to make the update
    train_loss_before = []
    train_loss_meta = []
    if meta_iteration < 10:
        m = "pretrain"
    elif meta_iteration == 10:
        m = "select"
    else:
        m = "selective_training"
    print(m)
    shuffle(tasks)
    for k in range(steps):
        st = k * meta_batch_size
        ed = st + meta_batch_size
        if ed > len(tasks):
            ed = len(tasks)
        batch_loss=0
        weights_original = deepcopy(meta_net.state_dict())
        for i in range(st, ed):
            per = tasks[i]
            train_iter, val_iter = p.get_data_loader(persona=per,batch_size=config.batch_size, split='train')
            v_loss, v_ppl = do_evaluation(meta_net, val_iter)
            train_loss_before.append(math.exp(v_loss))
            # Update fast nets 
            if m != "select":
                val_loss, v_ppl = do_learning_fix_step(meta_net, train_iter, val_iter, iterations=config.meta_iteration, mode=m, task=per)
            else:
                val_loss, v_ppl = do_learning_fix_step(meta_net, train_iter, val_iter, iterations=5, mode=m, task=per)
            train_loss_meta.append(math.exp(val_loss.item()))
            batch_loss+=val_loss
            meta_net.load_state_dict({ name: weights_original[name] for name in weights_original })
        writer.add_scalars('loss_before', {'train_loss_before': np.mean(train_loss_before)}, meta_iteration)
        writer.add_scalars('loss_meta', {'train_loss_meta': np.mean(train_loss_meta)}, meta_iteration)
        # meta Update
        if(config.meta_optimizer=='noam'):
            meta_optimizer.optimizer.zero_grad()
        else:
            meta_optimizer.zero_grad()
        batch_loss/=meta_batch_size
        if m != "select":
            batch_loss.backward()
            # clip gradient
            nn.utils.clip_grad_norm_(meta_net.parameters(), config.max_grad_norm)
            meta_optimizer.step()

    print('Meta_iteration:', meta_iteration)
    val_loss_before = []
    val_loss_meta = []
    weights_original = deepcopy(meta_net.state_dict())
    for idx ,per in enumerate(p.get_personas('valid')):
        #num_of_dialog = p.get_num_of_dialog(persona=per, split='valid')
        #for dial_i in range(num_of_dialog):
        if config.fix_dialnum_train:
            train_iter, val_iter = p.get_balanced_loader(persona=per,batch_size=config.batch_size, split='valid', fold=0)

        else:
            train_iter, val_iter = p.get_data_loader(persona=per,batch_size=config.batch_size, split='valid', fold=0)
        # zero shot result
        loss, ppl = do_evaluation(meta_net, val_iter)
        val_loss_before.append(math.exp(loss))
        # meta tuning
        val_loss, val_ppl = do_learning_fix_step(meta_net, train_iter, val_iter, iterations=config.meta_iteration)
        val_loss_meta.append(math.exp(val_loss.item()))
        # updated result

        meta_net.load_state_dict({ name: weights_original[name] for name in weights_original })

    writer.add_scalars('loss_before', {'val_loss_before': np.mean(val_loss_before)}, meta_iteration)
    writer.add_scalars('loss_meta', {'val_loss_meta': np.mean(val_loss_meta)}, meta_iteration)
    #check early stop
    if np.mean(val_loss_meta)< best_loss:
        best_loss = np.mean(val_loss_meta)
        stop_count = 0
        meta_net.save_model(best_loss,meta_iteration,0.0,0.0,0.0,1.1)
    else:
        stop_count+=1
        if stop_count>patience:
            break
