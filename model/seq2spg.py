import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from model.common_layer import share_embedding, LabelSmoothing, NoamOpt, get_input_from_batch, get_output_from_batch
from utils import config
import random
from numpy import random
import os
import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=1)
import os
import time

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

class SoftDotAttention(nn.Module):
    """Soft Dot Attention.
    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context):
        """Propogate input through the network.
        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn

class LSTMAttentionDot(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, batch_first=True):
        """Initialize params."""
        super(LSTMAttentionDot, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

        self.attention_layer = SoftDotAttention(hidden_size)

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the network."""
        def recurrence(input, hidden):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + \
                self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim
            h_tilde, alpha = self.attention_layer(hy, ctx.transpose(0, 1))

            return h_tilde, cy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)
            #output.append(isinstance(hidden, tuple) and hidden[0] or hidden)
            #output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x, attn_dist=None, enc_batch_extend_vocab=None, extra_zeros=None, temp=1, beam_search=False):
        logit = self.proj(x)
        return F.log_softmax(logit,dim=-1)

class MLP(nn.Module):
    "The private part in Seq2SPG"
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size[0]), nn.Linear(hidden_size[0], hidden_size[1]),
                                    nn.Linear(hidden_size[1], hidden_size[2]), 
                                    nn.Linear(hidden_size[2], output_size)])
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        for i, linear in enumerate(self.layers):
            if i < 3:
                x = self.relu(linear(x))
            else:
                out = self.tanh(linear(x))
        return out


def make_hook(hook):
    def hooker(grad):
        return grad * Variable(hook, requires_grad=False)
    return hooker

class Seq2SPG(nn.Module):
    def __init__(self, vocab, model_file_path=None, is_eval=False, load_optim=False):
        super(Seq2SPG, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab,config.preptrained)
        self.encoder = nn.LSTM(config.emb_dim, config.hidden_dim, config.hop, bidirectional=False, batch_first=True,
                               dropout=0.2)
        self.encoder2decoder = nn.Linear(
            config.hidden_dim,
            config.hidden_dim)
        self.decoder = LSTMAttentionDot(config.emb_dim, config.hidden_dim, batch_first=True)
        self.memory = MLP(config.hidden_dim + config.emb_dim, [config.private_dim1, config.private_dim2, config.private_dim3], config.hidden_dim)
        self.dec_gate = nn.Linear(config.hidden_dim, 2 * config.hidden_dim)
        self.mem_gate = nn.Linear(config.hidden_dim, 2 * config.hidden_dim)
        self.generator = Generator(config.hidden_dim,self.vocab_size)
        self.hooks = {} #Save the model structure of each task as masks of the parameters
        if config.weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.weight
        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if (config.label_smoothing):
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)
        if is_eval:
            self.encoder = self.encoder.eval()
            self.encoder2decoder = self.encoder2decoder.eval()
            self.decoder = self.decoder.eval()
            self.generator = self.generator.eval()
            self.embedding = self.embedding.eval()
            self.memory = self.memory.eval()
            self.dec_gate = self.dec_gate.eval()
            self.mem_gate = self.mem_gate.eval()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if(config.noam):
            self.optimizer = NoamOpt(config.hidden_dim, 1, 4000, torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        if config.use_sgd:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=config.lr)
        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            print("LOSS",state['current_loss'])
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.encoder2decoder.load_state_dict(state['encoder2decoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'])
            self.generator.load_state_dict(state['generator_dict'])
            self.embedding.load_state_dict(state['embedding_dict'])
            self.memory.load_state_dict(state['memory_dict'])
            self.dec_gate.load_state_dict(state['dec_gate_dict'])
            self.mem_gate.load_state_dict(state['mem_gate_dict'])
            if (load_optim):
                self.optimizer.load_state_dict(state['optimizer'])

        if (config.USE_CUDA):
            self.encoder = self.encoder.cuda()
            self.encoder2decoder = self.encoder2decoder.cuda()
            self.decoder = self.decoder.cuda()
            self.generator = self.generator.cuda()
            self.criterion = self.criterion.cuda()
            self.embedding = self.embedding.cuda()
            self.memory = self.memory.cuda()
            self.dec_gate = self.dec_gate.cuda()
            self.mem_gate = self.mem_gate.cuda()
        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""
    
    def save_model(self, running_avg_ppl, iter, f1_g,f1_b,ent_g,ent_b,log=False, d="tmaml_sim_model"):
        state = {
            'iter': iter,
            'encoder_state_dict': self.encoder.state_dict(),
            'encoder2decoder_state_dict': self.encoder2decoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'generator_dict': self.generator.state_dict(),
            'embedding_dict': self.embedding.state_dict(),
            'memory_dict': self.memory.state_dict(),
            'dec_gate_dict': self.dec_gate.state_dict(),
            'mem_gate_dict': self.mem_gate.state_dict(),
            #'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_ppl
        }
        if log:
            model_save_path = os.path.join(d, 'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(iter,running_avg_ppl,f1_g,f1_b,ent_g,ent_b) )
        else:
            model_save_path = os.path.join(self.model_dir, 'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(iter,running_avg_ppl,f1_g,f1_b,ent_g,ent_b) )
        self.best_path = model_save_path
        torch.save(state, model_save_path)
    
    def get_state(self, batch):
        """Get cell states and hidden states for LSTM"""
        batch_size = batch.size(0) \
            if self.encoder.batch_first else batch.size(1)
        h0_encoder = Variable(torch.zeros(
            self.encoder.num_layers,
            batch_size,
            config.hidden_dim
        ), requires_grad=False)
        c0_encoder = Variable(torch.zeros(
            self.encoder.num_layers,
            batch_size,
            config.hidden_dim
        ), requires_grad=False)

        return h0_encoder.cuda(), c0_encoder.cuda()
    
    def compute_hooks(self, task):
        """Compute the masks of the private module"""
        current_layer = 3
        out_mask = torch.ones(self.memory.output_size)
        self.hooks[task] = {}
        self.hooks[task]["w_hooks"] = {}
        self.hooks[task]["b_hooks"] = {}
        while(current_layer >= 0):
            connections = self.memory.layers[current_layer].weight.data
            output_size, input_size = connections.shape
            mask = connections.abs() > 0.05
            in_mask = torch.zeros(input_size)
            for index, line in enumerate(mask):
                if(out_mask[index] == 1):
                    torch.max(in_mask, (line.cpu() != 0).float(), out=in_mask)
            if (config.USE_CUDA):
                self.hooks[task]["b_hooks"][current_layer] = out_mask.cuda()
                self.hooks[task]["w_hooks"][current_layer] = torch.mm(out_mask.unsqueeze(1), in_mask.unsqueeze(0)).cuda()
            else:
                self.hooks[task]["b_hooks"][current_layer] = out_mask
                self.hooks[task]["w_hooks"][current_layer] = torch.mm(out_mask.unsqueeze(1), in_mask.unsqueeze(0))
            out_mask = in_mask
            current_layer -= 1
            
    def register_hooks(self, task):
        if "hook_handles" not in self.hooks[task]:
            self.hooks[task]["hook_handles"] = []
        for i, l in enumerate(self.memory.layers):
            self.hooks[task]["hook_handles"].append(l.bias.register_hook(make_hook(self.hooks[task]["b_hooks"][i])))
            self.hooks[task]["hook_handles"].append(l.weight.register_hook(make_hook(self.hooks[task]["w_hooks"][i])))
    
    def unhook(self, task):
        for handle in self.hooks[task]["hook_handles"]:
            handle.remove()
        self.hooks[task]["hook_handles"] = []
                        
    def train_one_batch(self, batch, train=True, mode="pretrain", task=0):
        enc_batch, _, enc_lens, enc_batch_extend_vocab, extra_zeros, _, _, _ = get_input_from_batch(batch)
        dec_batch, _, _, _, _, _ = get_output_from_batch(batch)
        
        if(config.noam):
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        ## Encode
        self.h0_encoder, self.c0_encoder = self.get_state(enc_batch)
        src_h, (src_h_t, src_c_t) = self.encoder(
            self.embedding(enc_batch), (self.h0_encoder, self.c0_encoder))
        h_t = src_h_t[-1]
        c_t = src_c_t[-1]

        # Decode
        decoder_init_state = nn.Tanh()(self.encoder2decoder(h_t))
        
        sos_token = torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1)
        if config.USE_CUDA: sos_token = sos_token.cuda()
        dec_batch_shift = torch.cat((sos_token,dec_batch[:, :-1]),1)
        target_embedding = self.embedding(dec_batch_shift)
        ctx = src_h.transpose(0, 1)
        trg_h, (_, _) = self.decoder(
            target_embedding,
            (decoder_init_state, c_t),
            ctx    
        )
        
        #Memory
        mem_h_input = torch.cat((decoder_init_state.unsqueeze(1), trg_h[:,0:-1,:]), 1)
        mem_input = torch.cat((target_embedding, mem_h_input), 2)
        mem_output = self.memory(mem_input)
        
        #Combine
        gates = self.dec_gate(trg_h) + self.mem_gate(mem_output)
        decoder_gate, memory_gate = gates.chunk(2, 2)
        decoder_gate = F.sigmoid(decoder_gate)
        memory_gate = F.sigmoid(memory_gate)
        pre_logit = F.tanh(decoder_gate * trg_h + memory_gate * mem_output)
        logit = self.generator(pre_logit)
        
        if mode == "pretrain":
            loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
            if train:
                loss.backward()
                self.optimizer.step()
            if(config.label_smoothing): 
                loss = self.criterion_ppl(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))        
            return loss.item(), math.exp(min(loss.item(), 100)), loss
        
        elif mode == "select":
            loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
            if(train):
                l1_loss = 0.0
                for p in self.memory.parameters():
                    l1_loss += torch.sum(torch.abs(p))
                loss += 0.0005 * l1_loss
                loss.backward()
                self.optimizer.step()
                self.compute_hooks(task)
            if(config.label_smoothing): 
                loss = self.criterion_ppl(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))  
            return loss.item(), math.exp(min(loss.item(), 100)), loss
        
        else:
            loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
            if(train):
                self.register_hooks(task)
                loss.backward()
                self.optimizer.step()
                self.unhook(task)
            if(config.label_smoothing): 
                loss = self.criterion_ppl(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))  
            return loss.item(), math.exp(min(loss.item(), 100)), loss
