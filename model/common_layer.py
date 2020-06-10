### MOSTO OF IT TAKEN FROM https://github.com/kolloldas/torchnlp
## MINOR CHANGES
import torch
#torch.cuda.set_device(1)
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as I
import numpy as np
import math
from collections import Counter
import subprocess
from utils import config
from utils.metric import rouge, moses_multi_bleu, _prec_recall_f1_score, entailtment_score
from utils.beam_omt import Translator
import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=1)
from utils.load_bert import bert_model
def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return bleu(stats)


def gen_embeddings(vocab):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    embeddings = np.random.randn(vocab.n_words, config.emb_dim) * 0.01 
    print('Embeddings: %d x %d' % (vocab.n_words, config.emb_dim))
    if config.emb_file is not None:
        print('Loading embedding file: %s' % config.emb_file)
        pre_trained = 0
        for line in open(config.emb_file).readlines():
            sp = line.split()
            if(len(sp) == config.emb_dim + 1):
                if sp[0] in vocab.word2index:
                    pre_trained += 1
                    embeddings[vocab.word2index[sp[0]]] = [float(x) for x in sp[1:]]
            else:
                print(sp[0])
        print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / vocab.n_words))
    return embeddings


def share_embedding(vocab, pretrain=True):
    embedding = nn.Embedding(vocab.n_words, config.emb_dim)
    if(pretrain):
        pre_embedding = gen_embeddings(vocab)
        embedding.weight.data.copy_(torch.FloatTensor(pre_embedding))
        embedding.weight.data.requires_grad = True
    return embedding

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(config.PAD_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_input_from_batch(batch):
    enc_batch = batch["input_batch"].transpose(0,1)
    enc_lens = batch["input_lengths"]
    persona_index = batch["persona_index"]
    batch_size, max_enc_len = enc_batch.size()
    assert enc_lens.size(0) == batch_size

    enc_padding_mask = sequence_mask(enc_lens, max_len=max_enc_len).float()

    extra_zeros = None
    enc_batch_extend_vocab = None

    if config.pointer_gen:
        enc_batch_extend_vocab = batch["input_ext_vocab_batch"].transpose(0,1)
        # max_art_oovs is the max over all the article oov list in the batch
        if batch["max_art_oovs"] > 0:
            extra_zeros = torch.zeros((batch_size, batch["max_art_oovs"]))

    c_t_1 = torch.zeros((batch_size, 2 * config.hidden_dim))

    coverage = None
    if config.is_coverage:
        coverage = torch.zeros(enc_batch.size())

    if config.USE_CUDA:
        if enc_batch_extend_vocab is not None:
                enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
        if extra_zeros is not None:
            extra_zeros = extra_zeros.cuda()
        c_t_1 = c_t_1.cuda()

        if coverage is not None:
            coverage = coverage.cuda()

    return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage, persona_index

def get_output_from_batch(batch):

    dec_batch = batch["target_batch"].transpose(0,1)

    if(config.pointer_gen):
        target_batch = batch["target_ext_vocab_batch"].transpose(0,1)
    else:
        target_batch = dec_batch
        
    dec_lens_var = batch["target_lengths"]
    persona_index = batch["persona_index"]
    max_dec_len = max(dec_lens_var)

    assert max_dec_len == target_batch.size(1)

    dec_padding_mask = sequence_mask(dec_lens_var, max_len=max_dec_len).float()

    return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch, persona_index

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = seq_range_expand
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                        .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def print_all(dial,ref,hyp_b,max_print):
    for i in range(len(ref)):
        print(pp.pformat(dial[i]))
        print("Beam: {}".format(hyp_b[i]))
        print("Ref:{}".format(ref[i]))
        print("----------------------------------------------------------------------")
        print("----------------------------------------------------------------------")
        if(i>max_print):break

def log_all(dial,ref,hyp_b,per,log_file):
    f = open(log_file, "a")
    for i in range(len(ref)):
        f.write(pp.pformat(dial[i]))
        f.write("\n")
        f.write(pp.pformat(per[i]))
        f.write("\n")
        f.write("Beam: {}".format(hyp_b[i]))
        f.write("\n")
        f.write("Ref:{}".format(ref[i]))
        f.write("\n")
        f.write("----------------------------------------------------------------------")
        f.write("\n")
    f.close()

bert = bert_model()

def evaluate(model, data, model_name='trs', ty='valid', writer=None, n_iter=0, ty_eval="before", verbose=False, log=False, result_file="results/results_our.txt", ref_file="results/ref_our.txt", case_file="results/case_our.txt"):
    if log:
        f1 = open(result_file, "a")
        f2 = open(ref_file, "a")
    dial,ref, hyp_b, per= [],[],[], []
    t = Translator(model, model.vocab)

    l = []
    p = []
    ent_b = []
    
    pbar = tqdm(enumerate(data),total=len(data))
    for j, batch in pbar:
        #print(len(batch["input_batch"]))
        #print(len(batch["target_batch"]))
        loss, ppl, _ = model.train_one_batch(batch, train=False)
        l.append(loss)
        p.append(ppl)
        if((j<3 and ty != "test") or ty =="test"): 

            sent_b, _ = t.translate_batch(batch)

            for i in range(len(batch["target_txt"])):
                new_words = []
                for w in sent_b[i][0]:
                    if w==config.EOS_idx:
                        break
                    new_words.append(w)
                    if len(new_words)>2 and (new_words[-2]==w):
                        new_words.pop()
                
                sent_beam_search = ' '.join([model.vocab.index2word[idx] for idx in new_words])
                hyp_b.append(sent_beam_search)
                if log:
                    f1.write(sent_beam_search)
                    f1.write("\n")
                ref.append(batch["target_txt"][i])
                if log:
                    f2.write(batch["target_txt"][i])
                    f2.write("\n")
                dial.append(batch['input_txt'][i])
                per.append(batch['persona_txt'][i])
                #ent_b.append(0)
                ent_b.append(bert.predict_label([sent_beam_search for _ in range(len(batch['persona_txt'][i]))], batch['persona_txt'][i]))

        pbar.set_description("loss:{:.4f} ppl:{:.1f}".format(np.mean(l),np.mean(p)))
        if(j>4 and ty=="train"): break
    loss = np.mean(l)
    ppl = np.mean(p)
    ent_b = np.mean(ent_b)
    bleu_score_b = moses_multi_bleu(np.array(hyp_b), np.array(ref), lowercase=True)
    #bleu_score_b = moses_multi_bleu(np.array(hyp_b), np.array(hyp_b), lowercase=True)
    #bleu_score_b = get_bleu(np.array(hyp_b), np.array(ref))
    if log:
        f1.close()
        f2.close()
 
    if(verbose):
        print("----------------------------------------------------------------------")
        print("----------------------------------------------------------------------")
        print_all(dial,ref,hyp_b,max_print=3 if ty != "test" else 100000000 )
        print("EVAL\tLoss\tPeplexity\tEntl_b\tBleu_b")
        print("{}\t{:.4f}\t{:.4f}\t{:.2f}\t{:.2f}".format(ty,loss,ppl,ent_b,bleu_score_b))
    if log:
        log_all(dial,ref,hyp_b, per, case_file)
    return loss,ppl,ent_b,bleu_score_b

