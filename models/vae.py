import torch.nn.functional as F
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import BertModel, BertTokenizer, BertConfig
from abc import ABC, abstractmethod
from sumeval.metrics.rouge import RougeCalculator
from sumeval.metrics.bleu import BLEUCalculator
from tqdm import tqdm

import numpy as np
import itertools
import torch
import torch.nn as nn
import math
import os

rng = torch.quasirandom.SobolEngine(1, scramble=True)

# GPT
EOS_ID = 50256
# Bert
SEP_ID = 102
PAD_ID= 0
# T5
PAD_ID_T5 = 0
SEP_ID_T5 = 1

def calc_bleu(original_sentences, predict_sentences, default= "None"):
    bleu = 0.0
    num_sample = len(original_sentences)
    for original, predict in zip(original_sentences, predict_sentences):
        # Remove padding
        original, predict = original.replace("<PAD>", "").strip(), predict.replace("<PAD>", "").strip()
        if original == "": original = default
        if predict == "": predict = default
        bleu += BLEUCalculator(lang="en").bleu(summary=predict, references=original)
    return bleu/num_sample

def calc_rouge(original_sentences, predict_sentences, default= "None"):
    rouge_1 = 0.0; rouge_2 = 0.0; rouge_l = 0.0
    num_sample = len(original_sentences)
    for original, predict in zip(original_sentences, predict_sentences):
        # Remove padding
        original, predict = original.replace("<PAD>", "").strip(), predict.replace("<PAD>", "").strip()
        if original == "": original = default
        if predict == "": predict = default
        rouge = RougeCalculator(stopwords=True, lang="en")
        r1 = rouge.rouge_1(summary=predict, references=original)
        r2 = rouge.rouge_2(summary=predict, references=original)
        rl = rouge.rouge_l(summary=predict, references=original)
        rouge_1 += r1
        rouge_2 += r2
        rouge_l += rl
    return rouge_1/num_sample, rouge_2/num_sample, rouge_l/num_sample


def generate_sequence(model, temperature=1, top_k=1, top_p = 1.0, length=20, sample=False, past=None, device='cuda'):
    output = past[0][0].new_zeros([past[0][0].size(0),0])
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    bsz = past[0][0].size(0)
    prev = torch.Tensor([EOS_ID]*bsz).long().cuda().unsqueeze(1)
    output = torch.cat((output, prev), dim=1)
    for i in range(length):
        prev, probs, past = generate_next_token(model, prev, temperature=temperature, top_k=top_k, top_p=top_p, sample=sample, past=past)
        output = torch.cat((output, prev), dim=1)
    return output

def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

def generate_next_token(model_gpt, prev, temperature=1, top_k = 0, top_p=1.0, sample=False, past=None):

    with torch.no_grad():
        #pdb.set_trace()
        gpt_output = model_gpt.transformer(prev, position_ids=None, token_type_ids=None, past_key_values=past)
        hidden_states, past = gpt_output['last_hidden_state'], gpt_output['past_key_values']
        logits = model_gpt.lm_head(hidden_states)
        logits = logits[:, -1, :] / temperature
#        if top_p < 1.0:
#            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
#        else:
        logits = top_k_logits(logits, k=top_k)
        probs = F.softmax(logits, dim=-1)


        if sample:
            prev = torch.multinomial(probs, num_samples=1)
            return prev, probs[0][prev], past
        else:
            probs_sel, prev = torch.topk(probs, k=top_k, dim=-1)
            return prev, probs_sel, past


def prepare_for_bleu(sentence, tokenizer, skip_special_tokens = False):
    sent=[]
    tokenizer_name = tokenizer.__class__.__name__
    if skip_special_tokens:
        end_of_sentence = {'BertTokenizer': [], 'GPT2Tokenizer': [], 'T5Tokenizer': []}
    else:
        end_of_sentence = {'BertTokenizer': [SEP_ID, PAD_ID], 'GPT2Tokenizer': [EOS_ID], 'T5Tokenizer': [SEP_ID_T5, PAD_ID_T5],}
    for s in sentence[1:]:
        if s not in end_of_sentence[tokenizer_name]:
            sent.append(s)
        else:
            break
    return sent

def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return (1.0 - x)/(1.0 - warmup)

SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
}

class Adamax(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix (and no ).
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay_rate: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 betas=(0.9, 0.999), eps=1e-6, weight_decay_rate=0.01,
                 max_grad_norm=1.0):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        betas=betas, eps=eps, weight_decay_rate=weight_decay_rate,
                        max_grad_norm=max_grad_norm)
        super(Adamax, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def to(self, device):
        """ Move the optimizer state to a specified device"""
        for state in self.state.values():
            state['exp_avg'].to(device)
            state['exp_avg_sq'].to(device)

    def initialize_step(self, initial_step):
        """Initialize state with a defined step (but we don't have stored averaged).
        Arguments:
            initial_step (int): Initial step number.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # State initialization
                state['step'] = initial_step
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_inf'] = torch.zeros_like(p.data)

                exp_avg, exp_inf = state['exp_avg'], state['exp_inf']
                beta1, beta2 = group['betas']
                eps = group['eps']
                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # Update the exponentially weighted infinity norm.
                norm_buf = torch.cat([
                    exp_inf.mul_(beta2).unsqueeze(0),
                    grad.abs().add_(eps).unsqueeze_(0)
                ], 0)
                torch.max(norm_buf, 0, keepdim=False, out=(exp_inf, exp_inf.new().long()))
                update = exp_avg / (exp_inf + eps)

                if group['weight_decay_rate'] > 0.0:
                    update += group['weight_decay_rate'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']


                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want ot decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay_rate'] > 0.0:
                    update += group['weight_decay_rate'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

        return loss

class BertEncoder(nn.Module):
    def __init__(self, enc_model, latent_size, num_feature, load_enc=False, model_enc=None, hidden_size=None):
        super().__init__()
        self.name = enc_model
        if not hidden_size: hidden_size = latent_size
        self.model_enc = model_enc
        self.tokenizer = BertTokenizer.from_pretrained(self.name)
        self.num_feature = num_feature
        if model_enc is None:
            if load_enc:
                # load pretrained bert 
                self.model_enc = BertModel.from_pretrained(self.name)
                latent_size = self.model_enc.config.hidden_size
            else: 
                # from scratch
                config = BertConfig.from_pretrained(self.name)
                config.hidden_size = hidden_size          
                self.model_enc = BertModel(config)  

        self.model_size = sum(t.numel() for t in self.model_enc.parameters())
    
    def forward(self, input_ids_bert=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ne(input_ids_bert, 0)
        encoded_output = self.model_enc(input_ids_bert, attention_mask)
        hidden_state = encoded_output['last_hidden_state'][:, :self.num_feature, :]
        return hidden_state.permute(0, 2, 1)  # bsz x latent x feature num

    def named_parameters(self):
        return self.model_enc.named_parameters()

    def save(self, output_dir, prefix):
        torch.save(self.model_enc.state_dict(), os.path.join(output_dir, prefix + '-BERT.pkl'))

    @staticmethod
    def from_pretrained(encoder, input_dir, prefix, name='bert-base-uncased'):
        model_enc = BertModel.from_pretrained(name, state_dict=torch.load(os.path.join(input_dir, prefix + '-BERT.pkl'), map_location='cpu'))
        encoder['model_args']['model_enc'] = model_enc
        return encoder

class GPT2Decoder(nn.Module):
    def __init__(self, dec_model, latent_size, num_feature, sentence_len, load_dec, n_head, share_gpts, model_gpt=None, model_pre=None):
        super().__init__()
        self.name = dec_model
        self.model_gpt = model_gpt
        self.model_pre = model_pre
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.name) 
        self.num_feature = num_feature
        self.max_len = sentence_len
        if model_gpt is None:
            if load_dec:
                # load pretrained bert 
                self.model_gpt = GPT2LMHeadModel.from_pretrained(self.name)
                config = GPT2Config.from_pretrained(self.name)
                if config.n_embd == latent_size:
                    self.embd_adapter = nn.Identity()
                else:
                    self.embd_adapter = nn.Linear(latent_size, config.n_embd) 
            else: 
                # from scratch
                config = GPT2Config.from_pretrained(self.name)
                config.n_embd = latent_size
                config.n_head = n_head              
                self.model_gpt = GPT2LMHeadModel(config) 
        if model_pre is None:
            if share_gpts:
                self.model_pre = self.model_gpt
                self.model_size = sum(t.numel() for t in self.model_gpt.parameters()) 
                return
            else:
                config = GPT2Config.from_pretrained(self.name)
                config.n_embd = latent_size           
                self.model_pre = GPT2LMHeadModel(config)
        self.model_size = sum(t.numel() for t in self.model_gpt.parameters()) + sum(t.numel() for t in self.model_pre.parameters())

    def forward(self, hidden_state, input_ids_dec=None, lm_labels=None):
        # TODO: if input length is shorter than num_feature, there will be issue with BERT-DECONV 
        # assert(hidden_state.shape[2] == self.num_feature)  
        # bsz x latent x feature num

        hidden_state = hidden_state.permute(0, 2, 1).contiguous()
        hidden_state = self.embd_adapter(hidden_state)
        hidden_state = hidden_state.permute(0, 2, 1).contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        context = self.model_pre(inputs_embeds=hidden_state.permute(0, 2, 1))[-1] #list
        lm_logits = self.model_gpt(input_ids=input_ids_dec, past_key_values=context)[0]
        bsz, seq_len, vocab_size = lm_logits.size()
        loss = loss_fct(lm_logits.view(-1, vocab_size), lm_labels.view(-1))
        loss = loss.view(bsz, seq_len)
        total = torch.ne(lm_labels, -1).float().sum()
        loss = torch.sum(loss) / total
        correct = (lm_logits.max(dim=-1)[1] == lm_labels).sum()

        # if hidden_state is None:
        #     correct, total = correct.item(), total.item()
        return loss, correct, total, hidden_state

    # do  we ever call save? 
    def save(self, output_dir, prefix):
        torch.save(self.model_gpt.state_dict(), os.path.join(output_dir, prefix+'-GPT2.pkl'))
        torch.save(self.model_pre.state_dict(), os.path.join(output_dir, prefix+'-PRE.pkl'))
    
    def from_pretrained(decoder, input_dir, prefix, name = 'gpt'):
        model_gpt = GPT2LMHeadModel.from_pretrained(name, state_dict=torch.load(os.path.join(input_dir, prefix+'-GPT2.pkl'), map_location='cpu'))
        model_pre = GPT2LMHeadModel.from_pretrained(name, state_dict=torch.load(os.path.join(input_dir, prefix+'-PRE.pkl'), map_location='cpu'))
        decoder['model_args']['model_gpt'] = model_gpt
        decoder['model_args']['model_pre'] = model_pre
        return decoder

    def named_parameters(self):
        return list(self.model_pre.named_parameters()) + list(self.model_gpt.named_parameters())

    def generate_from(self, hidden_states, sample=False, beam_width = -1, top_k = 1, skip_special_tokens = False):
        hidden_states = hidden_states.permute(0, 2, 1).contiguous()
        hidden_states = self.embd_adapter(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1).contiguous()
        with torch.no_grad():
            if beam_width == -1:
                #greedy/sample
                hidden_states = hidden_states.permute(0, 2, 1)

                batch_size = 64
                num_batches = (hidden_states.shape[0] - 1)  // batch_size + 1
                batches = [hidden_states[i*batch_size : (i+1)*batch_size] for i in range(num_batches)]
                resps = []
                for b in tqdm(batches):
                    context = self.model_pre(inputs_embeds=b)[-1]
                    out = generate_sequence(self.model_gpt, temperature=1, top_k=top_k, length=self.max_len, sample=sample, past=context, device='cuda')
                    out = out.tolist()
                    gen = [self.tokenizer.decode(prepare_for_bleu(s, self.tokenizer, skip_special_tokens = skip_special_tokens), skip_special_tokens = skip_special_tokens) for s in out]
                    resps.extend([g.encode('ascii','ignore').decode('ascii') for g in gen])
          
        return resps

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, h_noiser, h_noiser_ratio, h_tanh):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.h_noiser = h_noiser
        self.h_noiser_ratio = h_noiser_ratio
        self.h_tanh = h_tanh 

    def forward(self, input_ids_enc, input_ids_dec=None, lm_labels=None):
        hidden_state = self.encoder_mean(input_ids_enc)
        if self.h_noiser == 'normal':
            hidden_state = hidden_state + self.h_noiser_ratio*torch.randn_like(hidden_state)
        elif self.h_noiser == 'none':
            hidden_state = hidden_state
        else:
            NotImplementedError
        if isinstance(self.decoder, GPT2Decoder):
            return self.decoder(hidden_state, input_ids_dec=input_ids_dec, lm_labels=lm_labels)
        else:
            NotImplementedError

    def encoder_mean(self, input_ids_enc, *kargs):
        hidden_state = self.encoder(input_ids_enc, *kargs)
        if self.h_tanh:
            hidden_state = torch.tanh(hidden_state)
        return hidden_state

    def save(self, output_dir, prefix):
        self.encoder.save(output_dir, prefix)
        self.decoder.save(output_dir, prefix)
              
    @classmethod
    def from_pretrained(cls, encoder, decoder, input_dir, args):
        prefix = args.resume_ckpt
        encoder_new = encoder['model_cls'].from_pretrained(encoder, input_dir, prefix, name=args.enc_model)
        decoder_new = decoder['model_cls'].from_pretrained(decoder, input_dir, prefix, name=args.dec_model)
        model = cls(encoder_new, decoder_new, args)
        return model

    def named_enc_parameters(self):
        return self.encoder.named_parameters()

    def named_dec_parameters(self):
        return self.decoder.named_parameters()

    # def named_pretrained_parameters(self):
    #     return list(self.model_enc.named_parameters()) + list(self.model_gpt.named_parameters())

    def generate_from(self, *kargs, **kwargs):
        return self.decoder.generate_from(*kargs, **kwargs)

    # def generate_from_beam(self, *kargs):
    #     return self.decoder.generate_from_beam(*kargs)

    def decode(self, outs, tokenizer = 'enc'):
        resps = []
        self.tokenizers = {'enc': self.encoder.tokenizer, 'dec': self.decoder.tokenizer}
        for out in outs:
            out = out.tolist()
            gen = self.tokenizers[tokenizer].decode(prepare_for_bleu(out, tokenizer=self.tokenizers[tokenizer]))
            resps.append(gen.encode('ascii','ignore').decode('ascii'))
        return resps

    def encode(self, text):
        input_ids = self.encoder.tokenizer.encode(text)
        input_ids = torch.tensor([input_ids]).cuda()
        return self.encoder_mean(input_ids)

class VAE(AutoEncoder):
    def __init__(self, encoder, decoder, h_noiser, h_noiser_ratio, h_tanh):
        super().__init__(encoder, decoder, h_noiser, h_noiser_ratio, h_tanh)
        latent_size = self.encoder.model_enc.config.hidden_size
        self.fc_mean = torch.nn.Linear(latent_size, latent_size)
        self.fc_log_var = torch.nn.Linear(latent_size, latent_size)
        self.beta = h_noiser_ratio

    def forward(self, input_ids_enc, input_ids_dec=None, lm_labels=None):
        # bsz x latent x feature num
        mean = self.encoder_mean(input_ids_enc)
        log_var = self.encoder_log_var(input_ids_enc)
        sampled_h = self.reparameterize(mean, log_var)
        CE, correct, total, _ = self.decoder(sampled_h, input_ids_dec=input_ids_dec, lm_labels=lm_labels)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())  # https://leenashekhar.github.io/2019-01-30-KL-Divergence/
        loss = CE + self.beta * KLD  
        return loss, correct, total, mean

    def encoder_mean(self, input_ids_enc, *kargs):
        hidden_state = self.encoder(input_ids_enc, *kargs)
        shapes = hidden_state.shape
        mean = self.fc_mean(hidden_state.permute(0, 2, 1)).view(-1, shapes[2], shapes[1]).permute(0, 2, 1)
        if self.h_tanh:
            mean = torch.tanh(mean)
        return mean

    def encoder_log_var(self, input_ids_enc, *kargs):
        hidden_state = self.encoder(input_ids_enc, *kargs)
        shapes = hidden_state.shape
        log_var = self.fc_log_var(hidden_state.permute(0, 2, 1)).view(-1, shapes[2], shapes[1]).permute(0, 2, 1) # offset with -5
        return log_var

    def save(self, output_dir, prefix):
        torch.save(self.state_dict(), os.path.join(output_dir, prefix+'-VAE.pkl'))
              
    @classmethod
    def from_pretrained(cls, encoder, decoder, input_dir, args):
        prefix = args.resume_ckpt
        model = cls(encoder, decoder, args =args)
        model.load_state_dict(torch.load(os.path.join(input_dir, prefix+'-VAE.pkl'), map_location='cpu'))
        return model

    def named_enc_parameters(self):
        return itertools.chain(self.encoder.named_parameters(), self.fc_mean.named_parameters(), self.fc_log_var.named_parameters())

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

##################################################
# dataloader & noising tools
##################################################

class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights

from torch.nn.utils.rnn import pad_sequence
class BertNoiser:
    def __init__(self, tokenizer, mlm_prob):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_prob
    def _noise(self, inputs):
        """ Prepare masked tokens for masked language modeling: 80% MASK, 20% random. """

        # We sample a few tokens in each sequence for masked-LM training (with probability mlm_probability defaults to 0.15 in Bert/RoBERTa)
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.long)
        probability = torch.full(inputs.shape, self.mlm_probability)

        special_tokens_mask = [ 1 if x in [self.tokenizer.sep_token_id, self.tokenizer.cls_token_id, 0] else 0 for x in inputs.tolist()]
        probability.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        masked_indices = torch.bernoulli(probability).bool()
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 20% of the time, we replace masked input tokens with random word
        indices_random = masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), inputs.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs

    def noise(self, tensor):
        # TODO implement noise
        def _noise(inputs):
            if not torch.is_tensor(inputs):
                inputs = torch.tensor(inputs, dtype=torch.long)
            return inputs
        noised = []
        keep = torch.ne(tensor, 0)
        for sent, keep in zip(tensor.split(1,dim=0), keep.split(1, dim=0)):
            noised.append(_noise(sent[keep]))
        return pad_sequence(noised, batch_first=True, padding_value=0)
