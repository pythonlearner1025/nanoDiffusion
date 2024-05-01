from models.vae import VAE, BertEncoder, GPT2Decoder, Adamax, BertNoiser, calc_bleu, calc_rouge
from dataloader import FeatureDataset, BucketSampler, DataLoader, get_data
from dataloader import load_winogrande, load_gsm8k, load_tripadvisor
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer
from tqdm import tqdm

import torch
import os

DATA_DIR = 'data'
EPOCH = 10
SAVE_INTERVAL = 1
VAL_INTERVAL = 5
BS = 4
LATENT_CODES = 16
SENTENCE_LEN = 256
SAVE_DIR = 'model'
H_NOISER = "vae"
H_NOISER_RATIO = 0.00001
H_TANH = True
ENC_MODEL = "bert-large-uncased"
DEC_MODEL = "gpt2-medium"
LOAD_DEC = True
LATENT_SIZE = 1024
SHARE_GPTS = True
lr = 0.0005
N_HEADS = 16
WORLD_SZ = 1
GRAD_ACC_STEPS = 1
enc_lr = dec_lr = lr
local_rank = 0

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')    

def preprocess(trainset, testset):
    data = {'train': None, 'test': None}
    for splitname, split in {'train':trainset, 'test':testset}.items():
        data_file = f'data/{dataset}/{splitname}.pt'
        if os.path.exists(data_file):
            features = torch.load(data_file)
        else:
            features = [
                {
                    'bert_ids': bert_tokenizer.encode(text['text'])[:SENTENCE_LEN],
                    'gpt2_ids': gpt_tokenizer.encode(text['text'])[:SENTENCE_LEN],
                    'raw_text': text['text'].strip()
                }
                for text in split
            ]
            os.makedirs(f'data/{dataset}', exist_ok=True)
            torch.save(features, data_file)
        trunc_chunk = []; lens = []; total = len(features) # discard long examples
        for feat in features:
            if len(feat['gpt2_ids'])+2 > gpt_tokenizer.max_len_single_sentence or len(feat['bert_ids']) > bert_tokenizer.max_len_single_sentence:
                continue
            lens.append(len(feat['gpt2_ids']))
            feat['gpt2_ids'] = feat['gpt2_ids'][:SENTENCE_LEN]
            feat['bert_ids'] = feat['bert_ids'][:SENTENCE_LEN]
            trunc_chunk.append(feat)
        featureset = FeatureDataset(trunc_chunk)
        sampler = BucketSampler(lens, 100*BS, BS,
                                droplast=True, shuffle=1)
        loader = DataLoader(featureset, batch_sampler=sampler,
                            num_workers=0, 
                            collate_fn=FeatureDataset.collate)
        data[splitname] = loader
    return data['train'], data['test']

def train_vae(trainset, testset):
    encoder = BertEncoder(ENC_MODEL, LATENT_SIZE, LATENT_CODES)
    decoder = GPT2Decoder(DEC_MODEL, LATENT_SIZE, LATENT_CODES, SENTENCE_LEN, LOAD_DEC, N_HEADS, SHARE_GPTS)
    model = VAE(encoder, decoder, H_NOISER, H_NOISER_RATIO, H_TANH)

    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optim_params = []
    for i,params in enumerate([encoder.named_parameters(), decoder.named_parameters()]):
        optim_params += [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01, 'lr':enc_lr if i == 0 else dec_lr},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr':enc_lr if i == 0 else dec_lr}
        ]

    optimizer = Adamax(optim_params, lr, max_grad_norm=1.0)

    global_step, step = 0, 0
    tr_loss, tr_correct, tr_tokens = 0., 0., 0
    train_dataloader, test_dataloader = preprocess(trainset, testset)
    
    noiser = BertNoiser(bert_tokenizer, 0.3)

    model = model.to(device)
    model.train()
    for epoch in range(EPOCH):
        for batch in tqdm(train_dataloader):
            enc_ids, dec_ids, lm_labels = noiser.noise(batch[0]).to(device), batch[1].to(device), batch[2].to(device)
            assert len(enc_ids) == len(dec_ids)
            loss, correct, ntokens, h = model(enc_ids, input_ids_dec=dec_ids, lm_labels=lm_labels)

            tr_loss += loss.item() * ntokens
            tr_correct += correct.item()
            tr_tokens += ntokens.item()
            step += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            print(loss.item()) 

            # eval
            if (epoch+1) % VAL_INTERVAL == 0:
                model.eval()
                input_sentence, predict_sentence = [], []
                tot_loss, tot_correct, tot_tokens = 0., 0, 0
                for i,batch in tqdm(enumerate(test_dataloader)):
                    enc_ids, dec_ids, lm_labels = noiser.noise(batch[0]).to(device), batch[1].to(device), batch[2].to(device)
                    loss, correct, ntokens, h = model(enc_ids, dec_ids, lm_labels)
                    tot_loss += loss.item() * ntokens
                    tot_correct += correct.item()
                    tot_tokens += ntokens.cpu().item()

                    input_sentence += model.decode(enc_ids)
                    predict_sentence += [x.lower() for x in model.generate_from(h)]
                    print('input sentence:')
                    print(input_sentence[-1])
                    print('predict sentence:')
                    print(predict_sentence[-1])

                if tot_tokens == 0: continue # sometimes no data loaded
                loss = tot_loss/tot_tokens
                ppl = torch.exp(torch.tensor(loss)).item()
                input_sentence = [sent.strip() for sent in input_sentence]
                predict_sentence = [sent.strip() for sent in predict_sentence]

                # bleu supposedly calc conversion error
                bleu = calc_bleu(input_sentence, predict_sentence)
                rouge = calc_rouge(input_sentence, predict_sentence)[2]
                print(f'noise ratio {0.3} bleu score: {bleu}, rouge_l score: {rouge}')
            
                model.train()
                tr_loss, tr_correct, tr_tokens = 0., 0., 0

        if (epoch+1) % SAVE_INTERVAL == 0:
            model.save(SAVE_DIR, f'vae-epoch{epoch}')

from transformers import T5ForConditionalGeneration, T5Tokenizer
from models.diffusion import DDIM
from models.dit import DiT
import torch.nn.functional as F

BS = 16
device = 'cuda'
epochs = 10
eval_inter = 5
save_inter = 1
cond_model_name = 't5-small'
cond_embedding_dim = 131072 if 't5-small' else 131072
load_ckpt = 1
diffusion_steps = T = 300

def get(vae, noiser, batch):
    #print(batch)
    input_ids_bert = batch[0].to(device)
    input_ids_enc = noiser.noise(input_ids_bert)
    input_ids_enc = input_ids_enc
    enc = vae.encoder_mean(input_ids_enc)
    #print(batch[0][0])
    classes = bert_tokenizer.batch_decode(batch[0])
    enc = enc.permute([0,2,1]).unsqueeze(1)
    return enc, classes

def train_conditional_diffusion(trainset, testset):
    
    encoder = BertEncoder(ENC_MODEL, LATENT_SIZE, LATENT_CODES)
    decoder = GPT2Decoder(DEC_MODEL, LATENT_SIZE, LATENT_CODES, SENTENCE_LEN, LOAD_DEC, N_HEADS, SHARE_GPTS)
    vae = VAE(encoder, decoder, H_NOISER, H_NOISER_RATIO, H_TANH)

    if load_ckpt:
        print('loading from ckpt')
        vae_ckpt_path = sorted([f for f in os.listdir(SAVE_DIR) if f.startswith('vae')])[-1]
        vae_ckpt = torch.load(os.path.join(SAVE_DIR, vae_ckpt_path))
        vae.load_state_dict(vae_ckpt)

    vae = vae.to(device) 
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False    

    cond_model = T5ForConditionalGeneration.from_pretrained(cond_model_name).encoder.to(device)
    cond_tokenizer = T5Tokenizer.from_pretrained(cond_model_name)
    noise_predictor =  DiT(num_classes=cond_embedding_dim)
    noiser = BertNoiser(bert_tokenizer, 0.3)

    diffusion = DDIM(noise_predictor, T).to(device) 
    train_data, test_data = preprocess(trainset, testset)
    optimizer = torch.optim.Adam(
        [*noise_predictor.parameters(), *cond_model.parameters()],lr=1e-4
    )

    def sample():
        batch,cond_txt = get(vae, noiser, next(iter(test_data)))
        cond_inputs = cond_tokenizer(
            cond,
            padding='max_length',
            truncation=True,
            max_length=SENTENCE_LEN,
            return_tensors='pt'
        )
        cond_ids = cond_inputs['input_ids'].to(device) 
        shape = batch.shape
        cond = cond_model(input_ids=cond_ids).last_hidden_state.reshape(cond_ids.shape[0], -1) 
        batch, cond = batch.to(device), torch.tensor(cond_ids).to(device)
        samples = diffusion.ddim_sample(cond, shape=shape, steps=30)
        input_sentence, predict_sentence = [], []
        for i in range(len(samples)):
            h = samples[i].permute(0,2,1)
            input_sentence += cond_txt[i]
            predict_sentence += [x.lower() for x in vae.generate_from(h)]
            print("--- 1 sentence ---")
            print(input_sentence[-1])
            print(predict_sentence[-1])
        input_sentence = [sent.strip() for sent in input_sentence]
        predict_sentence = [sent.strip() for sent in predict_sentence]
        bleu = calc_bleu(input_sentence, predict_sentence)
        rouge = calc_rouge(input_sentence, predict_sentence)[2]
        print(f'blue score {bleu}, rouge score {rouge}')

    cond_model.train()
    diffusion.train()
    noise_predictor.train()

    for epoch in range(epochs):
        for i, batch in enumerate(train_data):
            batch, cond = get(vae, noiser, batch)

            cond_inputs = cond_tokenizer(
                cond,
                padding='max_length',
                truncation=True,
                max_length=SENTENCE_LEN,
                return_tensors='pt'
            )
            
            cond_ids = cond_inputs['input_ids'].to(device)
            #cond_mask = cond_inputs['attention_mask'].to(device)

            batch, cond = batch.to(device), torch.tensor(cond_ids).to(device)
            
            cond = cond_model(input_ids=cond_ids).last_hidden_state.reshape(cond_ids.shape[0], -1)

            t = torch.randint(0, T, (batch.shape[0],), device=device, dtype=torch.long)

            true_noise = torch.randn_like(batch)
            x_t = diffusion.q_sample(batch, t, true_noise)

            noise_pred = noise_predictor(x_t, t, cond)

            loss = F.smooth_l1_loss(true_noise.squeeze(), noise_pred.squeeze())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 10 == 0: print(loss.item())

        if (epoch + 1) % save_inter == 0:
            torch.save(noise_predictor.state_dict(), os.path.join(SAVE_DIR, f'dit-epoch{epoch}'))
            torch.save(cond_model.state_dict(), os.path.join(SAVE_DIR, f'cond_model-epoch{epoch}'))
        if (epoch + 1) % eval_inter == 0:
            sample()

if __name__ == '__main__': 
    import argparse
    global dataset

    parser = argparse.ArgumentParser(description="Train a text diffusion model on a specified dataset.")
    parser.add_argument('--dataset', default='tripadvisor', type=str, choices=['reasoning', 'math', 'summarization'], help='Choose the dataset to train on: reasoning, math, or summarization.')
    args = parser.parse_args()
    dataset = args.dataset
    
    if dataset == 'winogrande':
        trainset, testset = load_winogrande()
    elif dataset == 'gsm8k':
        trainset, testset = load_gsm8k()
    # only tripadvisor summarization is supported
    elif dataset == 'tripadvisor':
        trainset, testset = load_tripadvisor()
    
    #train_vae(trainset, testset)
    train_conditional_diffusion(trainset, testset)