import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm

from utils.model_utils import *
from models.masking import get_subsequent_mask


def evaluate(model, device, criterion, test_dataloader, stepp, args, tokenizer, latent_size):

    total = len(test_dataloader)
    
    with tqdm(total = total, leave=False, desc='Validation round', position = 0) as ee:
        with torch.no_grad():
            nll_total_loss, kl_total_loss, n_word_total, n_word_correct, perp_total = 0, 0, 0, 0, 0
            for step, batch in enumerate(test_dataloader):
                model.eval()

                src = batch[0].to(device)
                trg = batch[1].to(device)
                label = batch[2].to(device)
                batch_size = batch[0].size(0)

                src_seq = patch_src(src, tokenizer.pad_token_id)
                trg_seq, gold = map(lambda x: x.to(device), patch_trg(trg, tokenizer.pad_token_id))

                prior = sample_from_prior(batch_size, latent_size, device)
                prior[:,0] = label

                pred, kl_loss = model(src_seq.long(), trg_seq.long(), prior, debug = False)

                loss, n_correct, n_word = cal_performance(pred, gold, tokenizer.pad_token_id, smoothing=False)

                perplexity = torch.exp(loss)

                if args.posterior_collapse:
                    kl_weight = kl_anneal_function(
                        args.anneal_function,
                        stepp,
                        args.k,
                        args.x0
                    )
                    kl_loss = kl_loss * kl_weight

                overall_loss = (kl_loss + loss) / 2

                n_word_total += n_word
                n_word_correct += n_correct

                kl_total_loss += kl_loss.item()
                nll_total_loss += loss.item()
                perp_total += perplexity.item()
                
                ee.update()

        loss_per_word = nll_total_loss/n_word_total
        accuracy = n_word_correct/n_word_total
        perp_total = perp_total / len(test_dataloader)

        return accuracy, loss_per_word, kl_total_loss, perp_total

def generate(model, device, tokenizer, latent_size, n_classes, n_samples_per_class, generate_len):
    total = n_classes * n_samples_per_class * (generate_len - 2)
    generated = {str(i): [] for i in range(n_classes)}
    model = model.to(device)
    with tqdm(total = total, leave=False, desc='Generation round', position = 0) as gen:

        for i in range(n_classes):
            for j in range(n_samples_per_class):
                prior = sample_from_prior(1, latent_size, device)
                prior[:,0] = i
                trg = [tokenizer.start_token_id]
                trg_seq = torch.Tensor(trg)[None, :].long().to(device)
                with torch.no_grad():
                    for step in range(2, generate_len):
                        trg_mask = get_subsequent_mask(trg_seq)
                        dec_output = model.generate(trg_seq, trg_mask, prior, None)
                        gen_seq = F.softmax(model.trg_word_prj(dec_output), dim=-1).squeeze(0).squeeze(0)
                        try:
                            max_prob = gen_seq[-1,:].argmax(dim = -1)
                        except:
                            max_prob = gen_seq.argmax(dim = -1)

                        trg.append(max_prob.item())
                        trg_seq = torch.Tensor(trg)[None, :].long().to(device)
                        gen.update()
                generated[str(i)].append(' '.join(tokenizer.decode(trg, remove_special_tokens=True)))
    return generated

# def generate(model, device, tokenizer, latent_size, n_classes, n_samples_per_class, generate_len):
#     samples = defaultdict(list)
#     total = n_classes * n_samples_per_class * generate_len
#     model = model.to(device)
#     with tqdm(total = total, leave=False, desc='Generation round', position = 0) as gg:
#         for i in range(n_classes):
#             samples[str(i)] = []
#             for j in range(n_samples_per_class):
#                 prior = sample_from_prior(1, latent_size, device)
#                 prior[:,0] = i
#                 trg = [tokenizer.start_token_id]
#                 trg_seq = torch.Tensor(trg)[None, :].long().to(device)
            
#                 with torch.no_grad():
#                     model.eval()
#                     for step in range(2, generate_len):
#                         trg_mask = get_subsequent_mask(trg_seq)
#                         dec_output = model.generate(trg_seq, trg_mask, prior, None)
                        
#                         gen_seq = F.softmax(model.trg_word_prj(dec_output), dim=-1).squeeze(0).squeeze(0)

#                         try:
#                             max_prob = gen_seq[-1,:].argmax(dim = -1)
#                         except:
#                             max_prob = gen_seq.argmax(dim = -1)
                            
#                         trg.append(max_prob.item())
#                         trg_seq = torch.Tensor(trg)[None, :].long().to(device)
#                         gg.update()

#                 generated_sentence = ' '.join(tokenizer.decode(trg, remove_special_tokens=True))
#                 samples[str(i)].append(generated_sentence)

#     return samples
