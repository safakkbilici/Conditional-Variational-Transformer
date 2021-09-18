import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm

from utils.model_utils import *


def evaluate(model, device, criterion, test_dataloader, stepp, args):

    total = len(test_dataloader)
    
    with tqdm(total = total) as ee:
        with torch.inference_mode():
            nll_total_loss, kl_total_loss, n_word_total, n_word_correct = 0, 0, 0, 0
            for step, batch in enumerate(test_dataloader):
                model.eval()

                src = batch[0].to(device)
                trg = batch[1].to(device)
                label = batch[2].to(device)
                batch_size = batch[0].size(0)

                src_seq = patch_src(src, tokenizer.pad_token_id)
                trg_seq, gold = map(lambda x: x.to(device), patch_trg(trg, tokenizer.pad_token_id))

                prior = sample_from_prior(batch_size, args.latent_size, device)
                prior[:,0] = label

                pred, kl_loss = model(src_seq.long(), trg_seq.long(), prior, debug = False)

                loss, n_correct, n_word = cal_performance(pred, gold, args.tokenizer.pad_token_id, smoothing=True)

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
                ee.update()

        loss_per_word = nll_total_loss/n_word_total
        accuracy = n_word_correct/n_word_total

        return accuracy, loss_per_word, kl_total_loss
