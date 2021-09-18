import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm

from utils.model_utils import *
from models.eval import evaluate


def train(**params):
    model = params["model"]
    device = params["device"]
    optimizer = params["optimizer"]
    criterion = params["criterion"]
    scheduler = params["scheduler"]
    train_dataloader = params["train_dataloader"]
    test_dataloader = params["train_dataloader"]

    args = params["args"]

    kl = []
    nll = []
    returns = {}

    total = len(train_dataloader) * args.epochs
    
    model = model.to(device)
    with tqdm(total = total) as tt:
        stepp = 0
        for epoch in range(args.epochs):
            nll_total_loss, kl_total_loss, n_word_total, n_word_correct = 0, 0, 0, 0
            for step, batch in enumerate(train_dataloader):
                stepp += 1
                model.train()

                src = batch[0].to(device)
                trg = batch[1].to(device)
                label = batch[2].to(device)
                batch_size = batch[0].size(0)

                src_seq = patch_src(src, tokenizer.pad_token_id)
                trg_seq, gold = map(lambda x: x.to(device), patch_trg(trg, tokenizer.pad_token_id))

                prior = sample_from_prior(batch_size, args.latent_size, device)
                prior[:,0] = label

                optimizer.zero_grad()

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
                overall_loss.backward()
                optimizer.step()

                n_word_total += n_word
                n_word_correct += n_correct

                kl_total_loss += kl_loss.item()
                nll_total_loss += loss.item()
                tt.update()

            loss_per_word = nll_total_loss/n_word_total
            accuracy = n_word_correct/n_word_total
            test_acc, test_nll, test_kl = evaluate(
                model = model,
                device = device,
                criterion = criterion,
                test_dataloader = test_dataloader,
                stepp = stepp,
                args = args
            )
            
            print(str(epoch)+"-" * 30)
            print(f"Train Accuracy: {accuracy}")
            print(f"Train Negative Log Likelihood: {loss_per_word}")
            print(f"Train KL-Divergence: {kl_total_loss}")
            print(f"Test Accuracy: {test_accuracy}")
            print(f"Test Negative Log Likelihood: {test_nll}")
            print(f"Test KL-Divergence: {test_kl}")

            kl.append(kl_total_loss)
            nll.append(loss_per_word)

            if scheduler != None:
                scheduler.step()

    returns["model"] = model
    returns["optimizer"] = optimizer
    returns["scheduler"] = scheduler
    returns["nll_history"] = nll
    returns["kl_history"] = kl

    return returns