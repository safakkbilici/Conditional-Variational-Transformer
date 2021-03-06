import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from utils.model_utils import *

def generate(model, device, tokenizer, latent_size, generate_len, n_generate):
    total = n_generate * (generate_len - 2)
    model = model.to(device)
    generated = []
    with tqdm(total = total, leave=False, desc='Generation round', position = 0) as gen:
        for i in range(n_generate):
            prior = sample_from_prior(1, latent_size, device)
            trg = [tokenizer.start_token_id, random.randint(5, tokenizer.vocab_size)]
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
            generated.append(' '.join(tokenizer.decode(trg, remove_special_tokens=True)))
    return generated

def evaluate(model, device, criterion, test_dataloader, stepp, args, tokenizer, latent_size):

    total = len(test_dataloader)
    
    with tqdm(total = total, leave=False, desc='Validation round', position = 0) as ee:
        with torch.no_grad():
            nll_total_loss, kl_total_loss, n_word_total, n_word_correct, perp_total = 0, 0, 0, 0, 0
            for step, batch in enumerate(test_dataloader):
                model.eval()

                src = batch["original_segment"].to(device)
                trg = batch["noised_segment"].to(device)
                batch_size = batch["original_segment"].size(0)

                src_seq = patch_src(src, tokenizer.pad_token_id)
                trg_seq, gold = map(lambda x: x.to(device), patch_trg(trg, tokenizer.pad_token_id))

                prior = sample_from_prior(batch_size, latent_size, device)

                pred, kl_loss = model(src_seq.long(), trg_seq.long(), prior, debug = False)

                loss, n_correct, n_word = cal_performance(pred, gold, tokenizer.pad_token_id, smoothing=False)

                #perplexity = torch.exp(loss)

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
                #perp_total += perplexity.item()
                
                ee.update()

        loss_per_word = nll_total_loss/n_word_total
        accuracy = n_word_correct/n_word_total
        perp_total = torch.exp(torch.Tensor([loss_per_word])).item()

        return accuracy, loss_per_word, kl_total_loss, perp_total

def train(**params):
    model = params["model"]
    device = params["device"]
    optimizer = params["optimizer"]
    criterion = params["criterion"]
    scheduler = params["scheduler"]
    train_dataloader = params["train_dataloader"]
    test_dataloader = params["train_dataloader"]
    tokenizer = params["tokenizer"]

    args = params["args"]

    kl = []
    nll = []
    returns = {}

    total = len(train_dataloader) * args.epochs

    model = model.to(device)
    with tqdm(total = total, desc='Training round') as tt:
        stepp = 0
        for epoch in range(args.epochs):
            nll_total_loss, kl_total_loss, n_word_total, n_word_correct = 0, 0, 0, 0
            for step, batch in enumerate(train_dataloader):
                stepp += 1
                model.train()

                src = batch["original_segment"].to(device)
                trg = batch["noised_segment"].to(device)
                batch_size = batch["original_segment"].size(0)

                src_seq = patch_src(src, tokenizer.pad_token_id)
                trg_seq, gold = map(lambda x: x.to(device), patch_trg(trg, tokenizer.pad_token_id))

                prior = sample_from_prior(batch_size, args.latent_size, device)

                optimizer.zero_grad()
                pred, kl_loss = model(src_seq.long(), trg_seq.long(), prior, debug = False)
                loss, n_correct, n_word = cal_performance(pred, gold, tokenizer.pad_token_id, smoothing=True)

                if args.posterior_collapse == "true":
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

            if (epoch+1) % args.evaluate_per_epoch == 0:
                loss_per_word = nll_total_loss/(n_word_total+1)
                accuracy = n_word_correct/(n_word_total+1)
            
                test_acc, test_nll, test_kl, test_perp = evaluate(
                    model = model,
                    device = device,
                    criterion = criterion,
                    test_dataloader = test_dataloader,
                    stepp = stepp,
                    args = args,
                    tokenizer = tokenizer,
                    latent_size = args.latent_size
                )

                generated = generate(
                    model = model,
                    device = device,
                    tokenizer = tokenizer,
                    latent_size = args.latent_size,
                    generate_len = 50,
                    n_generate = 3
                )

                torch.save(model.state_dict(), "model.pt")
                torch.save(optimizer.state_dict(), "optimizer.pt")
                if scheduler != None:
                    torch.save(scheduler.state_dict(), "scheduler.pt")
                    

                print(str(epoch)+"-" * 30)
                print(f"Train Accuracy: {accuracy}")
                print(f"Train Negative Log Likelihood: {loss_per_word}")
                print(f"Train KL-Divergence: {kl_total_loss}")
                print(f"Test Accuracy: {test_acc}")
                print(f"Test Negative Log Likelihood: {test_nll}")
                print(f"Test KL-Divergence: {test_kl}")
                print(f"Test Perplexity: {test_perp}")

                kl.append(kl_total_loss)
                nll.append(loss_per_word)

                for g in generated:
                    print(f"  - {g}")

            if scheduler != None:
                scheduler.step()

    returns["model"] = model
    returns["optimizer"] = optimizer
    returns["scheduler"] = scheduler
    returns["nll_history"] = nll
    returns["kl_history"] = kl

    return returns
