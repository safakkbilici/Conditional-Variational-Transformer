import torch
from typing import Union, Dict

class Freezer():
    def __init__(self, t: Union[str, Dict] = "last", trg_word_proj = False):
        if t not in ["first","last", "all"] and type(t) != dict:
            raise NotImplementedError
        
        self.t = t
        self.trg_word_proj = trg_word_proj

    def freeze(self, model, check = True):
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        
        for param in model.encoder.src_word_emb.parameters():
            param.requires_grad = False

        for param in model.decoder.trg_word_emb.parameters():
            param.requires_grad = False

        if self.t == "last":
            enc_layer_stack = model.encoder.layer_stack[:-1]
            dec_layer_stack = model.decoder.layer_stack[:-1]
        elif self.t == "first":
            enc_layer_stack = model.encoder.layer_stack[1:]
            dec_layer_stack = model.decoder.layer_stack[1:]
        elif self.t == "all":
            enc_layer_stack = model.encoder.layer_stack
            dec_layer_stack = model.decoder.layer_stack
        elif type(t) == dict:
            enc_layer_stack = []
            dec_layer_stack = []
            for to_freeze_enc_layers in self.t["enc"]:
                enc_layer_stack.append(model.encoder.layer_stack[to_freeze_enc_layers])

            for to_freeze_dec_layers in self.t["dec"]:
                dec_layer_stack.append(model.decoder.layer_stack[to_freeze_dec_layers])

        for child in enc_layer_stack:
            for param in child.parameters():
                param.requires_grad = False

        for child in dec_layer_stack:
            for param in child.parameters():
                param.requires_grad = False

        if self.trg_word_proj:
            for param in self.model.trg_word_prj:
                param.reqiures_grad = False

        print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        return model
