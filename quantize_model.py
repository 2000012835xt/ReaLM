

import torch
import numpy as np
import re
import json
import os

from torch import nn
from functools import partial
from smoothquant.fake_quant import W8A8Linear, W8A8BMM, NoisyW8A8Linear, NoisyW8A8BMM, W8A8MatMul, NoisyW8A8MatMul
from datasets import load_dataset
from smoothquant.smooth import smooth_lm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM

from torch.utils.data import DataLoader
from transformers import TextDataset, DataCollatorForLanguageModeling
from torch.nn import CrossEntropyLoss
import pdb
from tqdm import tqdm
import time

from sampling.autoregressive_sampling import autoregressive_sampling
import contexttimer
from rouge import Rouge
from rouge_score import rouge_scorer

from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )


def quantize_mistral_model(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )

    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, (LlamaMLP, MistralMLP)):
            m.gate_proj = W8A8Linear.from_float(
                m.gate_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.up_proj = W8A8Linear.from_float(
                m.up_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.down_proj = W8A8Linear.from_float(
                m.down_proj, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            print(name)
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = W8A8Linear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant
            )

            # m.matmul1=W8A8MatMul(act_quant=act_quant,quantize_output=False)
            # m.matmul2=W8A8MatMul(act_quant=act_quant,quantize_output=True)
    return model

def quantize_mistral_model_error(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True, err_prob = 0
):
    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )
    i = 0
    for name, m in model.model.named_modules():
        #print(name)
        if isinstance(m,MistralMLP):
            if i<59:
               
                m.gate_proj = NoisyW8A8Linear.from_float(
                    m.gate_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input,err_prob=err_prob
                )
                m.up_proj = NoisyW8A8Linear.from_float(
                    m.up_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input,err_prob=err_prob
                )
                m.down_proj = NoisyW8A8Linear.from_float(
                    m.down_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input,err_prob=err_prob
                )
                i += 1
            else:
                m.gate_proj = W8A8Linear.from_float(
                    m.gate_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input
                )
                m.up_proj = W8A8Linear.from_float(
                    m.up_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input
                )
                m.down_proj = W8A8Linear.from_float(
                    m.down_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input
                )               
        elif isinstance(m, MistralAttention):
            if i<59:
                # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
                # m.q_proj = W8A8Linear.from_float(
                #     m.q_proj,
                #     weight_quant=weight_quant,
                #     act_quant=act_quant,
                #     quantize_output=quantize_bmm_input,
                # )
                print(name)
                m.q_proj = NoisyW8A8Linear.from_float(
                    m.q_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                    err_prob=err_prob
                )
                # m.k_proj = W8A8Linear.from_float(
                #     m.k_proj,
                #     weight_quant=weight_quant,
                #     act_quant=act_quant,
                #     quantize_output=quantize_bmm_input,
                # )
                m.k_proj = NoisyW8A8Linear.from_float(
                    m.k_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                    err_prob=err_prob
                )
                # m.v_proj = W8A8Linear.from_float(
                #     m.v_proj,
                #     weight_quant=weight_quant,
                #     act_quant=act_quant,
                #     quantize_output=quantize_bmm_input,
                # )
                m.v_proj = NoisyW8A8Linear.from_float(
                    m.v_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                    err_prob=err_prob
                )
                # m.o_proj = W8A8Linear.from_float(
                #     m.o_proj, weight_quant=weight_quant, act_quant=act_quant
                # )
                m.o_proj = NoisyW8A8Linear.from_float(
                    m.o_proj, weight_quant=weight_quant, act_quant=act_quant,err_prob=err_prob
                )
            else:
                m.q_proj = W8A8Linear.from_float(
                    m.q_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                )
                m.k_proj = W8A8Linear.from_float(
                    m.k_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                )  
                m.v_proj = W8A8Linear.from_float(
                    m.v_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                ) 
                m.o_proj = W8A8Linear.from_float(
                    m.o_proj, weight_quant=weight_quant, act_quant=act_quant
                )                                    
    return model


def quantize_model(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True):
    i = 0
    for name, m in model.model.named_modules():
        if i < 1:
            if isinstance(m, OPTDecoderLayer):
                m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant, 
                                                quantize_output=True)
                m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
            elif isinstance(m, OPTAttention):
                print(name)
                # i = i + 1
                # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
                m.q_proj = W8A8Linear.from_float(
                    m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                m.k_proj = W8A8Linear.from_float(
                    m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                m.v_proj = W8A8Linear.from_float(
                    m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                m.out_proj = W8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant)

                m.bmm1=W8A8BMM(act_quant=act_quant,quantize_output=False)
                m.bmm2=W8A8BMM(act_quant=act_quant,quantize_output=True)

    return model

def quantize_model_error(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True, err_prob=0.0):
    i = 0
    for name, m in model.model.named_modules():
        if i < 2:
            if isinstance(m, OPTDecoderLayer):
                # m.fc1 = NoisyW8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant,err_prob=err_prob,
                #                                    quantize_output=True)
                m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant, 
                                              quantize_output=True)
                m.fc2 = NoisyW8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant,err_prob=err_prob)
                # m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
                # m.final_layer_norm=layer_norm_without_outlier.from_float(m.final_layer_norm, percentage=1.0)
            elif isinstance(m, OPTAttention):
                print(name)
                i = i + 1
                # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
                # m.q_proj = NoisyW8A8Linear.from_float(
                #     m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input,err_prob=err_prob)
                m.q_proj = W8A8Linear.from_float(
                     m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                # m.k_proj = NoisyW8A8Linear.from_float(
                #     m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input,err_prob=err_prob)
                m.k_proj = W8A8Linear.from_float(
                    m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                # m.v_proj = NoisyW8A8Linear.from_float(
                #     m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input,err_prob=err_prob)
                m.v_proj = W8A8Linear.from_float(
                    m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                # m.out_proj = NoisyW8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant,err_prob=err_prob)
                m.out_proj = W8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant)

                # m.bmm1=NoisyW8A8BMM(act_quant=act_quant,quantize_output=False,err_prob=err_prob)
                m.bmm1=W8A8BMM(act_quant=act_quant,quantize_output=False)
                # m.bmm2=NoisyW8A8BMM(act_quant=act_quant,quantize_output=True,err_prob=err_prob)
                m.bmm2=W8A8BMM(act_quant=act_quant,quantize_output=True)
        else:
            if isinstance(m, OPTDecoderLayer):
                m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant, 
                                                quantize_output=True)
                m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
            elif isinstance(m, OPTAttention):
                print(name)
                i = i + 1
                # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
                m.q_proj = W8A8Linear.from_float(
                    m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                m.k_proj = W8A8Linear.from_float(
                    m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                m.v_proj = W8A8Linear.from_float(
                    m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                m.out_proj = W8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant)

                m.bmm1=W8A8BMM(act_quant=act_quant,quantize_output=False)
                m.bmm2=W8A8BMM(act_quant=act_quant,quantize_output=True)
    return model