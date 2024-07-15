
import torch
import numpy as np
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import W8A8Linear, W8A8BMM, NoisyW8A8Linear, NoisyW8A8BMM, W8A8MatMul, NoisyW8A8MatMul
#from smoothquant.error_inject_msd import W8A8Linear, W8A8BMM, NoisyW8A8Linear, NoisyW8A8BMM
from datasets import load_dataset

from torch.utils.data import DataLoader
from transformers import TextDataset, DataCollatorForLanguageModeling
from torch.nn import CrossEntropyLoss
import pdb
from tqdm import tqdm
import time

import re
import argparse
import jsonlines
import datasets
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
)
from transformers import LlamaTokenizer
from sampling.autoregressive_sampling import autoregressive_sampling

def quantize_model(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True):
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant,quantize_output=True)
            m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, OPTAttention):
            print(name)
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

def quantize_model_error(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True, err_prob=0):
    i = 0
    for name, m in model.model.named_modules():
        if i <59:
            if isinstance(m, OPTDecoderLayer):
                #m.fc1 = NoisyW8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant,err_prob=err_prob, quantize_output=True)
                m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant)
                #m.fc2 = NoisyW8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant,err_prob=err_prob)
                #print(name)
                m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
            elif isinstance(m, OPTAttention):
                #print(name)
                # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
                #m.q_proj = NoisyW8A8Linear.from_float(
                    #m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input,err_prob=err_prob)
                m.q_proj = W8A8Linear.from_float(
                    m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                #m.k_proj = NoisyW8A8Linear.from_float(
                    #m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input,err_prob=err_prob)
                m.k_proj = W8A8Linear.from_float(
                    m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                #m.v_proj = NoisyW8A8Linear.from_float(
                        #m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input,err_prob=err_prob)
                m.v_proj = W8A8Linear.from_float(
                    m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                #m.out_proj = NoisyW8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant,err_prob=err_prob)
                m.out_proj = W8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant)

                m.bmm1=NoisyW8A8BMM(act_quant=act_quant,quantize_output=False,err_prob=err_prob)
                #m.bmm1=W8A8BMM(act_quant=act_quant,quantize_output=False)
                #m.bmm2=NoisyW8A8BMM(act_quant=act_quant,quantize_output=True,err_prob=err_prob)
                m.bmm2=W8A8BMM(act_quant=act_quant,quantize_output=True)
            i=i+1
        else:            
            if isinstance(m, OPTDecoderLayer):
                m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant)
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



def quantize_mistral_model(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False
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

            m.matmul1=W8A8MatMul(act_quant=act_quant,quantize_output=False)
            m.matmul2=W8A8MatMul(act_quant=act_quant,quantize_output=True)

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
            if i<1:
               
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
            if i<1:
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
                m.matmul1=NoisyW8A8MatMul(act_quant=act_quant,quantize_output=False,err_prob=err_prob)
                m.matmul2=NoisyW8A8MatMul(act_quant=act_quant,quantize_output=True,err_prob=err_prob)               
            else:
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
                m.matmul1=W8A8MatMul(act_quant=act_quant,quantize_output=False)
                m.matmul2=W8A8MatMul(act_quant=act_quant,quantize_output=True)                                    
    return model


# origin code from https://github.com/QwenLM/Qwen/blob/main/eval/evaluate_gsm8k.py

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

ans_re1 = re.compile(r"(\-?[0-9][0-9\.\,]*)")

ans_re2 = re.compile(r'=\s*(\$?-?[0-9][0-9\.\,]*)')

prefix_sky1 = 'answer is'
prefix_sky2 = '答案是'

INVALID_ANS = "[invalid]"

def get_match_str(match, idx):
    match_str = match[idx]
    match_str = match_str.replace(",", "")
    if match_str.endswith('.'):
        match_str = match_str[:-1]
    if match_str.endswith('.00'):
        match_str = match_str[:-3]
    if match_str.endswith('.0'):
        match_str = match_str[:-2]
    return match_str

def doc_to_text(doc):
    return (
        fewshot_prompt
        + "\nQuestion: "
        + doc["question"]
        + "\nLet's think step by step\n"
    )


def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    # print(len(tokens_list))
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.decode(
            tokens[raw_text_len:])
        sents.append(sent)
    return sents

def generate_sample(model, model_decode, tokenizer, input_txt):
    input_ids = tokenizer([input_txt], padding=False)["input_ids"]
    context_enc = torch.tensor(input_ids, device=model.device)
    raw_text_len = len(input_ids[0])
    #print(f"Input text: {input_txt}\n")
    # outputs = model.generate(context_enc, pad_token_id=tokenizer.pad_token_id, max_new_tokens=150)
    outputs = autoregressive_sampling(x=context_enc,model=model,model_decode=model_decode, N=150, temperature=1, top_k=1, top_p=0.)
    output_text = decode(outputs, tokenizer, raw_text_len)[0]
    #print(f"\nOutput text: {output_text}\n")
    return output_text


def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS

def extract_answer(text):
    if prefix_sky1 in text:
        text = text.split(prefix_sky1)[-1]
    if prefix_sky2 in text:
        text = text.split(prefix_sky2)[-1]
    match1 = re.findall(ans_re1, text)
    match2 = re.findall(ans_re2, text)
    ans = []
    if match1:
        match_str1 = get_match_str(match1, -1)
        ans.append(match_str1)
    if match2:
        match_str2 = get_match_str(match2, -1).replace('$','')
        ans.append(match_str2)
    if len(ans) > 0:
        return eval(ans[-1])
    else:
        return INVALID_ANS

def is_correct(completion, answer):
    completion = completion.split('</s>')[0]
    completion = completion.split('\n\n\n')[0]
    completion = completion.split("\n\n")[0]
    completion = completion.split("Question:")[0]

    gold = extract_answer_hf(answer)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."
    clear_answer = extract_answer(completion)
    print(completion)
    print(clear_answer, gold)
    return clear_answer == gold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
        default="",
    )
    parser.add_argument("-f", "--sample-input-file", type=str, default=None)
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="gsm8k_res.jsonl"
    )

    args = parser.parse_args()

    fewshot_prompt = open("./gsm8k_prompt.txt").read()
    if args.sample_input_file is not None:
        dataset = load_from_disk(args.sample_input_file)
    else:
        config = datasets.DownloadConfig(resume_download=True, max_retries=100)
        dataset = load_dataset("gsm8k", "main", download_config=config)

    test = dataset["test"]

    err_prob_list=[1e-4, 1e-3, 1e-2]
    # err_prob_list = [1]#, 2, 4, 8, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    start_time_all= time.time()
    for i in range(len(err_prob_list)):
        time_start=time.time()
        print(err_prob_list[i])
        print("Loading tokenizer ...")
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-v0.1", trust_remote_code=True, padding_side='left'
        )
        if "qwen" in "mistralai/Mistral-7B-v0.1".lower():
            tokenizer.pad_token = '<|extra_0|>'
            tokenizer.eos_token = '<|endoftext|>'
        else:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "[PAD]"

        print("Loading model ...")
        #model_path = '/home/jiawangzhao/sunking/llama/llama-2-7b'
        model_noisy = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1", device_map="auto", trust_remote_code=True
        ).eval()
        model_normal= AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1", device_map="auto", trust_remote_code=True
        ).eval()
       
        model_noisy.generation_config.do_sample = False
        model_normal.generation_config.do_sample = False
        act_scales = torch.load('act_scales/Mistral-7B.pt')
        print('smoothing')
        smooth_lm(model_normal, act_scales, 0.8)
        smooth_lm(model_noisy, act_scales, 0.8)
        print('quantizing')
        GSM8K_model_noisy = quantize_mistral_model_error(model_noisy,err_prob=err_prob_list[i])
        GSM8K_model_normal = quantize_mistral_model(model_normal)
        # GSM8K_model =model
        model_normal.generation_config = GenerationConfig.from_pretrained(
            "mistralai/Mistral-7B-v0.1", trust_remote_code=True
        )
        model_noisy.generation_config = GenerationConfig.from_pretrained(
            "mistralai/Mistral-7B-v0.1", trust_remote_code=True
        )
        print(GSM8K_model_noisy, GSM8K_model_normal)
      
        f_output = jsonlines.Writer(open(args.sample_output_file, "w", encoding="utf-8"))
        tot_length = test.num_rows
        acc_res = []
        for doc in tqdm(test,desc='evaluating'):
            #pdb.set_trace()
            context = doc_to_text(doc)
            completion = generate_sample(GSM8K_model_normal, GSM8K_model_noisy, tokenizer, context)
            answer = doc["answer"]
            acc = is_correct(completion, answer)
            doc["completion"] = completion
            doc["acc"] = acc
            f_output.write(doc)
            acc_res.append(acc)

        f_output.close()
        print(acc_res)
        print("Acc: ", np.mean(acc_res))
        time_end = time.time()
        time_i = time_end-time_start
        print('time_i',time_i/60)
        torch.cuda.empty_cache()

    end_time_all=time.time()
    time_all=end_time_all - start_time_all
    print('time_all', time_all/60)
    

