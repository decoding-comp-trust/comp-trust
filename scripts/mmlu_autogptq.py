"""Different from the original MMLU test code. We let LLM generate 16 tokens and 
match the answer (A,B,C or D) in the generation. We also faciliate the parallel 
evaluation through `--subject=<task name>`."""
import argparse
import os
import numpy as np
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM,BaseQuantizeConfig
import wandb

from crop import crop

choices = ["A", "B", "C", "D"]


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:" 
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def predict_one_sample(prompt, answers):
    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        input_ids=torch.as_tensor(input_ids).cuda(),
        do_sample=True,
        # top_k=1,
        top_p=1,
        temperature=1e-7,
        max_new_tokens=16,
    )
    output_ids = output_ids[0][len(input_ids[0]):]
    outputs = tokenizer.decode(output_ids, skip_special_tokens=True).replace("</s>", "").lower()
    
    pred=outputs[0].upper()

    if pred in ['A','B','C','D']:
        return pred
    else:
        return None

def eval(args, subject,  dev_df, test_df):
    lazy_load_model()
    cors = []
    refs = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]

    for i in tqdm(range(test_df.shape[0]), desc=subject):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]

        pred = predict_one_sample(prompt, answers)
        

        cor = pred == label
        ref = pred == None
        cors.append(cor)
        refs.append(ref)

    acc = np.mean(cors)
    refusal = np.mean(refs)
    cors = np.array(cors)
    refs = np.array(refs)

    print("Average accuracy {:.3f} - {}, refusal:{:.3f}".format(acc, subject,refusal))

    return cors, refs, acc, refusal, all_probs

def main(args):
    if args.subject is None:
        subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])
    else:
        subjects = [args.subject]

    print("test subjects:", subjects)
    # print(args)

    all_cors = []
    all_refs = []

    for subject in tqdm(subjects, desc='subj'):
        result_path = os.path.join(args.save_path, f"results_{subject}.csv")
        if args.resume and os.path.exists(result_path):
            print(f"resume: {result_path}")
            test_df = pd.read_csv(result_path)
            cors = test_df["{}_correct".format(args.model_name)].tolist()
            refs = test_df["{}_refusal".format(args.model_name)].tolist()
        else:
            dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
            test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

            cors, refs, acc, refusal, _ = eval(args, subject,  dev_df, test_df)

            test_df["{}_correct".format(args.model_name)] = cors
            test_df["{}_refusal".format(args.model_name)] = refs 
            test_df.to_csv(result_path, index=None)
        all_cors.append(cors)
        all_refs.append(refs)
        
        wandb.log({
            f'{subject} accuracy': np.mean(cors),
            f'{subject} refusal': np.mean(refs),
        })

    weighted_acc = np.mean(np.concatenate(all_cors))
    weighted_ref = np.mean(np.concatenate(all_refs))

    print("Average accuracy: {:.3f}".format(weighted_acc))
    print("Average refusal: {:.3f}".format(weighted_ref))
    wandb.log({
        'accuracy': weighted_acc,
        'refusal': weighted_ref,
        'n': len(all_cors)
    })

def lazy_load_model():
    global model
    if model is None:
        print("Lazy load model...")
        kwargs = {}
        if '70b' in args.model_name:
            kwargs = {**kwargs, 
                    "disable_exllama": True,
                    "disable_exllamav2": True,}
        model = AutoGPTQForCausalLM.from_quantized(
            args.model_name,
            inject_fused_mlp=True,
            inject_fused_attention=False,
            quantize_config=BaseQuantizeConfig(bits=args.bits, group_size=128, desc_act=True),
            revision=args.revision,
            use_safetensors=False,
            device_map='auto',
            **kwargs,
        )
        model.config.pad_token_id = model.config.eos_token_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name')
    parser.add_argument('--tokenizer-name')
    parser.add_argument('--bits', type=int, default=None)
    parser.add_argument('--num_sample', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--template', default='default')
    parser.add_argument('--revision', type=str, default=None)
    parser.add_argument('--test_seed', type=int, default=None, help='seed for test (not model).')
    parser.add_argument('--temperature', default=1e-7, type=float)
    # mmlu
    parser.add_argument("--subject", type=str, default=None, 
                        choices=['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'])
    parser.add_argument("--ntrain", "-k", type=int, default=5, help='shots of demos in ICL.')
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--resume", action='store_true')
    args = parser.parse_args()
    
    if args.revision is None:
        args.revision = f"{args.bits}bit_{args.num_sample}g_{args.seed}seed"
        print(f"Auto revision: {args.revision}")
    
    args.save_path = f'./mmlu-autogptq-results/{args.model_name}-{args.revision}'
    if args.temperature != 1e-7:
        args.save_path += f'_t{args.temperature:g}'
    if args.test_seed is not None:
        args.save_path += f'_ts{args.test_seed}'
    if args.template != 'default':
        args.save_path += f'_te-{args.template}'
    print("save_path: ", args.save_path)
    os.makedirs(args.save_path, exist_ok=True)

    wandb.init(project='comp-test', config=vars(args))
    
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        use_fast=False,
        torch_dtype=torch.bfloat16,
        use_safetensors=False,
        device_map='auto'
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    model = None
    
    main(args)
