"""This script is used to log workers for each data point.
Usage:
# create result csv
python extract_csv.py
# create worker csv
# NOTE: Use your name initial for --worker!!! For example, JH for Junyuan Hong
# NOTE: Change result_dir to your local result path!!!
python extract_csv.py --mode=worker --result_dir=<Path-to-local-results-folder> --worker=<Your-Name-Initial>
git add ./ipynb/data/worker_sheet.csv

# remove content based on blacklist file.
python extract_csv.py --mode=clean_blacklist --result_dir=<Path-to-local-results-folder>
# python extract_csv.py --mode=clean_blacklist --result_dir=../DecodingTrust/results --worker=JH --dry_run
# Double check removing:
# Example: python extract_csv.py --mode=worker --result_dir=../DecodingTrust/results/ --worker=JH
"""
import os
import json
import sys
import argparse
import numpy as np
import csv

import pandas as pd
from glob import glob
from pandas import DataFrame
import re


perspective_name_mapping = {
    'adv-glue-plus-plus': 'AdvGLUE++',
    'adv_demonstration': 'Adv Demo',
    'fairness': 'Fairness',
    'machine_ethics': 'Ethics',
    'ood': 'OOD',
    'privacy': 'Privacy',
    'toxicity': 'Toxicity',
    'stereotype': 'Stereotype',
}

def load_blacklist():
    # determine if the perspective is blacklisted.
    blacklist_file = f'./ipynb/data/{args.worker}_blacklist.csv'
    print(f"Read blacklist from {blacklist_file}")
    assert os.path.exists(blacklist_file), f"Not found blacklist file: {blacklist_file}"
    df = pd.read_csv(blacklist_file)
    pers_cols = [v for k, v in perspective_name_mapping.items() if v in df.columns]
    df = df[['model_name'] + pers_cols]
    
    # blk_df = df[pers_cols].apply(lambda x: 'X' in x)
    for c in pers_cols:
        df[c] = df[c].apply(lambda x: 'X' in x if isinstance(x, str) else False)
    # blk_df['model_name'] = df['model_name']
    df = df.set_index('model_name')
    return df


def check_blacklist(pers_name, model_name, rm_file=None, rm_df_file=None, rm_dict_file=None):
    pers_name, _ = map_perspective_name_to_display_name(pers_name)
    do_rm_file = True

    vita_compressed_name_mapping = {
        'hf/compressed-llm/llama-2-13b-chat-magnitude-semistruct@0.5_2to4': 'hf/vita-group/llama-2-13b-chat_magnitude_semistruct@0.5_2to4',
        'hf/compressed-llm/llama-2-13b-chat-sparsegpt-semistruct@0.5_2to4': 'hf/vita-group/llama-2-13b-chat_sparsegpt_semistruct@0.5_2to4',
        'hf/compressed-llm/llama-2-13b-magnitude-semistruct@0.5_2to4': 'hf/vita-group/llama-2-13b_magnitude_semistruct@0.5_2to4',
        'hf/compressed-llm/llama-2-13b-sparsegpt-semistruct@0.5_2to4': 'hf/vita-group/llama-2-13b_sparsegpt_semistruct@0.5_2to4',
        'hf/compressed-llm/vicuna-13b-v1.3-magnitude-semistruct@0.5_2to4': 'hf/vita-group/vicuna-13b-v1.3_magnitude_semistruct@0.5_2to4',
        'hf/compressed-llm/vicuna-13b-v1.3-sparsegpt-semistruct@0.5_2to4': 'hf/vita-group/vicuna-13b-v1.3_sparsegpt_semistruct@0.5_2to4'
    }
    new_vita_name_mapping = {}
    for k, v in vita_compressed_name_mapping.items():
        new_vita_name_mapping.update({v: k})
    vita_compressed_name_mapping = new_vita_name_mapping
    
    if rm_df_file is not None:
        model_name_fmt = model_name
        rm_list = []
        for df_model_name in BLACKLIST_DF.index:
            if BLACKLIST_DF.loc[df_model_name][pers_name].item():
                model_name = model_name_fmt(df_model_name)
                rm_list.append(model_name)
        with open(rm_df_file, 'r') as file:
            # Read lines into a list
            lines = file.readlines()

        def _check_inline(patterns, line):
            for k, v in vita_compressed_name_mapping.items():
                if 'k' in line:
                    line.replace(k, v)
                    break
            for p in patterns:
                if p in line:
                    return True
            return False

        with open(rm_df_file, 'w') as file:
            for line in lines:
                if _check_inline(rm_list, line):
                    if args.dry_run:
                        print(f" REMOVE LINE: {line}")
                        file.write(line + '\n')
                    else:
                        pass
                else:
                    file.write(line + '\n')
                # for model_name in rm_list:
                #     if model_name in line:
                #         if args.dry_run:
                #             print(f" REMOVE LINE: {line}")
                #             file.write(line + '\n')
                #         else:
                #             break
                #     else:
                #         file.write(line + '\n')
    else:
        if not isinstance(model_name, str):
            print(f"ERROR: model_name={model_name}, type: {type(model_name)}")
        # determine if the perspective is blacklisted.

        try:
            if pers_name not in BLACKLIST_DF.columns or not BLACKLIST_DF.loc[model_name][pers_name].item():
                return
        except KeyError as e:
            if model_name in vita_compressed_name_mapping.keys():
                model_name = vita_compressed_name_mapping[model_name]

                if pers_name not in BLACKLIST_DF.columns or not BLACKLIST_DF.loc[model_name][pers_name].item():
                    return
            else:
                print(f"Error: {model_name} not found!")

        if rm_file is not None:
            print(f"Remove file in blacklist: {rm_file}")
            if not args.dry_run:
                os.remove(rm_file)
    # elif rm_df_file is not None:
    #     raise NotImplementedError()
    # elif rm_dict_file is not None:
    #     raise NotImplementedError()
    # else:
    #     raise RuntimeError("No file to delete")

def get_adv_demo_scores(breakdown=False):
    print('==> AdvDemo')
    fs = glob(os.path.join(RESULT_DIR, "adv_demonstration", "**", "*_score.json"), recursive=True)
    # assert any([f for f in fs if 'hf_compressed-llm_llama-2-13b-awq@3bit_128g_score' in f])
    # TODO: This won't work if OpenAI or Anthropic models start to have underscores
    model_names = [os.path.basename(f).removesuffix("_score.json").replace("_", "/", 2) for f in fs]
    model_scores = {}
    model_rejections = {}
    model_breakdowns = {}
    for (idx, f), model_name in zip(enumerate(fs), model_names):
        with open(f) as src:
            scores = json.load(src)
        if not scores:
            print(f"[AdvDemo] Found Null: {f}")
            continue
        if args.mode == 'clean_blacklist':
            check_blacklist('adv_demonstration', model_name, rm_file=f)
        model_scores[model_name] = scores["adv_demonstration"] * 100
        model_rejections[model_name] = scores["adv_demonstration_rej"] * 100
        model_breakdowns[model_name] = scores
        
        if args.check:
            if idx == 0:
                check_keys = set(scores.keys())
            else:
                cur_keys = set(scores.keys())
                if len(check_keys) > len(cur_keys):
                    print(f"  - ERROR: {model_name} has missing keys: {check_keys-cur_keys}")
                elif len(check_keys) < len(cur_keys):
                    print(f"  + ERROR: {model_name} has extra keys: {cur_keys - check_keys}")
    if breakdown:
        return model_breakdowns
    else:
        return {"score": model_scores, "rejection_rate": model_rejections}


def get_advglue_scores(breakdown=False):
    print('==> AdvGlue')
    fname = os.path.join(RESULT_DIR, "adv-glue-plus-plus", "summary.json")
    # print()
    if args.mode == 'clean_blacklist':
        check_blacklist('adv-glue-plus-plus', lambda model_name: '/'+model_name, rm_df_file=fname)
    
    scores = pd.read_json(fname).to_dict()
    model_scores = {k.removeprefix("/"): v * 100 for k, v in scores["Accuracy"].items()}
    model_rejections = {k.removeprefix("/"): v * 100 for k, v in scores["RR+NE"].items()}
    if breakdown:
        with open(os.path.join(RESULT_DIR, "adv-glue-plus-plus", "breakdown.json")) as src:
            breakdown_scores = json.load(src)
            return breakdown_scores
    else:
        return {"score": model_scores, "rejection_rate": model_rejections}


def get_fairness_scores(breakdown=False):
    print('==> Fairness')
    fs = glob(os.path.join(RESULT_DIR, "fairness", "**", "final_scores.json"), recursive=True)
    model_names = [
        os.path.dirname(x).removeprefix(os.path.join(RESULT_DIR, "fairness", "results")).removeprefix("/") for x in fs
    ]
    model_scores = {}
    model_rejections = {}
    model_breakdown = {}
    for (idx, f), model_name in zip(enumerate(fs), model_names):
        with open(f) as src:
            scores = json.load(src)
        if args.mode == 'clean_blacklist':
            check_blacklist('fairness', model_name, rm_file=f)
        # model_scores[model_name] = scores.get("fairness score", None)
        model_scores[model_name] = scores.get("fairness score (new)", None)
        model_rejections[model_name] = scores.get("rejection rate", None)
        # model_breakdown[model_name] = {
        #     "zero-shot": {
        #         "Acc": "",
        #         "Demographic Parity Difference": "",
        #         "Equalized Odds Difference": " "
        #     },
        #     "few-shot-1": {},
        #     "few-shot-2": {},
        #     "Averaged Score": {},
        # }
        if args.check:
            if idx == 0:
                check_keys = set(scores.keys())
                print(f"  # REFER: {model_name}")
                print(f"   keys: {check_keys}")
            else:
                cur_keys = set(scores.keys())
                if len(check_keys) > len(cur_keys):
                    print(f"  - ERROR: {model_name} has missing keys: {check_keys-cur_keys}")
                elif len(check_keys) < len(cur_keys):
                    print(f"  + ERROR: {model_name} has extra keys: {cur_keys - check_keys}")
    return {"score": model_scores, "rejection_rate": model_rejections}


def get_ethics_scores(breakdown=False):
    print('==> Ethics')
    fname = os.path.join(RESULT_DIR, "machine_ethics", "generations", "scores.jsonl")
    df = pd.read_json(fname, lines=True)
    if args.mode == 'clean_blacklist':
        check_blacklist('machine_ethics', lambda model_name: model_name.replace('/', r'\/'), rm_df_file=fname)
    if breakdown:
        keys = ["avg_fpr_ev", "avg_fpr_jb", "acc_few", "acc_zero"]
        df = df[df["dataset"] == "ethics_commonsense_short"].drop_duplicates()
        df = df[["model"] + keys]
        df = df.rename({
            "acc_few": "few-shot benchmark",
            "acc_zero": "zero-shot benchmark",
            "avg_fpr_jb": "jailbreak",
            "avg_fpr_ev": "evasive"
        }, axis=1)

        model_breakdown = {}
        for record in df.to_dict(orient="records"):
            model_breakdown["model"] = {
                "few-shot benchmark": record["few-shot benchmark"],
                "zero-shot benchmark": record["zero-shot benchmark"],
                "jailbreak": record["jailbreak"],
                "evasive": record["evasive"]
            }
        # "jailbreak": {
        #     "brittleness": 1.0
        # },
        # "evasive": {
        #     "brittleness": 1.0
        # },
        # "zero-shot benchmark": {
        #     "performance": 0.533902323376007
        # },
        # "few-shot benchmark": {
        #     "performance": 0.683262209577999
        # }
        return model_breakdown
    else:
        keys = ["agg_score", "ref_rate"]
        df = df[df["dataset"] == "ethics_commonsense_short"].drop_duplicates().set_index("model")[keys]
        return df.to_dict()


def get_ood_scores(breakdown=False):
    print('==> OOD')
    path_prefix = os.path.join(RESULT_DIR, "ood", "results/")
    fs = glob(os.path.join(path_prefix, "**", "final_scores.json"), recursive=True)
    model_names = [os.path.dirname(f).removeprefix(path_prefix) for f in fs]
    model_scores = {}
    model_rejections = {}
    model_breakdowns = {}
    for (idx, f), model_name in zip(enumerate(fs), model_names):
        with open(f) as src:
            try:
                scores = json.load(src)
            except json.JSONDecodeError:
                print(f"JSONDecodeError: {f}")
                continue
        if not scores:
            continue
        if args.mode == 'clean_blacklist':
            check_blacklist('ood', model_name, rm_file=f)
        if "score" not in scores:
            print(f"!!ERROR score is missing in model: {model_name}")
            continue
        model_scores[model_name] = scores["score"]
        model_rejections[model_name] = scores.get("rr", None)
        model_breakdowns[model_name] = scores
        
        if args.check:
            if idx == 0:
                check_keys = set(scores.keys())
            else:
                cur_keys = set(scores.keys())
                if len(check_keys) > len(cur_keys):
                    print(f"  - ERROR: {model_name} has missing keys: {check_keys-cur_keys}")
                elif len(check_keys) < len(cur_keys):
                    print(f"  + ERROR: {model_name} has extra keys: {cur_keys - check_keys}")
    if breakdown:
        return model_breakdowns
    else:
        return {"score": model_scores, "rejection_rate": model_rejections}


def get_privacy_scores(breakdown=False):
    print('==> Privacy')
    fname = os.path.join(RESULT_DIR, "privacy", "generations", "scores.jsonl")
    df = pd.read_json(fname, lines=True)
    if args.mode == 'clean_blacklist':
        check_blacklist('privacy', lambda model_name: model_name.replace('/', r'_'), rm_df_file=fname)
    # TODO: This won't work if OpenAI or Anthropic models start to have underscores
    df["model"] = df["model"].apply(lambda x: x.replace("_", "/", 2))
    if breakdown:
        keys = ["enron", "pii", "understanding"]
        model_breakdown = {}
        models = df["model"].unique().tolist()
        for model in models:
            model_breakdown[model] = {}
            for key in keys:
                df_key = df[df["dataset"] == key].drop_duplicates().set_index("model")
                model_breakdown[model][key] = {"asr": df_key.loc[model, "leak_rate"]}
        return model_breakdown
    else:
        df = df[df["dataset"] == "all"].drop_duplicates().set_index("model")
        return df[["privacy_score", "reject_rate", "privacy_score_wo_reject"]].to_dict()


def get_stereotype_scores():
    print('==> Stereotype')
    path_prefix = os.path.join(RESULT_DIR, "stereotype", "generations/")
    fs = glob(os.path.join(path_prefix, "**", "25_compiled.json"), recursive=True)
    model_names = [os.path.dirname(f).removeprefix(path_prefix) for f in fs]
    model_scores = {}
    model_rejections = {}
    for (idx, f), model_name in zip(enumerate(fs), model_names):
        with open(f) as src:
            scores = json.load(src)
        if not scores:
            continue
        if args.mode == 'clean_blacklist':
            check_blacklist('stereotype', model_name, rm_file=f)
        model_scores[model_name] = scores["overall_score"] * 100
        model_rejections[model_name] = scores["overall_rejection_rate"] * 100
        
        if args.check:
            if idx == 0:
                check_keys = set(scores.keys())
            else:
                cur_keys = set(scores.keys())
                if len(check_keys) > len(cur_keys):
                    print(f"  - ERROR: {model_name} has missing keys: {check_keys-cur_keys}")
                elif len(check_keys) < len(cur_keys):
                    print(f"  + ERROR: {model_name} has extra keys: {cur_keys - check_keys}")
        
    return {"score": model_scores, "rejection_rate": model_rejections}


def get_toxicity_scores():
    print('==> Toxicity')
    path_prefix = os.path.join(RESULT_DIR, "toxicity", "user_prompts", "generations/")
    fs = glob(os.path.join(path_prefix, "**", "report.jsonl"), recursive=True)
    model_names = [os.path.dirname(f).removeprefix(path_prefix) for f in fs]
    model_scores = {}
    model_rejections = {}
    for (idx, f), model_name in zip(enumerate(fs), model_names):
        with open(f) as src:
            scores = json.load(src)
        if not scores:
            continue
        if args.mode == 'clean_blacklist':
            check_blacklist('toxicity', model_name, rm_file=f)
        score_key = os.path.join(model_name, "aggregated-score")
        # score_key = os.path.join(model_name + '_strip', "aggregated-score")
        # print(f"### score_key: {score_key}")
        if score_key not in scores or scores[score_key] is None or np.isnan(scores[score_key]):
            continue
        model_scores[model_name] = scores[score_key] * 100
        model_rejections[model_name] = np.mean([v for k, v in scores.items() if k.endswith("rej_rates")])
        
        if args.check:
            if idx == 0:
                check_keys = {k.split('/')[-1] for k in set(scores.keys())}
            else:
                cur_keys = {k.split('/')[-1] for k in set(scores.keys())}
                if len(check_keys) > len(cur_keys):
                    print(f"  - ERROR: {model_name} has missing keys: {check_keys-cur_keys}")
                elif len(check_keys) < len(cur_keys):
                    print(f"  + ERROR: {model_name} has extra keys: {cur_keys - check_keys}")
                dif_set = cur_keys - check_keys
                if len(dif_set) > 0:
                    print(f"  * ERROR: different keys: {dif_set}")
                assert len(check_keys - cur_keys) == 0
    return {"score": model_scores, "rejection_rate": model_rejections}


def summarize_results():
    summarized_results = {
        "aggregated_results": {
            "adv_demonstration": get_adv_demo_scores(),
            "adv-glue-plus-plus": get_advglue_scores(),
            "fairness": get_fairness_scores(),
            "machine_ethics": get_ethics_scores(),
            "ood": get_ood_scores(),
            "privacy": get_privacy_scores(),
            "stereotype": get_stereotype_scores(),
            "toxicity": get_toxicity_scores()
        },

    }

    summarized_results = sort_keys(summarized_results)

    # mapping



    return summarized_results


def map_perspective_name_to_display_name(perspective_name):
    perspective_name_ref_mapping = {}
    for key, v in perspective_name_mapping.items():
        perspective_name_ref_mapping.update({key: v + ' Ref'})

    return perspective_name_mapping[perspective_name], perspective_name_ref_mapping[perspective_name]

def load_avg_acc(path):
    df = pd.read_csv(path)
    df_ = df[['Display Name', 'Avg. Acc']]
    return df_

def get_display_name(model_name):
    if model_name in ['anthropic/claude-2.0', 'openai/gpt-3.5-turbo-0301']:
        return model_name, model_name
    sparsity = get_sparsity(model_name)
    method, submethod, compression_suffix, model_size = get_compression_method(model_name)

    if 'llama-2' in model_name.lower():
        family = 'Llama-2'
    elif 'vicuna' in model_name:
        family = 'Vicuna'
    else:
        raise NotImplementedError

    if 'vicuna' in model_name or 'chat' in model_name:
        chat = '-chat'
    else:
        chat = ''

    sparsity = str(sparsity) if method == 'quantization' else ''

    old_display_name = family + f'-{model_size}' + chat + '-' + submethod.lower() + sparsity
    new_display_name = family + f'-{model_size}' + chat
    if submethod.lower() != 'none':
        new_display_name += '-' + submethod.lower() + compression_suffix
    return old_display_name, new_display_name


def get_base_model(model_name):
    if 'llama-2' in model_name.lower():
        family = 'LLAMA2'
    elif 'vicuna' in model_name:
        family = 'Vicuna'
    else:
        raise NotImplementedError

    if 'vicuna' in model_name or 'chat' in model_name:
        chat = ' Chat'
    else:
        chat = ''
    found = re.search(r"(?<=\D)\d+b(?=\D|$)", model_name)
    if not found:
        raise RuntimeError()
    model_size = found[0] if found else None

    return family + f' {model_size.lower()}' + chat


def get_compression_method(model_name):
    if "claude" in model_name.lower():
        return 'none', 'none', '', None
    found = re.search(r"(?<=\D)\d+b(?=\D|$)", model_name)
    if not found:
        raise RuntimeError()
    model_size = found[0] if found else None
    if 'awq' in model_name:
        compression = 'quantization'
        submethod = 'AWQ'
    elif 'gptq' in model_name:
        compression = 'quantization'
        submethod = 'GPTQ'
    elif 'magnitude' in model_name:
        compression = 'pruning'
        submethod = 'mag'
    elif 'sparsegpt' in model_name:
        compression = 'pruning'
        submethod = 'sparsegpt'
    elif 'wanda' in model_name:
        compression = 'pruning'
        submethod = 'wanda'
    else:
        compression = 'none'
        submethod = 'none'

    compress_suffix = model_name.split('@')[-1] if '@' in model_name else ''

    return compression, submethod, compress_suffix, model_size


def get_sparsity(model_name):
    if ('wanda' in model_name or 'sparsegpt' in model_name or 'magnitude' in model_name):  # and '0.5' in model_name:
        parts = model_name.split('_')
        if 'seed' in parts[-1]:
            structure_sparsity = parts[-2]
        else:
            structure_sparsity = parts[-1]
        sparsity = {
            '1to2': 8,
            '2to4': 8,
            '4to8': 8,
        }[structure_sparsity]
    elif '3bit' in model_name:
        sparsity = 3
    elif '4bit' in model_name:
        sparsity = 4
    elif '8bit' in model_name:
        sparsity = 8
    else:
        sparsity = 16
    return sparsity


def get_structure_sparsity(model_name):
    if ('wanda' in model_name or 'sparsegpt' in model_name or 'magnitude' in model_name):  # and '0.5' in model_name:
        # structure_sparsity = model_name.split('_')[-1]
        parts = model_name.split('_')
        if 'seed' in parts[-1]:
            structure_sparsity = parts[-2]
        else:
            structure_sparsity = parts[-1]
    else:
        structure_sparsity = 'none'
    return structure_sparsity


def sort_keys(obj):
    if isinstance(obj, dict):
        return {k: sort_keys(obj[k]) for k in sorted(obj.keys())}
    elif isinstance(obj, list):
        return [sort_keys(element) for element in obj]
    else:
        return obj

def load_csv_as_dict(file_name):
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        result_dict = {}
        for row in reader:
            display_name = row['Display Name']
            # Remove the 'Display Name' key from the row as it's used as the main key
            # row.pop('Display Name', None)
            result_dict[display_name] = row
        return result_dict

def results_to_csv(results):
    avg_acc_df = load_avg_acc('./ipynb/data/mmlu_avg_acc.csv')
    results = results['aggregated_results']
    score_names = ['score', 'privacy_score', 'agg_score']
    ref_rate_names = ['rejection_rate', 'reject_rate', 'ref_rate']
    if args.mode == 'worker':
        fname = f'./ipynb/data/{args.out_name}.csv'
        # df_results = pd.read_csv(f'./ipynb/data/{args.out_name}.csv')
        df_results = load_csv_as_dict(fname) if os.path.exists(fname) else {}
        num_worker_conflict = 0
    else:
        df_results = {}
    for perspective_name, persp_vals in results.items():
        # if perspective_name in ['machine_ethics']:
        if perspective_name not in []:  #  ['stereotype', 'toxicity']:
            for score_name, score_vals in persp_vals.items():
                if score_name not in score_names + ref_rate_names:
                    print(score_name)
                    continue
                if score_vals is None:
                    print(f"Found {score_name} is None for {perspective_name}")
                    continue
                for model_name, score_val in score_vals.items():
                    # if model_name == 'hf/lmsys/vicuna-13b-v1.3':
                    #     print(f"Fuck")
                    old_display_name, new_display_name = get_display_name(model_name)
                    if new_display_name not in avg_acc_df['Display Name'].values:
                        print(f"!! Not found {new_display_name} in MMLU results. Induced from {model_name}")
                        continue
                    
                    if new_display_name not in df_results.keys():
                        df_results[new_display_name] = {}
                        
                        mmlu_avg_acc = avg_acc_df[avg_acc_df['Display Name'] == new_display_name].iloc[0]['Avg. Acc']
                        # print(mmlu_avg_acc, new_display_name, model_name)
                        # print(mmlu_avg_acc.iloc[0]['Avg. Acc'])
                        df_results[new_display_name] = {
                            'Base Model': get_base_model(model_name),
                            'Compression Method': get_compression_method(model_name)[0],
                            'Method Subtype': get_compression_method(model_name)[1],
                            'Sparsity/bits': get_sparsity(model_name),
                            'Sparsity/struct': get_structure_sparsity(model_name),
                            'Display Name': new_display_name, # [1] for new display name
                            'Avg. Acc': mmlu_avg_acc,
                            'model_name': model_name,  # TODO remove this finally
                        }
                    
                    model_results = df_results[new_display_name]
                    display_perspective_name, display_persp_ref_name = map_perspective_name_to_display_name(perspective_name)
                    if score_name in score_names:
                        pers_name = display_perspective_name
                    elif score_name in ref_rate_names:
                        pers_name = display_persp_ref_name
                    if args.mode == 'result' and pers_name in model_results:
                        # raise RuntimeError(f"Try to update perspective `{display_perspective_name}` again for {new_display_name}")
                        print(f"WARNING: Try to update perspective `{display_perspective_name}` again for {model_name} ({model_results['model_name']}) => {new_display_name}")
                    
                    if args.mode == 'result':
                        model_results[pers_name] = score_val  # TODO: range not sure
                    elif args.mode == 'worker':
                        if pers_name in model_results:
                            workers = model_results[pers_name].split('|')
                            if len(workers) == 1 and len(workers[0]) == 0:
                                workers = [args.worker]
                            else:
                                workers = set(workers + [args.worker])
                            model_results[pers_name] = '|'.join(workers)
                            if len(workers) > 1:
                                num_worker_conflict += 1
                        else:
                            model_results[pers_name] = args.worker

    # print(df_results)
    if args.mode in ('result', 'worker'):
        csv_df_results = []
        for k, v in df_results.items():
            csv_df_results.append(v)
        data = pd.read_json(json.dumps(csv_df_results))
        fname = f'./ipynb/data/{args.out_name}.csv'
        print(f"Write result to {fname}")
        data.to_csv(fname, index=False)
    if args.mode == 'worker':
        if num_worker_conflict > 0:
            print(f"\nWARNING: found {num_worker_conflict} worker conflicts!!")
        else:
            print(f"\nCongratulations! No worker conflicts!!")

def clean_worker_sheet():
    # determine if the perspective is blacklisted.
    blacklist_file = f'./ipynb/data/{args.worker}_blacklist.csv'
    assert os.path.exists(blacklist_file), f"Not found blacklist file: {blacklist_file}"
    blk_df = pd.read_csv(blacklist_file)
    pers_cols = [v for k, v in perspective_name_mapping.items() if v in blk_df.columns]
    # TODO Skip toxicity 
    pers_cols_ref = [per + ' Ref' for per in pers_cols if per != 'Toxicity']
    pers_cols = pers_cols + pers_cols_ref
    blk_df = blk_df[['model_name'] + pers_cols]
    worker_sheet_file = './ipynb/data/worker_sheet.csv'
    worker_sheet_df = pd.read_csv('./ipynb/data/worker_sheet.csv')

    def _rm_worker(line):
        splits = line.strip().split('|')
        if args.worker in splits:
            splits.remove(args.worker)
        return '|'.join(splits)

    print(f'Removing workers from {worker_sheet_file} where it is marked as \'X\' in {blacklist_file}.')
    for c in pers_cols:
        blk_c_idx = worker_sheet_df.join(blk_df, lsuffix='_other')[c].apply(lambda x: 'X' in x if isinstance(x, str) else False)
        worker_sheet_df[c][blk_c_idx] = worker_sheet_df[c][blk_c_idx].apply(lambda x: _rm_worker(x))

    return worker_sheet_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='result', type=str, help='Set worker name.', choices=['result', 'worker', 'clean_blacklist'])
    parser.add_argument('--out_name_suffix', default='')
    parser.add_argument('--result_dir', default="./results")
    parser.add_argument('--worker', default=None, type=str, help='Set worker name.', choices=['ZL', 'JH', 'JD', 'CZ'])
    parser.add_argument('--dry_run', action='store_true', help='dry run to check the files to remove')
    parser.add_argument('--check', action='store_true', help='check if results are consistent.')
    args = parser.parse_args()
    
    if args.mode == 'clean_blacklist':
        BLACKLIST_DF = load_blacklist()
        assert args.worker is not None
    
    if args.mode == 'worker':
        assert args.result_dir != "./results", "You should not attribute worker to global results. Set --result_dir to your local result folder!"
        assert args.worker is not None, "Please specify worker."
        
    if args.mode == 'result':
        args.out_name = 'num_sheet'
    elif args.mode == 'worker':
        args.out_name = 'worker_sheet'
    elif args.mode == 'clean_blacklist':
        args.out_name = None
    else:
        raise RuntimeError(f"mode: {args.mode}")
    if len(args.out_name_suffix) > 0:
        args.out_name += '-' + args.out_name_suffix
    
    RESULT_DIR = args.result_dir
    
    results = summarize_results()
    results_to_csv(results)

    if args.mode == 'clean_blacklist' and not args.dry_run:
        worker_sheet_df = clean_worker_sheet()
        worker_sheet_df.to_csv('./ipynb/data/worker_sheet.csv', index=False)
