import os
import json
import pandas as pd
from glob import glob
import shutil
import argparse


def copyfile(src, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy(src, dest)

def merge_nested_dicts(dict1, dict2):
    """
    Merge two nested dictionaries into one.

    Args:
    dict1 (dict): First dictionary.
    dict2 (dict): Second dictionary. Use this value if conflict

    Returns:
    dict: Merged dictionary.
    """
    merged_dict = {**dict1}  # Start with dict1's keys and values

    for key, value in dict2.items():
        if key in merged_dict:
            # If the key is present in both dictionaries and both values are dictionaries, merge them
            if isinstance(merged_dict[key], dict) and isinstance(value, dict):
                merged_dict[key] = merge_nested_dicts(merged_dict[key], value)
            else:
                # If the key is present but the values aren't both dictionaries, use the value from dict2
                merged_dict[key] = value
        else:
            # If the key is not present in dict1, add it to the merged dictionary
            merged_dict[key] = value

    return merged_dict

def merge_dataframe(df1, df2, index):
    # df1 = df1.set_index()
    
    # Append DataFrames
    appended_df = pd.concat([df1, df2], ignore_index=True)

    # Drop duplicates, keep last (from df2)
    appended_df = appended_df.drop_duplicates(subset=index, keep='last')
    appended_df = appended_df.sort_values(by=index)

    # Set index back (if it was reset)
    # appended_df = appended_df.set_index(index)
    return appended_df.reset_index(drop=True)

def get_adv_demo_scores():
    fs = glob(os.path.join(RESULT_DIR, "adv_demonstration", "**", "*_score.json"), recursive=True)
    trg_fs = [f.replace(RESULT_DIR, GIT_RESULT_DIR) for f in fs]
    for f, tf in zip(fs, trg_fs):
        copyfile(f, tf)

def copy_or_merge_json_dict(relative_path):
    f = os.path.join(RESULT_DIR, relative_path)
    tf = os.path.join(GIT_RESULT_DIR, relative_path)
    if os.path.exists(tf):
        # print(f"\nERROR!!! File exist, need manual merge content: {tf}\n")
        # return
        print(f"\nFile exist, will merge content: {f} and {tf}\n")
        src_scores = pd.read_json(f).to_dict()
        with open(tf) as open_f:
            dst_scores = json.load(open_f)
        scores = merge_nested_dicts(src_scores, dst_scores)
        with open(tf, 'w') as open_f:
            json.dump(scores, open_f, indent=4)
    else:
        copyfile(f, tf)

def get_advglue_scores():
    # print(os.path.join(RESULT_DIR, "adv-glue-plus-plus", "summary.json"))
    copy_or_merge_json_dict(
        os.path.join("adv-glue-plus-plus", "summary.json")
    )
    copy_or_merge_json_dict(
        os.path.join("adv-glue-plus-plus", "breakdown.json")
    )


def get_fairness_scores():
    fs = glob(os.path.join(RESULT_DIR, "fairness", "**", "final_scores.json"), recursive=True)
    model_names = [
        os.path.dirname(x).removeprefix(os.path.join(RESULT_DIR, "fairness", "results")).removeprefix("/") for x in fs
    ]
    for f, model_name in zip(fs, model_names):
        tf = f.replace(RESULT_DIR, GIT_RESULT_DIR)
        copyfile(f, tf)


def get_ethics_scores():
    f = os.path.join(RESULT_DIR, "machine_ethics", "generations", "scores.jsonl")
    tf = os.path.join(GIT_RESULT_DIR, "machine_ethics", "generations", "scores.jsonl")
    if os.path.exists(tf):
        # print(f"\nERROR!!! File exist, need manual merge content: {tf}\n")
        # return
        print(f"ETHICS: File exist, merge content: {tf}\n")
        df = pd.read_json(f, lines=True)
        t_df = pd.read_json(tf, lines=True)
        t_df = merge_dataframe(t_df, df, ['model', 'dataset'])
        # print(t_df)
        t_df.to_json(tf, orient='records', lines=True)
        # copyfile(f, tf)
    else:
        copyfile(f, tf)


def get_ood_scores():
    path_prefix = os.path.join(RESULT_DIR, "ood", "results/")
    fs = glob(os.path.join(path_prefix, "**", "final_scores.json"), recursive=True)
    model_names = [os.path.dirname(f).removeprefix(path_prefix) for f in fs]
    for f, model_name in zip(fs, model_names):
        tf = f.replace(RESULT_DIR, GIT_RESULT_DIR)
        copyfile(f, tf)


def get_privacy_scores():
    f = os.path.join(RESULT_DIR, "privacy", "generations", "scores.jsonl")
    tf = os.path.join(GIT_RESULT_DIR, "privacy", "generations", "scores.jsonl")
    if os.path.exists(tf):
        print(f"\nPRIVACY: File exist, merge content: {tf}\n")
        df = pd.read_json(f, lines=True)
        t_df = pd.read_json(tf, lines=True)
        t_df = merge_dataframe(t_df, df, ['model', 'dataset'])
        # print(t_df)
        t_df.to_json(tf, orient='records', lines=True)
    else:
        copyfile(f, tf)


def get_stereotype_scores():
    path_prefix = os.path.join(RESULT_DIR, "stereotype", "generations/")
    fs = glob(os.path.join(path_prefix, "**", "25_compiled.json"), recursive=True)
    model_names = [os.path.dirname(f).removeprefix(path_prefix) for f in fs]
    for f, model_name in zip(fs, model_names):
        tf = f.replace(RESULT_DIR, GIT_RESULT_DIR)
        copyfile(f, tf)


def get_toxicity_scores():
    path_prefix = os.path.join(RESULT_DIR, "toxicity", "user_prompts", "generations/")
    fs = glob(os.path.join(path_prefix, "**", "report.jsonl"), recursive=True)
    model_names = [os.path.dirname(f).removeprefix(path_prefix) for f in fs]
    for f, model_name in zip(fs, model_names):
        tf = f.replace(RESULT_DIR, GIT_RESULT_DIR)
        copyfile(f, tf)


def summarize_results(keys=None):
    func_dict = {
        "adv_demo": get_adv_demo_scores,
        "adv-glue": get_advglue_scores,
        "fairness": get_fairness_scores,
        "ethics": get_ethics_scores,
        "ood": get_ood_scores,
        "privacy": get_privacy_scores,
        "stereotype": get_stereotype_scores,
        "toxicity": get_toxicity_scores
    }
    if keys is None:
        keys = list(func_dict)
    for k in keys:
        print(f">> {k}")
        func_dict[k]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', default="../DecodingTrust/results")
    parser.add_argument('-p', '--perspective', default=None, type=str)
    parser.add_argument('--git_result_dir', default="./results")
    args = parser.parse_args()
    
    assert os.path.exists(args.result_dir), f"Not found path to source results: {args.result_dir}"

    RESULT_DIR = args.result_dir  # type: str
    while RESULT_DIR.endswith('/'):
        RESULT_DIR = RESULT_DIR[:-1]
    # GIT_RESULT_DIR = "./results"
    GIT_RESULT_DIR = args.git_result_dir

    summarize_results(keys=[args.perspective] if args.perspective is not None else None)
