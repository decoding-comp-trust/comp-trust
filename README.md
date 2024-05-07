# Decoding Compressed Trust

Codebase for the [Decoding Compressed Trust](https://decoding-comp-trust.github.io/).

## Model Preparation

We provide compressed models at [huggingface](https://huggingface.co/compressed-llm). Details for compressing models are provided here.

### Pruning

Our code is based on `git@github.com:locuslab/wanda.git`.
```bash
cd compression
git clone git@github.com:locuslab/wanda.git
```

Pruning Magnitude/SparseGPT/Wanda with semi-structured sparsity:
```bash
cd wanda
CUDA_VISIBLE_DEVICES=0 python main.py --model meta-llama/Llama-2-13b-chat-hf --prune_method magnitude --sparsity_type 2:4 --sparsity_ratio 0.5 --save=output/llama-2-13b-chat_mag_2to4
CUDA_VISIBLE_DEVICES=0 python main.py --model meta-llama/Llama-2-13b-chat-hf --prune_method sparsegpt --sparsity_type 2:4 --sparsity_ratio 0.5 --save=output/llama-2-13b-chat_sparsegpt_2to4
CUDA_VISIBLE_DEVICES=2 python main.py --model meta-llama/Llama-2-13b-chat-hf --prune_method wanda --sparsity_type 2:4 --sparsity_ratio 0.5 --save=output/llama-2-13b-chat_wanda_2to4
```
Change `meta-llama/Llama-2-13b-chat-hf` to other models upon demands.

### Quantization

GPTQ:
```bash
pip install auto-gptq
cd compression/gptq

CUDA_VISIBLE_DEVICES=0 python gptq.py --pretrained_model_dir meta-llama/Llama-2-13b-chat-hf --quantized_model_dir ./output --bits 4 --save_and_reload --desc_act --seed 0 --num_samples 128 --calibration-template llama-2
```
AWQ:
```bash
cd compression
git clone https://github.com/mit-han-lab/llm-awq
cd llm-awq

mkdir -p /storage/jinhaoduan/workspace/llm-awq-main/experiments/llama-2-13b-chat-bit4-seed0
CUDA_VISIBLE_DEVICES=1 python -m awq.entry --model_path meta-llama/Llama-2-13b-chat-hf --seed 0 --w_bit 4 --q_group_size 128 --run_awq --dump_awq awq_cache/llama-2-13b-chat-bit4-seed0.pt
CUDA_VISIBLE_DEVICES=1 python -m awq.entry --model_path meta-llama/Llama-2-13b-chat-hf --tasks wikitext --w_bit 4 --q_group_size 128 --load_awq awq_cache/llama-2-13b-chat-bit4-seed0.pt --q_backend fake --dump_awq_weights_to_hf ./llm-awq-main/llama-2-13b-chat-bit4-seed0
```

## Running Experiments

Install the modified DecodingTrust following [this link](https://github.com/decoding-comp-trust/DecodingTrust?tab=readme-ov-file#getting-started).

Due to the large volume of experiments, we recommend to run experiments using the Slurm job system.
We provide [a example of slurm config file](configs/slurm_config.yaml).
For each model, we provide a config file under [configs/model_config](configs/model_config).

> Note these files are tuned for VITA ACES servers and may not work on other servers.

Important files
* `scripts/multi-run.sh`: Use this to run all metrics on a given model_config
* `dt/configs/model_configs/vicuna_xxx.yaml`: This file is to config model
* `dt/configs/slurm_config.yaml`: To setup slurm, do not change this.

Setup
```shell
# find the gpu type
scontrol show node | grep Gres
# Add slurm
cd DecodingTrust
pip install -e ".[slurm]"
```

Modify `dt/configs/model_configs/vicuna-13b-v1.3-mag_2to4.yaml` for your model.
Add `vicuna-13b-v1.3-mag_2to4` to multi-run.sh
```shell
bash scripts/multi-run.sh
```

## Aggregating Results


Upload results to github
```shell
git pull
python gather_result_files.py --result_dir=<path-to-DT-result-folder> -p=<perspective_name>
# Example
# python gather_result_files.py -p=adv-glue
git add results/
git commit -m "Update results"
git push
```

Example:
```shell
git pull
python gather_result_files.py -p=adv-glue
git add results/
git commit -m "Update results"
git push
```

Extract results to csv file (data/num_sheet.csv) which will be used for visualization.
Run `python extract_csv.py`.


* Adversarial Demonstrations

  ```bash
  python src/dt/perspectives/adv_demonstration/aggregate_score.py
  ```

  Find aggregated results with the following patterns.

  ```bash
  #ls results/adv_demonstration/*_score.json
  grep -H "adv_demonstration\"" results/adv_demonstration/*_score.json
  ```

* Adversarial Robustness

  ```bash
  python src/dt/perspectives/advglue/adv_stats.py
  ```

  You can find the scores with the following patterns.

  ```bash
  cat ./results/adv-glue-plus-plus/summary.json | jq
  ```

* Fairness

  Use patch score calculation:
  ```shell
  cp dt-patch/src/dt/perspectives/fairness/score_calculation_script.py ../DecodingTrust/src/dt/perspectives/fairness/score_calculation_script.py
  ```

  ```bash
  python src/dt/perspectives/fairness/score_calculation_script.py
  ```

  ```bash
  #ls results/fairness/results/*/*/*/final_scores.json
  grep -H "fairness score\"" results/fairness/results/hf/compressed-llm/*/final_scores.json
  ```

* Machine Ethics

  Use patch score calculation:
  ```shell
  cp dt-patch/src/dt/perspectives/machine_ethics/calculate_score.py ../DecodingTrust/src/dt/perspectives/machine_ethics/calculate_score.py
  ```

  ```bash
  python src/dt/perspectives/machine_ethics/calculate_score.py
  ```

  ```bash
  # cat data/machine_ethics/generations/scores.jsonl | jq
  cat results/machine_ethics/generations/scores.jsonl | jq
  ```

* OOD

  Use patch score calculation:
  ```shell
  cp dt-patch/src/dt/perspectives/ood/evaluation_ood.py ../DecodingTrust/src/dt/perspectives/ood/evaluation_ood.py
  ```

  Find aggregated OOD results with the following patterns.

  ```bash
  #ls results/ood/results/*/*/*/final_scores.json
  #cat results/ood/results/hf/vita-group/llama-2-13b-chat_magnitude_semistruct@0.5_2to4/final_scores.json | grep "\"score"
  grep -H "\"score" results/ood/results/*/*/*/final_scores.json
  ```
  
  Note: To rerun score aggregation, use the following command.

  ```bash
  dt-run --config-name config --multirun +model_config=<YOUR MODEL LIST, EXAMPE: model_a,model_b> +ood=knowledge_2020_5shot,knowledge_idk,knowledge_standard,style_8shot,style ++ood.resume=True,True
  ```


* Privacy

  To rerun score aggregation, use the following command.

  ```bash
  python src/dt/perspectives/privacy/result_agg.py
  ```

  Find aggregated privacy results with the following patterns.

  ```bash
  cat results/privacy/generations/scores.jsonl | grep "all" | jq
  ```

* Stereotype

  Use patch score calculation:
  ```shell
  cp dt-patch/src/dt/perspectives/stereotype/agreement_func.py ../DecodingTrust/src/dt/perspectives/stereotype/agreement_func.py
  ```
  Main difference:
  1. Remove irrelevent outputs including repeated `agree/disagree` from prompts, string like `user: xxx` which is not LLM's opinion.
  2. Match the first opinion if there are multiple ones.

  To rerun stereotype aggregation, use the following command.

  ```bash
  dt-run --config-name config --multirun +model_config=$(ls src/dt/configs/model_config/*.yaml | xargs -n 1 basename | sed 's/.yaml//' | tr '\n' ',' | sed 's/,$//') +stereotype=benign,targeted,untargeted ++stereotype.skip_generation=True,True 
  ```

  Find aggregated stereotype results with the following patterns.

  ```bash
  ls results/stereotype/generations/*/*/*/*compiled.json

* Toxicity

  To calculate the toxicity scores, use the following command.
  
  ```bash
  python src/dt/perspectives/toxicity/perspective_api_evaluate.py --api AIzaSyDK58omxWcBQa-o6_V53uh3gk1ShJU-n08 --strip-outputs
  ```

  If you are rerunning the evaluation, remember to add `--strip-outputs`.
  
  Find aggregated toxicity results with the following patterns.
  
  ```bash
  cat results/toxicity/user_prompts/generations/*/*/*/report.jsonl | jq
  ```
  
+ Score Summary

  ```bash
  dt-run +model_config=hf
  ```

  or

  ```bash
  python src/dt/summarize.py
  ```

  Then check the final `Json` file

  ```bash
  cat results/sumamry.json | jq
  ```

