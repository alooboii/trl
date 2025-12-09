import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import datasets
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from trl import (
    DPOTrainer,
    GRPOConfig,
    GRPOTrainer,
    PPOConfig,
    PPOTrainer,
)

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class CommonConfig:
    model_name_or_path: str
    dataset_name: str
    dataset_split: str = "train"
    prompt_column: str = "prompt"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"
    max_prompt_length: int = 512
    max_response_length: int = 256
    output_dir: str = "outputs"
    log_with: Optional[str] = None
    load_in_8bit: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass
class DPOConfig(CommonConfig):
    beta: float = 0.1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    num_train_epochs: int = 1


@dataclass
class PPORewardConfig:
    model_name_or_path: str
    learning_rate: float = 5e-6
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2


@dataclass
class PPOFullConfig(CommonConfig):
    reward_config: PPORewardConfig = PPORewardConfig(model_name_or_path="")
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    num_train_epochs: int = 1
    ppo_batch_size: int = 8
    target_kl: float = 0.1
    dense_rewards: bool = False
    generation_samples: int = 64
    max_ppo_steps: int = 128


@dataclass
class GRPOFullConfig(CommonConfig):
    group_size: int = 4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    num_train_epochs: int = 1
    generation_samples: int = 64


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def init_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def maybe_bnb_config(enabled: bool) -> Optional[BitsAndBytesConfig]:
    if not enabled:
        return None
    return BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)


def maybe_apply_lora(model, cfg: CommonConfig, task_type: str = "CAUSAL_LM"):
    if not getattr(cfg, "use_lora", False):
        return model
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type=task_type,
    )
    return get_peft_model(model, lora_cfg)


def format_pair(example: Dict[str, Any], config: CommonConfig, tokenizer: AutoTokenizer):
    prompt = example[config.prompt_column]
    chosen = example[config.chosen_column]
    rejected = example[config.rejected_column]
    prompt_ids = tokenizer(prompt, truncation=True, max_length=config.max_prompt_length)
    chosen_ids = tokenizer(chosen, truncation=True, max_length=config.max_response_length)
    rejected_ids = tokenizer(rejected, truncation=True, max_length=config.max_response_length)
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "prompt_ids": prompt_ids,
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
    }


def load_pair_dataset(config: CommonConfig, tokenizer: AutoTokenizer):
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    return dataset.map(lambda x: format_pair(x, config, tokenizer))


def save_log_history(trainer, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(output_dir, "log_history.jsonl")
    with open(history_path, "w", encoding="utf-8") as f:
        for record in trainer.state.log_history:
            f.write(json.dumps(record) + "\n")
    LOG.info("Saved trainer log history to %s", history_path)


def train_dpo(config_dict: Dict[str, Any]):
    cfg = DPOConfig(**config_dict)
    tokenizer = init_tokenizer(cfg.model_name_or_path)
    dataset = load_pair_dataset(cfg, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        device_map="auto",
        quantization_config=maybe_bnb_config(cfg.load_in_8bit),
    )
    model = maybe_apply_lora(model, cfg)
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        device_map="auto",
        quantization_config=maybe_bnb_config(cfg.load_in_8bit),
    )

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=5,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to=[cfg.log_with] if cfg.log_with else [],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=cfg.beta,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    save_log_history(trainer, cfg.output_dir)
    return cfg.output_dir


def build_sequence_dataset(dataset: datasets.Dataset, cfg: CommonConfig, tokenizer: AutoTokenizer):
    def _map(ex):
        prompt = ex[cfg.prompt_column]
        chosen = ex[cfg.chosen_column]
        rejected = ex[cfg.rejected_column]
        chosen_text = f"{prompt}\n{chosen}"
        rejected_text = f"{prompt}\n{rejected}"
        enc_chosen = tokenizer(chosen_text, truncation=True, max_length=cfg.max_prompt_length + cfg.max_response_length)
        enc_rejected = tokenizer(rejected_text, truncation=True, max_length=cfg.max_prompt_length + cfg.max_response_length)
        return {
            "chosen_input_ids": enc_chosen["input_ids"],
            "chosen_attention_mask": enc_chosen["attention_mask"],
            "rejected_input_ids": enc_rejected["input_ids"],
            "rejected_attention_mask": enc_rejected["attention_mask"],
        }

    columns = [cfg.prompt_column, cfg.chosen_column, cfg.rejected_column]
    return dataset.select_columns(columns).map(_map)


def train_reward_model(dataset: datasets.Dataset, cfg: CommonConfig, reward_cfg: PPORewardConfig, tokenizer):
    LOG.info("Training reward model %s", reward_cfg.model_name_or_path or cfg.model_name_or_path)
    model_id = reward_cfg.model_name_or_path or cfg.model_name_or_path
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=1)
    model = maybe_apply_lora(model, cfg, task_type="SEQ_CLS")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def _stack(ex):
        chosen = {
            "input_ids": ex["chosen_input_ids"],
            "attention_mask": ex["chosen_attention_mask"],
            "labels": 1.0,
        }
        rejected = {
            "input_ids": ex["rejected_input_ids"],
            "attention_mask": ex["rejected_attention_mask"],
            "labels": 0.0,
        }
        return {"pair": [chosen, rejected]}

    # Flatten pairs into a single supervised dataset
    flat_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    for row in dataset:
        pair = _stack(row)["pair"]
        for item in pair:
            flat_inputs["input_ids"].append(item["input_ids"])
            flat_inputs["attention_mask"].append(item["attention_mask"])
            flat_inputs["labels"].append(item["labels"])
    reward_ds = datasets.Dataset.from_dict(flat_inputs)

    training_args = TrainingArguments(
        output_dir=os.path.join(cfg.output_dir, "reward_model"),
        per_device_train_batch_size=reward_cfg.per_device_train_batch_size,
        learning_rate=reward_cfg.learning_rate,
        num_train_epochs=reward_cfg.num_train_epochs,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=reward_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    save_log_history(trainer, training_args.output_dir)
    return training_args.output_dir


def score_reward(model, tokenizer, prompts: List[str], responses: List[str]):
    text_batch = [f"{p}\n{r}" for p, r in zip(prompts, responses)]
    encoded = tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**encoded)
        rewards = outputs.logits.squeeze(-1).detach().cpu()
    return rewards


def make_generation_prompts(dataset: datasets.Dataset, cfg: CommonConfig):
    prompts = []
    for row in dataset.shuffle(seed=42).select(range(min(len(dataset), cfg.max_ppo_steps))):
        prompts.append(row[cfg.prompt_column])
    return prompts


def train_ppo(config_dict: Dict[str, Any]):
    cfg = PPOFullConfig(**config_dict)
    tokenizer = init_tokenizer(cfg.model_name_or_path)
    pair_ds = load_pair_dataset(cfg, tokenizer)
    seq_ds = build_sequence_dataset(pair_ds, cfg, tokenizer)
    reward_dir = train_reward_model(seq_ds, cfg, cfg.reward_config, tokenizer)
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_dir).to("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        device_map="auto",
        quantization_config=maybe_bnb_config(cfg.load_in_8bit),
    )
    model = maybe_apply_lora(model, cfg)
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        device_map="auto",
        quantization_config=maybe_bnb_config(cfg.load_in_8bit),
    )

    ppo_config = PPOConfig(
        model_name=cfg.model_name_or_path,
        learning_rate=cfg.learning_rate,
        mini_batch_size=cfg.per_device_train_batch_size,
        batch_size=cfg.ppo_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        target_kl=cfg.target_kl,
        log_with=cfg.log_with,
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=None,
    )

    prompts = make_generation_prompts(pair_ds, cfg)
    device = trainer.accelerator.device
    reward_model.to(device)

    all_metrics = []
    for step in range(min(cfg.max_ppo_steps, len(prompts))):
        prompt = prompts[step]
        query_tensors = tokenizer(prompt, return_tensors="pt").to(device)
        response_tensors = trainer.generate(query_tensors["input_ids"], max_new_tokens=cfg.max_response_length)
        responses = tokenizer.batch_decode(response_tensors[:, query_tensors["input_ids"].shape[-1] :], skip_special_tokens=True)
        rewards = score_reward(reward_model, tokenizer, [prompt], responses)
        if cfg.dense_rewards:
            # repeat reward for every token
            dense = []
            for resp_ids, reward in zip(response_tensors, rewards):
                length = resp_ids.shape[-1] - query_tensors["input_ids"].shape[-1]
                dense.append([float(reward) / max(length, 1)] * length)
            ppo_rewards = dense
        else:
            ppo_rewards = [float(r) for r in rewards]
        stats = trainer.step(query_tensors["input_ids"], response_tensors, ppo_rewards)
        all_metrics.append({"step": step, **{k: float(v) for k, v in stats.items()}})
        if (step + 1) % 10 == 0:
            trainer.save_pretrained(os.path.join(cfg.output_dir, f"ppo_step_{step+1}"))
    trainer.save_pretrained(cfg.output_dir)

    metrics_path = os.path.join(cfg.output_dir, "ppo_metrics.jsonl")
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        for record in all_metrics:
            f.write(json.dumps(record) + "\n")
    LOG.info("Saved PPO metrics to %s", metrics_path)
    return cfg.output_dir


def train_grpo(config_dict: Dict[str, Any]):
    cfg = GRPOFullConfig(**config_dict)
    tokenizer = init_tokenizer(cfg.model_name_or_path)
    pair_ds = load_pair_dataset(cfg, tokenizer)

    reward_dir = os.path.join(cfg.output_dir, "reward_model")
    if not os.path.exists(reward_dir):
        LOG.warning("Reward model not found in %s; please train via PPO first.", reward_dir)
        reward_dir = None
    reward_model = None
    if reward_dir:
        reward_model = AutoModelForSequenceClassification.from_pretrained(reward_dir)
        reward_model = reward_model.to("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        device_map="auto",
        quantization_config=maybe_bnb_config(cfg.load_in_8bit),
    )
    model = maybe_apply_lora(model, cfg)

    grpo_config = GRPOConfig(
        model_name=cfg.model_name_or_path,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        log_with=cfg.log_with,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_response_length,
        num_generations=cfg.group_size,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_model=reward_model,
        args=grpo_config,
        tokenizer=tokenizer,
        train_dataset=pair_ds,
    )

    trainer.train()
    trainer.save_pretrained(cfg.output_dir)
    save_log_history(trainer, cfg.output_dir)
    return cfg.output_dir


def evaluate_perplexity(model_path: str, tokenizer, dataset: datasets.Dataset, cfg: CommonConfig):
    model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
    losses = []
    for row in dataset.select(range(min(len(dataset), 64))):
        text = f"{row[cfg.prompt_column]}\n{row[cfg.chosen_column]}"
        encoded = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**encoded, labels=encoded["input_ids"])
            loss = outputs.loss.item()
            losses.append(loss)
    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss)
    return ppl


def analyze_verbosity(model_path: str, tokenizer, prompts: List[str], word_limit: Optional[int] = None):
    model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
    lengths = []
    compliance = []
    for prompt in prompts:
        if word_limit:
            prompt = f"{prompt}\nPlease answer in {word_limit} words or less."
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=128)
        text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        token_count = len(tokenizer.encode(text))
        lengths.append(token_count)
        if word_limit:
            compliance.append(int(token_count <= word_limit))
    stats = {
        "mean_tokens": float(torch.tensor(lengths).mean()),
        "median_tokens": float(torch.median(torch.tensor(lengths)).item()),
        "std_tokens": float(torch.tensor(lengths, dtype=torch.float32).std()),
        "compliance_rate": float(sum(compliance) / len(compliance)) if compliance else None,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Alignment runners for DPO, PPO, and GRPO")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--method", type=str, choices=["dpo", "ppo", "grpo"], required=True)
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    os.makedirs(cfg_dict.get("output_dir", "outputs"), exist_ok=True)

    if args.method == "dpo":
        model_path = train_dpo(cfg_dict)
    elif args.method == "ppo":
        model_path = train_ppo(cfg_dict)
    else:
        model_path = train_grpo(cfg_dict)

    # Optional evaluation if a test prompt file is provided
    test_file = cfg_dict.get("test_prompts_file")
    tokenizer = init_tokenizer(cfg_dict["model_name_or_path"])
    pair_ds = load_dataset(cfg_dict["dataset_name"], split=cfg_dict.get("dataset_split", "train"))
    if test_file and os.path.exists(test_file):
        with open(test_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [row[cfg_dict.get("prompt_column", "prompt")] for row in pair_ds.select(range(min(50, len(pair_ds))))]

    verbosity_stats = analyze_verbosity(model_path, tokenizer, prompts, word_limit=cfg_dict.get("verbosity_word_limit"))
    ppl = evaluate_perplexity(model_path, tokenizer, pair_ds, CommonConfig(**{k: cfg_dict[k] for k in CommonConfig.__annotations__ if k in cfg_dict}))

    eval_path = os.path.join(cfg_dict.get("output_dir", "outputs"), "evaluation.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump({"perplexity": ppl, "verbosity": verbosity_stats}, f, indent=2)
    LOG.info("Saved evaluation metrics to %s", eval_path)


if __name__ == "__main__":
    main()
