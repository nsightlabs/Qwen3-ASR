# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import librosa
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from qwen_asr import Qwen3ASRModel
from transformers import (GenerationConfig, Trainer, TrainerCallback,
                          TrainingArguments)
import json
from collections import Counter
from typing import List, Tuple, Optional


def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
            "Your qwen3_asr model may be incompatible."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn_prefix_only(processor):
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        dummy_audio = None
        prefix_msgs = build_prefix_messages(prompt, dummy_audio)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "prompt": prompt,
            "audio": ex["audio"],
            "target": ex["text"],
            "prefix_text": prefix_text,
        }

    return _preprocess


@dataclass
class DataCollatorForQwen3ASRFinetuning:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]

        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs


class CastFloatInputsTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs


def copy_required_hf_files_for_qwen_asr(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    required = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        # "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
    ]
    for fn in required:
        if not os.path.exists(os.path.join(dst_dir, fn)):
            cached_path = hf_hub_download(repo_id=src_dir, filename=fn)
            shutil.copy(cached_path, os.path.join(dst_dir, fn))            


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        copy_required_hf_files_for_qwen_asr(self.base_model_path, ckpt_dir)
        return control
    
def extract_new_tokens(
    train_file: str,
    tokenizer,
    min_freq: int = 5,
    max_fragments_ok: int = 1,
    top_k: int = 500,
) -> List[Tuple[str, int, int]]:
    """
    Scan a jsonl train file and find whole words that the tokenizer currently
    splits into more than `max_fragments_ok` subword pieces, ranked by how
    often they appear (frequency-weighted "impact" of fixing each one).
 
    Returns a list of (word, frequency, n_pieces_before) tuples, most
    impactful first.
    """
    word_counter = Counter()
    with open(train_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec["text"].split("<asr_text>")[1].strip()
            if not text:
                continue
            for word in text.split():
                word_counter[word] += 1
 
    candidates = []
    for word, freq in word_counter.items():
        if freq < min_freq:
            continue
        n_pieces = len(tokenizer.tokenize(word))
        if n_pieces > max_fragments_ok:
            candidates.append((word, freq, n_pieces))
 
    # Rank by frequency: fixing a word that appears 500 times helps far more
    # than fixing one that appears 5 times, even if both fragment equally badly.
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]
    
# def add_tokens_and_resize(model, tokenizer, new_tokens: List[str]) -> int:
#     """
#     Add `new_tokens` (plain strings) to the tokenizer's vocabulary and resize
#     the model's embedding table (input embeddings + tied/untied LM head) to
#     match. Returns the number of tokens actually added (duplicates skipped).
#     """
#     num_added = tokenizer.add_tokens(new_tokens)
#     if num_added > 0:
#         model.resize_token_embeddings(len(tokenizer))
#     return num_added

def add_tokens_and_resize(model, tokenizer, new_tokens: List[str]) -> int:
    """
    Add `new_tokens` (plain strings) to the tokenizer's vocabulary and resize
    the underlying LM's embedding table (input embeddings + tied/untied LM head)
    to match. Returns the number of tokens actually added (duplicates skipped).
    """
    num_added = tokenizer.add_tokens(new_tokens)
    if num_added > 0:
        # The outer Qwen3ASRForConditionalGeneration wrapper doesn't implement
        # get_input_embeddings/set_input_embeddings itself — the real LM lives
        # in .thinker, so resize must happen there.
        target = model.thinker if hasattr(model, "thinker") else model
        target.resize_token_embeddings(len(tokenizer))

        # Keep configs in sync (some inference/save paths read vocab_size
        # off the outer config too).
        if hasattr(model, "config") and hasattr(target, "config"):
            model.config.vocab_size = target.config.vocab_size
    return num_added


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Finetuning")

    # Paths
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-finetuning-out")

    # Audio
    p.add_argument("--sr", type=int, default=16000)

    # Train hyper-params
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--grad_acc", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.02)
    p.add_argument("--optim", type=str, default="adamw_bnb_8bit")
    p.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")

    # DataLoader
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)

    # Save
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=5)

    # Resume
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)
    
    # Precision
    p.add_argument("--bf16", action="store_true", help="Use bf16 if available")
    p.add_argument("--fp16", action="store_true", help="Use fp16 if available")
    
    # Monitoring
    p.add_argument("--report_to", type=str, default="none")
    
    # Tokenizer
    p.add_argument("--add_tokens", action="store_true", help="Scan train file for new tokens to add to the tokenizer")
    p.add_argument("--min_freq", type=int, default=5, help="Minimum frequency for a word to be considered for adding to the tokenizer")
    p.add_argument("--max_fragments_ok", type=int, default=1, help="Maximum number of subword fragments allowed for a word to be considered for adding to the tokenizer")
    p.add_argument("--top_k", type=int, default=500, help="Maximum number of new tokens to add to the tokenizer")

    return p.parse_args()


def main():
    args_cli = parse_args()
    for arg_name, arg_val in vars(args_cli).items():
        print(f"{arg_name}: {arg_val}")

    if not args_cli.train_file:
        raise ValueError("TRAIN_FILE is required (json/jsonl). Needs fields: audio, text, optional prompt")

    use_bf16 = args_cli.bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    
    if use_bf16:
        dtype = torch.bfloat16
    elif args_cli.fp16:
        dtype = torch.float32
    else:
        dtype = torch.float32
        
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args_cli.model_path,
        dtype=dtype,
        device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor
    
    if args_cli.add_tokens:
        candidates = extract_new_tokens(
            args_cli.train_file,
            processor.tokenizer,
            min_freq=args_cli.min_freq,
            max_fragments_ok=args_cli.max_fragments_ok,
            top_k=args_cli.top_k,
        )
        for word, freq, n_pieces in candidates[:20]:
            print(f"  {word!r:25} freq={freq:<6} pieces={n_pieces}")
            
        new_tokens = [w for w, _, _ in candidates]
        before_size = len(processor.tokenizer)
        num_added = add_tokens_and_resize(model, processor.tokenizer, new_tokens)
        after_size = len(processor.tokenizer)
        print(f"\nVocab size: {before_size} -> {after_size} ({num_added} tokens actually added, "
            f"{len(new_tokens) - num_added} were already in the vocab)")

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args_cli.train_file,
            **({"validation": args_cli.eval_file} if args_cli.eval_file else {}),
        },
    )
    ds = raw_ds.map(make_preprocess_fn_prefix_only(processor), num_proc=1)

    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    collator = DataCollatorForQwen3ASRFinetuning(processor=processor, sampling_rate=args_cli.sr)

    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=args_cli.train_batch_size,
        per_device_eval_batch_size=args_cli.eval_batch_size,
        gradient_accumulation_steps=args_cli.grad_acc,
        learning_rate=args_cli.lr,
        num_train_epochs=args_cli.epochs,
        max_steps=args_cli.max_steps,
        logging_steps=args_cli.log_steps,
        lr_scheduler_type=args_cli.lr_scheduler_type,
        warmup_ratio=args_cli.warmup_ratio,
        dataloader_num_workers=args_cli.num_workers,
        dataloader_pin_memory=(args_cli.pin_memory == 1),
        dataloader_persistent_workers=(args_cli.persistent_workers == 1),
        dataloader_prefetch_factor=args_cli.prefetch_factor if args_cli.num_workers > 0 else None,
        save_strategy=args_cli.save_strategy,
        save_steps=args_cli.save_steps,
        save_total_limit=args_cli.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps",
        eval_steps=args_cli.save_steps,
        do_eval=bool(args_cli.eval_file),
        bf16=use_bf16,
        fp16=args_cli.fp16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        gradient_checkpointing=args_cli.gradient_checkpointing,
        optim=args_cli.optim,
        report_to=args_cli.report_to,
    )

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[MakeEveryCheckpointInferableCallback(base_model_path=args_cli.model_path)],
    )

    resume_from = (args_cli.resume_from or "").strip()
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        if trainer.args.process_index == 0:
            print(f"[resume] resume_from_checkpoint = {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
