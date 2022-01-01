import os, sys
import logging
import numpy as np
import pandas as pd
import argparse

import torchaudio
import torch
import re
import json 
import librosa
from datasets import DatasetDict

from transformers import (
    set_seed,
    Wav2Vec2Processor, 
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Config,
    Trainer,
    HfArgumentParser,
    EarlyStoppingCallback
)

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import datasets
import pickle

import editdistance
import jieba
from itertools import chain

import transformers
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from args_helper import ModelArguments, DataArguments, TrainingArguments
from utils import CHARS_TO_IGNORE, remove_special_characters, tokenize_for_mer, tokenize_for_cer
from data_utils import speech_file_to_array_fn, load_dataset, DataCollatorCTCWithPadding

import datasets
from datasets import load_from_disk, set_caching_enabled

set_caching_enabled(True)
logger = logging.getLogger(__name__)    

#####
# Main Functions
#####
def run(model_args, data_args, training_args):
    ###
    # Prepare Processor & Model    
    ###
    training_args.output_dir="{}/{}".format(training_args.output_dir, model_args.model_name_or_path)

    os.makedirs(training_args.output_dir, exist_ok=True)

    def load_processor(model_args, training_args):
        # Load processor
        print('Load Wav2Vec2 processor...')

        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_args.model_name_or_path)
        logger.info("Vocab length (initial): {}".format(len(tokenizer)))
        print("Vocab length (initial):", len(tokenizer))

        with open("{}/new_vocab.json".format(training_args.output_dir), "r") as vocab_file:
            vocab_list = json.load(vocab_file)
        print(vocab_list)
        num_added_tokens = tokenizer.add_tokens(vocab_list)

        logger.info("New vocabulary length: {} out of {}".format(num_added_tokens, len(vocab_list)))

        logger.info("Vocab length (final): {}".format(len(tokenizer)))
        print("Vocab length (final):", len(tokenizer))

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_args.model_name_or_path)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        
        return processor

    cache_dir_path = "./{}/{}".format(data_args.cache_dir_name, model_args.model_name_or_path)
    os.makedirs(cache_dir_path, exist_ok=True)

    print('cache_dir_path', cache_dir_path)
    if not os.path.exists("{}/preprocess_data.arrow".format(cache_dir_path)):
        ###
        # Prepare Dataset
        ###
        raw_datasets = DatasetDict()
        print('Loading train dataset...')
        raw_datasets["train"] = load_dataset(data_args.train_manifest_path, data_args.num_workers, 
                                        data_args.audio_column_name, data_args.text_column_name)
        print('Loading validation dataset...')
        raw_datasets["valid"] = load_dataset(data_args.valid_manifest_path, data_args.num_workers, 
                                        data_args.audio_column_name, data_args.text_column_name)
        print('Loading test dataset...')
        raw_datasets["test"] = load_dataset(data_args.test_manifest_path, data_args.num_workers, 
                                        data_args.audio_column_name, data_args.text_column_name)

        print('Preprocess dataset...')

        # Remove ignorable characters
        print('Removing ignorable characters')
        chars_to_ignore_re = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"
        def remove_special_characters(batch):
            if chars_to_ignore_re is not None:
                batch[data_args.text_column_name] = re.sub(chars_to_ignore_re, "", batch[data_args.text_column_name]).lower() + " "
            else:
                batch[data_args.text_column_name] = batch[data_args.text_column_name].lower() + " "
            return batch

        with training_args.main_process_first(desc="dataset map special characters removal"):
            raw_datasets = raw_datasets.map(
                remove_special_characters,
                num_proc=data_args.preprocessing_num_workers,
                desc="remove special characters from datasets",
                load_from_cache_file=True,
                cache_file_names={
                    "train": "{}/train_clean.arrow".format(cache_dir_path),
                    "valid": "{}/valid_clean.arrow".format(cache_dir_path),
                    "test": "{}/test_clean.arrow".format(cache_dir_path),
                }
            )

        # Build vocabulary
        print('Build vocabulary...')
        def extract_all_chars(batch):
            all_text = " ".join(batch[data_args.text_column_name])
            vocab = list(set(all_text))
            return {"vocab": [vocab], "all_text": [all_text]}

        with training_args.main_process_first(desc="vocab building"):
            _vocab = raw_datasets.map(
                extract_all_chars,
                num_proc=data_args.preprocessing_num_workers,
                desc="build vocabulary",
                load_from_cache_file=True,
                cache_file_names={
                    "train": "{}/train_vocab.arrow".format(cache_dir_path),
                    "valid": "{}/valid_vocab.arrow".format(cache_dir_path),
                    "test": "{}/test_vocab.arrow".format(cache_dir_path),
                }
            )

            def flatten(vocab_split):
                return list(chain.from_iterable(list(chain.from_iterable(vocab_split))))

            vocab_list = list(set(flatten(_vocab["train"]["vocab"]) + flatten(_vocab["valid"]["vocab"]) + flatten(_vocab["test"]["vocab"])))
            # vocab_dict = {v: k for k, v in enumerate(vocab_list)}
            # vocab_dict["|"] = vocab_dict[" "]
            # vocab_dict["[UNK]"] = len(vocab_dict)
            # vocab_dict["[PAD]"] = len(vocab_dict)

            # Dump vocabulary
            with open("{}/new_vocab.json".format(training_args.output_dir), "w") as vocab_file:
                json.dump(vocab_list, vocab_file)

        # Load processor
        processor = load_processor(model_args, training_args)

        # Preprocess audio sample and label text
        print('Vectorize dataset...')

        def prepare_dataset(batch):
            # Preprocess audio
            batch["input_values"] = processor(batch["speech_sample"], sampling_rate=16000).input_values[0]

            # Preprocess text
            with processor.as_target_processor():
                batch["labels"] = processor(batch[data_args.text_column_name]).input_ids

            return batch

        with training_args.main_process_first(desc="dataset map preprocessing"):
            vectorized_datasets = raw_datasets.map(
                prepare_dataset,
                remove_columns=raw_datasets["train"].column_names,
                num_proc=data_args.preprocessing_num_workers,
                desc="preprocess datasets",
                load_from_cache_file=True,
                cache_file_names={
                    "train": "{}/train_vec.arrow".format(cache_dir_path),
                    "valid": "{}/valid_vec.arrow".format(cache_dir_path),
                    "test": "{}/test_vec.arrow".format(cache_dir_path),
                }
            )
        
        vectorized_datasets.save_to_disk("{}/preprocess_data.arrow".format(cache_dir_path))
    else:
        print('Loading cached dataset...')
        vectorized_datasets = datasets.load_from_disk('{}/preprocess_data.arrow'.format(cache_dir_path))

        # Load processor
        processor = load_processor(model_args, training_args)

    if data_args.preprocessing_only:
        logger.info(f"Data preprocessing finished. Files cached at {vectorized_datasets.cache_files}")
        return
    
    ###
    # Prepare Data Collator and Trainer
    ###
    print('Preparing Trainer...')

    print('Load Wav2Vec2 model...')
    print('Model ID', model_args.model_name_or_path)
    config = Wav2Vec2Config.from_pretrained(model_args.model_name_or_path)
    config.update({
        "mask_time_prob": model_args.mask_time_prob,
        "mask_time_length": model_args.mask_time_length,
        "mask_feature_prob": model_args.mask_feature_prob,
        "mask_feature_length": model_args.mask_feature_length,
        "gradient_checkpointing": training_args.gradient_checkpointing,
    })
    model = Wav2Vec2ForCTC.from_pretrained(model_args.model_name_or_path, config=config)
    model.cuda()

    # Instantiate custom data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # Define compute metric function
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_strs = processor.batch_decode(pred_ids)

        # we do not want to group tokens when computing the metrics
        label_strs = processor.batch_decode(pred.label_ids, group_tokens=False)
        mixed_distance, mixed_tokens = 0, 0
        char_distance, char_tokens = 0, 0
        for pred_str, label_str in zip(pred_strs, label_strs):
            # Calculate 
            m_pred = tokenize_for_mer(pred_str)
            m_ref = tokenize_for_mer(label_str)
            mixed_distance += editdistance.distance(m_pred, m_ref)
            mixed_tokens += len(m_ref)

            c_pred = tokenize_for_cer(pred_str)
            c_ref = tokenize_for_cer(label_str)
            char_distance += editdistance.distance(c_pred, c_ref)
            char_tokens += len(c_ref)
        mer = mixed_distance / mixed_tokens
        cer = char_distance / char_tokens

        return {"mer": mer, "cer": cer} 

    # Initialize Trainer
    trainer = Trainer(
        train_dataset=vectorized_datasets["train"],
        eval_dataset=vectorized_datasets["valid"],
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor
    )

    ###
    # Training Phase
    ###
    print('*** Training Phase ***')
    
    # use last checkpoint if exist
    if os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    # Save the feature_extractor and the tokenizer
    if is_main_process(training_args.local_rank):
        processor.save_pretrained(training_args.output_dir)

    metrics = train_result.metrics
    metrics["train_samples"] = len(vectorized_datasets["train"])

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    ###
    # Evaluation Phase
    ###
    results = {}
    logger.info("*** Evaluation Phase ***")
    metrics = trainer.evaluate(eval_dataset=vectorized_datasets["valid"])
    metrics["eval_samples"] = len(vectorized_datasets["valid"])

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
#####
# Entry Point
#####
def main():
    ###
    # Parsing & Initialization
    ###
    # Parse argument
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set random seed
    set_seed(training_args.seed)
    
    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    ###
    # Prepare logger
    ###
    # Init logging
    os.makedirs("./log", exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(
            "./log/log__{}".format(model_args.model_name_or_path.replace("/", "_")), mode="w")],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to warn of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity(transformers.logging.WARNING)
    logger.info("Training/evaluation parameters %s", training_args)
    
    ###
    # RUN RUN RUN!!!
    ###
    run(model_args, data_args, training_args)
    
if __name__ == '__main__':
    main()