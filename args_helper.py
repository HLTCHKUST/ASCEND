from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to utilize.
    """
    model_name_or_path: Optional[str] = field(
        default="ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt", metadata={"help": "The path of the HuggingFace model."}
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to the data loading and preprocessing pipeline.
    """
    train_manifest_path: Optional[str] = field(
        default="dataset/train_metadata.csv", metadata={"help": "The path of the training dataset to use."}
    )
    valid_manifest_path: Optional[str] = field(
        default="dataset/val_metadata.csv", metadata={"help": "The path of the validation dataset to use."}
    )
    test_manifest_path: Optional[str] = field(
        default="dataset/test_metadata.csv", metadata={"help": "The path of the testing dataset to use."}
    )
    num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "The number of processes to use for the dataset."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=16,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    preprocessing_only: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to only run preprocessing."},
    )
    audio_column_name: Optional[str] = field(
        default="file_name",
        metadata={"help": "The name of the dataset column containing the audio path. Defaults to 'file_name'"},
    )
    text_column_name: Optional[str] = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )