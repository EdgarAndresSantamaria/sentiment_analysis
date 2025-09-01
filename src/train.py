import evaluate
import numpy as np
import pandas as pd
import ray
import ray.train.huggingface.transformers
from datasets import ClassLabel, Dataset, Features, Value
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from utils import id2label, load_model


def train_func():

    def load_dataset():
        train = pd.read_csv("/workspace/data/dataset.csv", sep=";")
        train = train[["Text", "Score"]]

        # rename Score to label in train
        train = train.rename(columns={"Text": "text"})
        train = train.rename(columns={"Score": "label"})
        train["label"] = train["label"].map(id2label)

        label_names = list(id2label.values())
        features = Features(
            {
                "text": Value(dtype="string", id=None),
                "label": ClassLabel(names=label_names, id=None),
            }
        )

        dataset = Dataset.from_list(train.to_dict("records"), features=features, split="train")
        dataset = dataset.train_test_split(test_size=0.3)
        return dataset, len(label_names)

    dataset, num_labels = load_dataset()
    model, tokenizer = load_model()

    def tokenize(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    train_dataset = dataset["train"].map(tokenize, batched=True)
    eval_dataset = dataset["test"].map(tokenize, batched=True)

    # Evaluation Metrics
    metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels, average="micro")

    # Hugging Face Trainer
    training_args = TrainingArguments(
        output_dir="/workspace/model",
        learning_rate=0.0001,  # convergence
        per_device_eval_batch_size=8,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.0001,  # regularization
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        gradient_accumulation_steps=8,  # gradient accumulation
        gradient_checkpointing=True,  # partial gradient calculation
        fp16=True,  # half precision
    )

    # Log on each process the small summary:
    print(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            ray.train.huggingface.transformers.RayTrainReportCallback(),
        ],
    )

    # [3] Prepare Transformers Trainer
    # ================================
    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)
    trainer.train()


# [4] Define a Ray TorchTrainer to launch `train_func` on all workers
# ===================================================================
ray_trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(num_workers=1, use_gpu=True),
)


context = ray.init(dashboard_host="0.0.0.0")
result: ray.train.Result = ray_trainer.fit()
