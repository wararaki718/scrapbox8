from datasets import Dataset
from transformers import TrainingArguments

from encoder import TextSparseEncoder
from trainer import TextSparseTrainer


def train(model: TextSparseEncoder, train_dataset: Dataset) -> TextSparseTrainer:
    training_args = TrainingArguments(
        output_dir="./text_sparse_encoder",
        per_device_train_batch_size=64,
        num_train_epochs=3,
        warmup_steps=10,
        weight_decay=0.01,
        save_total_limit=0,
        save_steps=0,
        logging_steps=10,
        report_to="none",
        save_strategy="no",
    )

    trainer = TextSparseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    return trainer