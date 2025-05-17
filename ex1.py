import argparse
from datasets import load_dataset
from transformers import (AutoTokenizer, TrainingArguments, Trainer,
                          DataCollatorWithPadding, AutoModelForSequenceClassification)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Constants from the exercise
BENCHMARK_PATH = 'glue'
DATASET_NAME = 'mrpc'
MODEL_TO_FINETUNE = 'bert-base-uncased'


def get_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_train_samples', type=int, default=-1)
    parser.add_argument('--max_eval_samples', type=int, default=-1)
    parser.add_argument('--max_predict_samples', type=int, default=-1)
    parser.add_argument('--num_train_epochs', type=int, default=2)  # max 5
    parser.add_argument('--lr', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--do_train', type=bool, default=False)
    parser.add_argument('--do_predict', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='./model')
    return parser.parse_args()


def get_split_dataset(command_line_args):
    dataset = load_dataset(BENCHMARK_PATH, DATASET_NAME).map(tokenize_data, batched=True)

    train_dataset = dataset['train']
    if command_line_args.max_train_samples != -1:
        train_dataset = train_dataset.select(range(command_line_args.max_train_samples))

    eval_dataset = dataset['validation']
    if command_line_args.max_eval_samples != -1:
        eval_dataset = eval_dataset.select(range(command_line_args.max_eval_samples))

    test_dataset = dataset['test']
    if command_line_args.max_predict_samples != -1:
        test_dataset = test_dataset.select(range(command_line_args.max_predict_samples))

    return train_dataset, eval_dataset, test_dataset


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, fbeta_score, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': fbeta_score,
        'precision': precision,
        'recall': recall
    }


# global, for the tokenizer usages.
# use_fast=True for faster tokenization if available for this model
tokenizer = AutoTokenizer.from_pretrained(MODEL_TO_FINETUNE, use_fast=True)


def tokenize_data(data):
    # Truncation and padding
    return tokenizer(
        data["sentence1"],
        data["sentence2"],
        truncation=True,
        max_length=tokenizer.model_max_length,
        # Dynamic padding is via DataCollatorWithPadding
        padding=False
    )


def get_training_args(command_line_args):
    return TrainingArguments(
        num_train_epochs=command_line_args.num_train_epochs,
        learning_rate=command_line_args.lr,
        per_device_train_batch_size=command_line_args.batch_size,
        per_device_eval_batch_size=command_line_args.batch_size,
        do_train=command_line_args.do_train,
        do_predict=command_line_args.do_predict,

        # Checkpoints of the model
        output_dir="./res",
        overwrite_output_dir=True,

        eval_strategy="steps",
        eval_steps=25,

        # wandb data
        report_to=None,  # wandb
        # "Make sure you log the training loss every step."
        logging_steps=1,
    )


def get_trainer(trainer_model, command_line_args, train_dataset, eval_dataset):
    return Trainer(
        model=trainer_model,
        args=get_training_args(command_line_args),
        # For dynamic padding
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )


def main():
    # wandb.init(project="projectName", id="runId", resume="allow")

    # Inner functions for writing predictions and results
    def write_predictions():
        pred_output_lines = []
        for sentence1, sentence2, pred_label in zip(test_data['sentence1'], test_data['sentence2'], pred_labels):
            pred_output_lines.append(
                f"Sentence 1: {sentence1}\n" +
                f"Sentence 2: {sentence2}\n" +
                f"Prediction: {pred_label}\n" +
                "-------------------------------\n")

        with open("predictions.txt", "w") as f:
            f.write('\n'.join(pred_output_lines))

    def write_results():
        with open('res.txt', 'a') as f:
            f.write(f'epoch_num: {args.num_train_epochs}, lr: {args.lr}, ' +
                    f'batch_size: {args.batch_size}, eval_acc: {eval_result["eval_accuracy"]:.4f}\n')

    # Get command line arguments
    args = get_command_line_args()

    # Split
    train_data, eval_data, test_data = get_split_dataset(args)

    # Hyperparameters (number of epochs, learning rate, batch size) in res.txt file

    # Model configuration ("Use the AutoModelForSequenceClassification class to load your models.")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_TO_FINETUNE, num_labels=2)

    # Truncation and padding in the tokenize_data function

    # Weights&Biases - ran via google colab, as I've had problems locally
    if args.do_train:
        # Training
        trainer = get_trainer(model, args, train_data, eval_data)
        trainer.train()
        trainer.save_model(args.model_path)
        eval_result = trainer.evaluate()
        write_results()

    if args.do_predict:
        # Predicting
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        trainer = get_trainer(model, args, train_data, eval_data)
        pred_labels = np.argmax(trainer.predict(test_data).predictions, axis=1)
        write_predictions()


# You may run this locally.
if __name__ == '__main__':
    main()
