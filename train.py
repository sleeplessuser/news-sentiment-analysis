from argparse import ArgumentParser, Namespace

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
)

from utils import compute_metrics, get_dataset


def parse_args():
   parser = ArgumentParser()
   parser.add_argument(
      "--lr",
      default=2e-5,
   )
   parser.add_argument("--epochs", default=5)
   parser.add_argument(
      "--batch-size",
      default=32,
   )
   parser.add_argument(
      "--weight-decay",
      default=0.01,
   )
   parser.add_argument(
      "--num-classes",
      dest='num_labels',
      default=3,
   )
   parser.add_argument(
      "--model-id",
      default="microsoft/deberta-v3-large",
   )

   return parser.parse_args()

def train(args: Namespace):
   tokenizer = AutoTokenizer.from_pretrained(args.model_id)
   dataset = get_dataset(tokenizer)

   data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
   model = AutoModelForSequenceClassification.from_pretrained(
      args.model_id, num_labels=args.num_labels
   )

   training_args = TrainingArguments(
      output_dir='test',
      learning_rate=args.lr,
      per_device_train_batch_size=args.batch_size,
      per_device_eval_batch_size=args.batch_size,
      num_train_epochs=args.epochs,
      weight_decay=args.weight_decay,
      save_strategy="epoch",
      report_to=None,
      fp16=True,
      warmup_steps=100,
   )

   trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=dataset["train"],
      eval_dataset=dataset["validation"],
      tokenizer=tokenizer,
      data_collator=data_collator,
      compute_metrics=compute_metrics,
   )

   trainer.train()
   eval_metrics = trainer.evaluate(dataset["validation"])
   test_metrics = trainer.evaluate(dataset["test"], metric_key_prefix="test")
   print(eval_metrics)
   print(test_metrics)


if __name__ == "__main__":
    args = parse_args()
    train(args)
