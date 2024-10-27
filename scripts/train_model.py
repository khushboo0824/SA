import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk

# Load the tokenized datasets
tokenized_train = load_from_disk('D:/NLP/data/processed')
tokenized_test = load_from_disk('D:/NLP/data/processed')

# Load the DistilBERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',            # output directory
    evaluation_strategy='epoch',       # evaluation strategy
    learning_rate=2e-5,                # learning rate
    per_device_train_batch_size=16,    # batch size for training
    per_device_eval_batch_size=64,     # batch size for evaluation
    num_train_epochs=3,                # total number of training epochs
    weight_decay=0.01,                 # strength of weight decay
)

# Create a Trainer instance
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=tokenized_train,       # training dataset
    eval_dataset=tokenized_test           # evaluation dataset
)

# Start training
trainer.train()
=======
import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk

# Load the tokenized datasets
tokenized_train = load_from_disk('D:/NLP/data/processed')
tokenized_test = load_from_disk('D:/NLP/data/processed')

# Load the DistilBERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',            # output directory
    evaluation_strategy='epoch',       # evaluation strategy
    learning_rate=2e-5,                # learning rate
    per_device_train_batch_size=16,    # batch size for training
    per_device_eval_batch_size=64,     # batch size for evaluation
    num_train_epochs=3,                # total number of training epochs
    weight_decay=0.01,                 # strength of weight decay
)

# Create a Trainer instance
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=tokenized_train,       # training dataset
    eval_dataset=tokenized_test           # evaluation dataset
)

# Start training
trainer.train()
>>>>>>> 6bc8fe0c4f55f62dcb0aea07c8186bca2604f0bd
