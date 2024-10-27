import pandas as pd
from datasets import load_dataset
from transformers import DistilBertTokenizer



# Load the IMDB dataset
imdb_dataset = load_dataset('imdb')

# Extract the train and test splits
train_data = imdb_dataset['train']
test_data = imdb_dataset['test']
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

# Tokenize the train and test datasets
tokenized_train = train_data.map(tokenize_function, batched=True)
tokenized_test = test_data.map(tokenize_function, batched=True)

tokenized_train.save_to_disk('D:/NLP/data/processed')
=======
import pandas as pd
from datasets import load_dataset
from transformers import DistilBertTokenizer



# Load the IMDB dataset
imdb_dataset = load_dataset('imdb')

# Extract the train and test splits
train_data = imdb_dataset['train']
test_data = imdb_dataset['test']
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

# Tokenize the train and test datasets
tokenized_train = train_data.map(tokenize_function, batched=True)
tokenized_test = test_data.map(tokenize_function, batched=True)

tokenized_train.save_to_disk('D:/NLP/data/processed')
>>>>>>> 6bc8fe0c4f55f62dcb0aea07c8186bca2604f0bd
tokenized_test.save_to_disk('D:/NLP/data/processed')