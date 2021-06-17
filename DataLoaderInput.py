import torch

from torch.utils.data import Dataset

from transformers import logging

logging.set_verbosity_error()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# the input class for the pytorch DataLoader utility. __len__ and __getitem__ are methods
# that are implemented for the DataLoader protocol

class TrainDataLoaderInput(Dataset):

    def __init__(self, sentences, targets, tokenizer, max_seq_len):
        self.sentences = sentences
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        bert_sents = self.tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=self.max_seq_len,
                                                padding='max_length', return_attention_mask=True, truncation=True)

        sequence = torch.tensor(bert_sents['input_ids'], dtype=torch.long)
        mask = torch.tensor(bert_sents['attention_mask'], dtype=torch.long)
        target = torch.tensor(self.targets[idx], dtype=torch.float)

        return {
            'sequence': sequence,
            'mask': mask,
            'targets': target
        }


# We don't have targets for the test dataset, so we create
# new input class for the test dataloader

class TestDataLoaderInput(Dataset):

    def __init__(self, sentences, tokenizer, max_seq_len):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        bert_sents = self.tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=self.max_seq_len,
                                                padding='max_length', return_attention_mask=True, truncation=True)

        sequence = torch.tensor(bert_sents['input_ids'], dtype=torch.long)
        mask = torch.tensor(bert_sents['attention_mask'], dtype=torch.long)

        return {
            'sequence': sequence,
            'mask': mask
        }
