import numpy as np
import pandas as pd
from DataLoaderInput import TestDataLoaderInput
import config
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import transformers
from transformers import get_linear_schedule_with_warmup
from transformers import logging

from utils import get_fold_data, save_model

logging.set_verbosity_error()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# the input class for the pytorch DataLoader utility. __len__ and __getitem__ are method
# that implemended for the DataLoader protocol


# the loss function - MSE loss
def calc_loss(output, target):
    return torch.sqrt(nn.MSELoss()(output, target))


# the traininig loop

def train(model, train_dataloader, optimizer, scheduler):
    model.train()

    epoch_losses = []

    for train_batch in train_dataloader:
        optimizer.zero_grad()

        # batch_id+=1

        sequence = train_batch['sequence'].to(device)
        masks = train_batch['mask'].to(device)
        targets = train_batch['targets'].to(device)

        outputs = model(sequence, masks)['logits'].squeeze(-1)

        loss = calc_loss(outputs, targets)
        
        epoch_losses.append(loss.item())

        loss.backward()
        optimizer.step()

        scheduler.step()

    epoch_mean_loss = np.mean(epoch_losses)

    return epoch_mean_loss


# the validating loop
def validate(model, valid_dataloader):
    model.eval()
    epoch_losses = []
    for valid_batch in valid_dataloader:
        sequence = valid_batch['sequence'].to(device)
        masks = valid_batch['mask'].to(device)
        targets = valid_batch['targets'].to(device)

        outputs = model(sequence, masks)['logits'].squeeze(-1)

        loss = calc_loss(outputs, targets)

        epoch_losses.append(loss.item())

    epoch_mean_loss = np.mean(epoch_losses)

    return epoch_mean_loss


# function for predicting the test dataset values
def predict(test_df):
    predictions = []

    model.eval()

    test_dataset = TestDataLoaderInput(test_df["excerpt"])

    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2,
                                 pin_memory=True)

    for test_batch in test_dataloader:
        sequence = test_batch['sequence'].to(device)
        masks = test_batch['mask'].to(device)

        outputs = model(sequence, masks)['logits'].squeeze(-1)
        outputs = outputs.cpu().detach().numpy()

        predictions = np.concatenate([predictions, outputs])

    return predictions


if __name__ == '__main__':

    # loading the train and test dataset
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    print(train_df.head(3))

    # The target values histogram
    train_df['target'].plot.hist(bins=30)

    # load the huggingface bert tokenizer. We will use the uncased one (no different between upper and lower case)
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    # find the maximum sequence length. The input data will be padded according to the maximum length
    len_arr = []
    for sent in train_df['excerpt']:
        encoded_sent = tokenizer.encode_plus(sent)['input_ids']
        len_arr.append(len(encoded_sent))

    max_seq_len = max(len_arr)

    # add a kfold column, will be used in the cross validation
    train_df["kfold"] = train_df.index % 5
    train_df.head()

    # lets check how the model performs on one fold (selecting k=0)
    le = []
    best_score = None
    k = 0
    train_dataloader, valid_dataloader, train_steps, num_steps = get_fold_data(train_df, k, tokenizer, max_seq_len)
    # create the bert model
    model = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_steps, train_steps)

    train_losses = []
    valid_losses = []

    for epoch in range(config.epochs):
        train_epoch_losses = []
        valid_epoch_losses = []

        train_epoch_loss = train(model, train_dataloader, optimizer, scheduler)
        train_losses.append(train_epoch_loss)

        valid_epoch_loss = validate(model, valid_dataloader)
        valid_losses.append(valid_epoch_loss)

        # save model weight for the best score

        if (best_score is None) or valid_epoch_loss < best_score:
            best_score = valid_epoch_loss
            save_model(model, optimizer, best_score, k)

        print(
            f'Epoch {epoch + 1}/{config.epochs}, Train Loss: {train_epoch_loss:.4f} ,  Test Loss: {valid_epoch_loss:.4f}')

    # the train/valid loss per epoch plot is under "outputs" folder
    plt.plot(train_losses, label="Train")
    plt.plot(valid_losses, label="Valid")
    plt.title("Loss per Epoch")
    plt.legend()
    plt.savefig('outputs/loss_plot.png', dpi=300, bbox_inches='tight')

    # lets continue the cross validation for each fold

    best_scores = [best_score]  # a list with the best score for each fold

    for k in range(1, 5):
        best_score = None

        train_dataloader, valid_dataloader, train_steps, num_steps = get_fold_data(train_df, k)

        model = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_steps, train_steps)

        train_losses = []
        valid_losses = []

        for epoch in range(config.epochs):
            train_epoch_losses = []
            valid_epoch_losses = []

            train_epoch_loss = train(model, train_dataloader, optimizer, scheduler)
            train_losses.append(train_epoch_loss)

            valid_epoch_loss = validate(model, valid_dataloader)
            valid_losses.append(valid_epoch_loss)

            if (best_score is None) or valid_epoch_loss < best_score:
                best_score = valid_epoch_loss
                save_model(model, optimizer, best_score, k)

            print(
                f'Fold {k + 1}/{5}, Epoch {epoch + 1}/{config.epochs}, Train Loss: {train_epoch_loss:.4f} ,  Test Loss: {valid_epoch_loss:.4f}')

        best_scores.append(best_score)

    print("Cross Validation Scores: ", np.mean(best_scores))

    predict(test_df)
