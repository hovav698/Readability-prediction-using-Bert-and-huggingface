
import torch
from torch.utils.data import DataLoader, Dataset
import config

from DataLoaderInput import TrainDataLoaderInput

from transformers import logging

logging.set_verbosity_error()


def get_fold_data(train_df, k,tokenizer,max_seq_len):
    # split to train and valid
    train_part = train_df[train_df["kfold"] != k].reset_index(drop=True)
    valid_part = train_df[train_df["kfold"] == k].reset_index(drop=True)

    n_batch = len(train_part) // config.batch_size  # num of batches

    # input parameters for the scheduler
    train_steps = n_batch * config.epochs
    num_steps = int(train_steps * 0.1)

    # load the data to the pytorch dataloader

    train_dataset = TrainDataLoaderInput(train_part["excerpt"], train_part["target"],tokenizer,max_seq_len)
    valid_dataset = TrainDataLoaderInput(valid_part["excerpt"], valid_part["target"],tokenizer,max_seq_len)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_dataloader, valid_dataloader, train_steps, num_steps


def save_model(model, optimizer, best_score, k):
    state = {
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        "bestscore": best_score
    }

    torch.save(state, "model" + str(k) + ".pth")
