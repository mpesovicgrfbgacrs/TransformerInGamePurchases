import math
import random
import joblib  
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, TensorDataset

import itertools
from itertools import product

from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, matthews_corrcoef


DAU, PLT, T, S, L, LCPR, LRPR, GB, GG, GS, G, RWU, BU = range(13)

FEATURES = {"PLT": 1, "S": 3, "L": 4, "LCPR": 5, "LRPR": 6, "GB": 7,
            "GG": 8, "GS": 9, "G": 10, "RWU": 11, "BU": 12, "OUT": 13}
            

class Payers(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]
        
        
class PositionalEncoding(nn.Module):

    def __init__(self, max_emb_size, max_seq_len, dropout_p):
        """
        :param max_emb_size: embedding size
        :param max_seq_len: maximal sequences length
        :param dropout_p: dropout value
        """
        super().__init__()
        # protections for eval dimension
        if max_emb_size % 2:
            max_emb_size += 1

        division_term = torch.exp(torch.arange(0, max_emb_size, 2).float() * (-math.log(10000.0)) / max_emb_size)

        pos_encoding = torch.zeros(max_seq_len, max_emb_size)
        positions_list = torch.arange(0, max_seq_len, dtype=torch.float).view(-1, 1)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        self.dropout = nn.Dropout(dropout_p)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, sequences):
        # sequences shape: (batch_size, sequence_length, embedding_size)
        return self.dropout(sequences + self.pos_encoding[:sequences.size(1), :sequences.size(2)])        
        
        
class TransactionClassifier(nn.Module):

    def __init__(self, len_src, voc_size, voc_out_size,
                 embedding_size, num_heads_enc, num_layers_enc, hid_size, dropout_p):
        """
        :param len_src: len of sequence
        :param voc_size: size of the vocabulary
        :param voc_out_size: size of the output vocabulary
        :param embedding_size: size of embedding layer
        :param num_heads_enc: # of heads in encoder layer
        :param num_layers_enc: # of layers in encoder layer
        :param hid_size: # of hidden size layer
        :param dropout_p: dropout param
        """
        super().__init__()

        self.len_src = len_src
        self.emb_size = embedding_size
        self.embedding = nn.Embedding(voc_size, embedding_size)

        self.weights = nn.Linear(len_src, 1)
        self.out = nn.Linear(embedding_size, voc_out_size)

        self.positional_encoder = PositionalEncoding(max_emb_size=embedding_size,
                                                     max_seq_len=len_src,
                                                     dropout_p=dropout_p)

        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embedding_size,
                                                                        nhead=num_heads_enc,
                                                                        dim_feedforward=hid_size),
                                             num_layers_enc)

    def forward(self, src, src_attn_mask=None, src_pad_mask=None):
    
        # src size (batch_size, src_sequence_len)
        src = self.embedding(src) * math.sqrt(self.emb_size)
        # src is now (batch_size, src_sequence_length, emb_size)
        src = self.positional_encoder(src)
        # transformer uses (sequence_length, batch_size, embedding_size inputs)
        src = src.permute(1, 0, 2)
        # encoder layer
        src = self.encoder(src, mask=src_attn_mask, src_key_padding_mask=src_pad_mask)
        # weighted mean
        src = src.permute(1, 2, 0)
        src = self.weights(src).squeeze(2)

        return self.out(src)

    def num_params(self):
    
        return sum([t.numel() for t in self.parameters()])        
        
        
def clear_cuda():
    torch.cuda.empty_cache()
    try:
        if 'model' in globals():
            del model
    except BaseException as exc:
        pass       
        

def train_loop(model, optimizer, loss_fun, data_loader, device_type, num_batches_before_back=10):
    """
    function for training model
    :param model: transformer model for training
    :param optimizer: optimizer
    :param loss_fun: loss function
    :param data_loader: data for training
    :param device_type: CUDA, CPU
    :param num_batches_before_back: # of bacth before back
    :output: total loss
    """
    model.train()
    total_loss, cur_batch = 0, 0

    for x, y in data_loader:
        # information about # of batch
        cur_batch += 1
        x, y = x.to(device_type), y.to(device_type)

        # part for prediction
        y_predicted = model(x, src_attn_mask=None, src_pad_mask=(x == 0)
        loss = loss_fun(y_predicted, y) / num_batches_before_back
        loss.backward()

        # previous batches
        if not (cur_batch % num_batches_before_back):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += loss.detach().item()

    # last batch
    if cur_batch % num_batches_before_back:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total_loss += loss.detach().item()

    return total_loss


def evaluate_loop(model, data_loader, device_type):
    """
    function for evaulation model
    :param model: transformer model for evaulation
    :param data_loader: data for evaulation
    :param device_type: CUDA, CPU
    :output: F1 (macro) measure, predicted values, and real values
    """
    model.eval()
    real_values, predicted_values = [], []
    
    for x, y in data_loader:
        x = x.to(device_type)
        with torch.no_grad():
            y_predicted_values = model(x, src_pad_mask=(x == 0).to(device_type))
            
            y_predicted = torch.argmax(y_predicted_values, dim=1)
            real_values.append(y)
            predicted_values.append(y_predicted.to("cpu"))

    # make torch tensor
    real_values, predicted_values = torch.cat(real_values), torch.cat(predicted_values)

    return f1_score(real_values, predicted_values, average='macro'), predicted_values, real_values


def fit(model_1, model_2, optimizer_1, optimizer_2, loss_fun, tr_loader, va_loader, tr_va_loader, te_loader,
        device_type, epochs, num_batches_before_back, lr_patience, lr_factor, lr_threshold, file):
    """
    function to fit transformer model
    :param model_1: transformer model for finding optimal # of epochs
    :param model_2: transformer model for testing
    :param optimizer_1: optimizer for finding optimal # of epochs
    :param optimizer_2: optimizer for testing
    :param loss_fun: loss function
    :param tr_loader: train data (for finding optimal # of epochs)
    :param va_loader: validation data (for finding optimal # of epochs)
    :param te_loader: test data 
    :param tr_va_loader: train and validation data together
    :param device_type: CUDA, CPU
    :param epochs: # of epochs
    :param num_batches_before_back:
    :param lr_patience: patience for learning rate
    :param lr_factor: factor for learning rate
    :param lr_threshold: threshold for learning rate
    :param file: file object to write the output
    :output: file, F1 (macro) score, model, predicted values, and real values
    """

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_1, 'min', threshold_mode='abs',
                                                     patience=lr_patience, factor=lr_factor, threshold=lr_threshold)

    file.write("Training and validating model\n")
    file.flush()

    tr_loss_list, va_loss_list = [], []
    min_va_loss, best_epoch = 1000., -1

    for epoch in range(epochs):

        # part of training of model
        tr_loss = train_loop(model_1, optimizer_1, loss_fun, tr_loader, device_type, num_batches_before_back)
        tr_loss_list += [tr_loss]

        # part of evaluation of model
        f, _, _ = evaluate_loop(model_1, va_loader, device_type)
        va_loss = 1 - f.sum() / 2
        # information about temporary loss
        va_loss_list += [va_loss]

        # information about the best # of epochs
        if va_loss < min_va_loss:
            min_va_loss, best_epoch = va_loss, epoch
  
        scheduler.step(va_loss)

    # the best model
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_2, 'min', threshold_mode='abs',
                                                     patience=lr_patience, factor=lr_factor, threshold=lr_threshold)

    # protections for the case when best_epoch = 0
    if best_epoch == 0:
        best_epoch = 1
    
    for epoch in range(best_epoch):
        tr_loss = train_loop(model_2, optimizer_2, loss_fun, tr_va_loader, device_type, num_batches_before_back)
        f, predicted_values, real_values = evaluate_loop(model_2, va_loader, device_type)

        scheduler.step(tr_loss)

    return file, min_va_loss, model_2, predicted_values, real_values
        
        
def transformer(tr_x, tr_y, va_x, va_y, te_x, te_y, device, emb_size_l, num_heads_l, num_layers_l, file, step, seed):
    """
    function to find optimal transformer model
    :param tr_x: train set of input data
    :param tr_y: train set of output data
    :param va_x: validation set of input data
    :param va_y: validation set of output data
    :param te_x: test set of input data
    :param te_y: test set of output data
    :param device: CUDA, CPU
    :param emb_size_l: list of embedding size
    :param num_heads_l: list of # of heads
    :param num_layers_l: list of # of layers
    :param file: file object to write the output
    :param step: step prediction value (3, 5, or 7)
    :param seed: seed value of data split
    :output: file, predicted values, and real values
    """
    tr, va, te = Payers(tr_x, tr_y), Payers(va_x, va_y), Payers(te_x, te_y)
    tr_va = Payers(torch.vstack((tr_x, va_x)), torch.cat((tr_y, va_y)))
  
    best_loss, best_emb, best_head, best_lay, predicted_v, real_v = 10000, 0, 0, 0, [], []
    

    for emb_size, num_heads, num_layers in product(emb_size_l, num_heads_l, num_layers_l):
    
        cond1 = emb_size == 8 and num_heads == 2 and num_layers == 2
        cond2 = emb_size == 8 and num_heads == 4 and num_layers == 2
        cond3 = emb_size == 16 and num_heads == 2 and num_layers == 2
        cond4 = emb_size == 16 and num_heads == 4 and num_layers == 1
        cond5 = emb_size == 16 and num_heads == 8 and num_layers == 2
        cond6 = emb_size == 32 and num_heads == 2 and num_layers == 2
        cond7 = emb_size == 32 and num_heads == 4 and num_layers == 2
        cond8 = emb_size == 32 and num_heads == 8 and num_layers == 2
        
        if cond1 or cond2 or cond3 or cond4 or cond5 or cond6 or cond7 or cond8:

            clear_cuda()
            # model for finding optimal hyperparametars
            model_1 = TransactionClassifier(len_src=tr.X.size(1),
                                            voc_size=tr.X.max() + 1,
                                            voc_out_size=tr.Y.max() + 1,
                                            embedding_size=emb_size,
                                            num_heads_enc=num_heads,
                                            num_layers_enc=num_layers,
                                            hid_size=64,
                                            dropout_p=0.1
                                            ).to(device)
            # optimal model
            model_2 = TransactionClassifier(len_src=tr_va.X.size(1),
                                            voc_size=tr_va.X.max() + 1,
                                            voc_out_size=tr_va.Y.max() + 1,
                                            embedding_size=emb_size,
                                            num_heads_enc=num_heads,
                                            num_layers_enc=num_layers,
                                            hid_size=64,
                                            dropout_p=0.1
                                            ).to(device)

            file, min_va_loss, model_2, predicted_values, real_values = fit(model_1, 
                                                                            model_2,
                                                                            optimizer_1=optim.Adam(model_1.parameters(), lr=0.01),
                                                                            optimizer_2=optim.Adam(model_2.parameters(), lr=0.01),
                                                                            loss_fun=nn.CrossEntropyLoss(),
                                                                            tr_loader=DataLoader(tr, batch_size=100, shuffle=True),
                                                                            va_loader=DataLoader(va, batch_size=100, shuffle=True),
                                                                            tr_va_loader=DataLoader(tr_va, batch_size=100, shuffle=True),
                                                                            te_loader=DataLoader(te, batch_size=100, shuffle=True),
                                                                            device_type=device,
                                                                            epochs=30,
                                                                            num_batches_before_back=5,
                                                                            lr_patience=2,
                                                                            lr_factor=0.2,
                                                                            lr_threshold=0.01,
                                                                            file=file
                                                                            )
                                  
            if min_va_loss < best_loss:
                best_loss = min_va_loss
                best_emb, best_head, best_lay = emb_size, num_heads, num_layers
                predicted_v, real_v = predicted_values, real_values

                # save the model 
                joblib.dump(best_model, f'best_transformer{step}_{seed}_model.joblib')

    # information about the best parameters
    file.write('Best parameters found using validation set - \n')
    file.flush()
    file.write(str({'emb_size': best_emb, 'num_heads': best_head, 'num_layers': best_lay}))
    file.flush()
    file.write('\n')
    file.flush()
    
    return file, predicted_v, real_v
        
   
if __name__ == "__main__":


    # device
    DEVICE = "cuda" 
    # seed values
    seed_l = [399, 1054, 1175, 1475, 1862, 1903, 1990, 2000, 2023, 2053]

    # parameters for transformer model
    emb_size_l = [8, 16, 32, 64]
    num_heads_l = [2, 4, 8]
    num_layers_l = [1, 2]
    steps_l = [3, 5, 7]

    # file for writing information
    file = open('Results.txt', 'a')
    
    predicted_values, real_values = torch.empty(), torch.empty()

    for seed in seed_l:
    
        # data path
        path = 'nl_'+str(seed)+'/'
        
        for step in steps_l:
        
            # training data
            tr_x_0 = torch.load(f"{path}tr_x_0_{seed}.pt")
            # tr_x_1 = torch.load(f"{path}tr_x_1_{seed}.pt") (for frequencies models)
            # tr_x_2 = torch.load(f"{path}tr_x_2_{seed}.pt") (for statistics models)
            tr_y = torch.load(f"{path}tr_y{step}_{seed}.pt")
        
            # validation data
            va_x_0 = torch.load(f"{path}va_x_0_{seed}.pt")
            # va_x_1 = torch.load(f"{path}va_x_1_{seed}.pt") (for frequencies models)
            # va_x_2 = torch.load(f"{path}va_x_2_{seed}.pt") (for statistics models)
            va_y = torch.load(f"{path}va_y{step}_{seed}.pt")
            
            # test data
            te_x_0 = torch.load(f"{path}te_x_0_{seed}.pt")
            # te_x_1 = torch.load(f"{path}te_x_1_{seed}.pt") (for frequencies models)
            # te_x_2 = torch.load(f"{path}te_x_2_{seed}.pt") (for statistics models)
            te_y = torch.load(f"{path}te_y{step}_{seed}.pt")
            
            # Transformer model
            file.write("Transformer model\n")
            file.flush()
            file, predicted_v, real_v = transformer(tr_x_0, tr_y, va_x_0, va_y, te_x_0, te_y, DEVICE, emb_size_l, num_heads_l, num_layers_l, file, step, seed)
            
            predicted_values, real_values = torch.cat((predicted_values, predicted_v), dim=0), torch.cat((real_values, real_v), dim=0)
                   
        
    # validation part
    predicted_values, real_values = predicted_values.numpy(), real_values.numpy()  
    
    # Confusion matrix
    file.write("Confusion matrix\n")
    file.write(str(confusion_matrix(real_values, predicted_values >= 0.5)))
    
    # F1 macro score    
    file.write("F1 (macro) score\n")
    file.write(str(f1_score(real_values, predicted_values >= 0.5)))
    
    # MCC score    
    file.write("MCC score\n")
    file.write(str(matthews_corrcoef(real_values, predicted_values >= 0.5)))
    
    # AUROC score    
    file.write("AUROC score\n")
    file.write(str(roc_auc_score(real_values, predicted_values)))
    
    
    
    