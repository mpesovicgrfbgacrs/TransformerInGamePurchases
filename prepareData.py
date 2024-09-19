import math
import random
import numpy as np

import torch
import torch.nn as nn

from itertools import product
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader

DAU, PLT, T, S, L, LCPR, LRPR, GB, GG, GS, G, RWU, BU = range(13)

FEATURES = {"PLT": 1, "S": 3, "L": 4, "LCPR": 5, "LRPR": 6, "GB": 7,
            "GG": 8, "GS": 9, "G": 10, "RWU": 11, "BU": 12, "OUT": 13}


def remove_features(payers, feature_ids):
    """
    function for removing some feature from data
    :param payers: 3D tensor of data (# players, # features, time)
    :param feature_ids: id of feature we want to remove
    :output: 3D tensor of clean data (# players, # features, time)
    """
    to_stay = sorted(list(set(range(payers.shape[1])) - set(feature_ids)))

    return payers[:, to_stay, :]


def make_output_payers(payers, predicted_period=7):
    """
    function for making output values using feature GB
    :param payers: 3D tensor of data (# players, # features, time)
    :param predicted_period: # of predicted period in the feature
    :output: 2D tensor of grouped data of GB values (# players, time - predicted_period + 1)
    """
    np, nf, nt = payers.shape
    out_payer7 = torch.zeros(np, nt - predicted_period + 1)
    out_payer5 = torch.zeros(np, nt - predicted_period + 1)
    out_payer3 = torch.zeros(np, nt - predicted_period + 1)

    for i in range(7):
        out_payer7 += payers[:, DAU, i: nt - predicted_period + 1 + i]
    for i in range(5):
        out_payer5 += payers[:, DAU, i: nt - predicted_period + 1 + i]
    for i in range(3):
        out_payer3 += payers[:, DAU, i: nt - predicted_period + 1 + i]

    return out_payer7, out_payer5, out_payer3 


def train_test_val_split(payers, out_payers7, out_payers5, out_payers3, seed, tr_prc=.7, te_prc=.2):
    """
    function for splitting data into train, test and validation part
    :param payers: 3D tensor of data (# players, # features, time)
    :param out_payers: 2D tensor of grouped data of GB values (# players, time - predicted_period + 1)
    :param seed: seed values
    :param tr_prc: percent value of train data
    :param te_prc: percent value of test data
    :output: 3D and 2D tensors of train, test and validation set of data
    """
    torch.manual_seed(seed)
    idx = torch.randperm(payers.shape[0])
    
    # # of data in train and test set
    n_tr, n_te = int(tr_prc * len(idx)), int((tr_prc + te_prc) * len(idx))
    
    # index of train, test and validation part
    id_tr, id_te, id_val = idx[:n_tr], idx[n_tr:n_te], idx[+n_te:]

    return payers[id_tr], payers[id_te], payers[id_val], out_payers7[id_tr], out_payers7[id_te], out_payers7[id_val], out_payers5[id_tr], out_payers5[id_te], out_payers5[id_val], out_payers3[id_tr], out_payers3[id_te], out_payers3[id_val]


def make_cluster(vec_data, seed):
    """
    function for clustering 1D data using KMeans algorithm into 2 classes
    :param vec_data: vector of data
    :param seed: seed value
    :output: centroids of clusters
    """
    # tensor of 2D data with one zero rows
    vector_data = torch.zeros((len(vec_data), 2))
    vector_data[:, 0] = vec_data
    
    # random seed
    random.seed(seed)
    k_means_instance = KMeans(n_clusters=2).fit(vector_data)
    return torch.tensor(k_means_instance.cluster_centers_[:, 0])


def feature_stats(payers, out_payers7, out_payers5, out_payers3, set_features, seed):
    """
    function for creating centroids of clusters and boundaries for outliers (for all features)
    :param payers: 3D tensor of data (# players, # features, time)
    :param out_payers: 2D tensor of grouped data of GB values (# players, time - predicted_period)
    :param set_features: list of all features in payers
    :param seed: seed value
    :output: dictionary of features with centroids of clusters and boundaries for outliers
    """
    f_boundaries = {}
    # flag of active days
    active = payers[:, DAU, :] == 1

    for f in set_features:
        if f == 'OUT':
            vals = out_payers7[out_payers7 != 0].ravel()
        else:
            vals = payers[:, set_features[f], :][active].ravel()
            vals = vals[vals != 0]

        # Q_25, IQR, Q_7
        q25, q75 = torch.quantile(vals, .25), torch.quantile(vals, .75)
        iqr = q75 - q25
        outlier_bound_up, outlier_bound_down = q75 + 1.5 * iqr, q25 - 1.5 * iqr
        
        non_outlier_vals = (vals > outlier_bound_down) & (vals < outlier_bound_up) 

        # centorids
        if f == 'OUT':
            centroids = [0, 0]
        else:
            centroids = make_cluster(vals[non_outlier_vals], type_of_clustering, seed)
            # add information about centroids and boundaries for outlier in dictionary
            f_boundaries[f] = [centroids[0], centroids[1], outlier_bound_up]

                
        # print information
        print('Feature {f}:')
        print('Type of clustering: {type_of_clustering}')
        print('Type of outliers: {type_of_outliers}')
        print(f_boundaries[f])

    return f_boundaries


def tokenize(payers, out_payers7, out_payers5, out_payers3, set_features, f_boundaries):
    """
    functions to tokenize data using centroids and boundaries for outliers
    :param payers: 3D tensor of data (# players, # features, time)
    :param out_payers: 2D tensor of grouped data of GB values (# players, time - predicted_period + 1)
    :param set_features: list of all features in payers
    :param f_boundaries: dictionary of features with centroids of clusters and boundaries for outliers
    :output: tokenized payers (# players, # features, time), out_tokenized payers (# players, time - predicted_period + 1)
    """
    np, nf, nt = payers.shape
    np_out, nt_out = out_payers7.shape
    
    tokenized_payers, out_tokenized_payers7 = torch.zeros(np, nf, nt), torch.zeros(np_out, nt_out)
    out_tokenized_payers5, out_tokenized_payers3 = torch.zeros(np_out, nt_out), torch.zeros(np_out, nt_out)
    
    # information of DAU is unchanged
    tokenized_payers[:, DAU, :] = payers[:, DAU, :]
    # information about active days
    active = payers[:, DAU, :] == 1
    # information about T
    t_plane = payers[:, T, :] == 0
    # token for padding is 0
    
    # tokens for Transactions are 1, 2 and 3
    tokenized_payers[:, T, :][~active] = 1
    tokenized_payers[:, T, :][active & t_plane] = 2
    tokenized_payers[:, T, :][active & ~t_plane] = 3

    tok_id = 4
    for f in set_features:
        if f != 'OUT':
            f_ind = set_features[f]
            f_plane = payers[:, f_ind, :]
            
            # token for inactive days
            tokenized_payers[:, f_ind, :][~active] = tok_id
            tok_id += 1
            
            # positions of outliers
            cluster_3 = f_plane >= f_boundaries[f][2]
            
            # positions of clusters 
            cluster_1 = f_plane <= (f_boundaries[f][0] + f_boundaries[f][1]) / 2
            cluster_2 = (~cluster_1) & (~cluster_3)
            
            # first cluster
            tokenized_payers[:, f_ind, :][(~active & cluster_1)] = tok_id
            tok_id += 1
            # second cluster
            tokenized_payers[:, f_ind, :][(~active & cluster_2)] = tok_id
            tok_id += 1
            # third cluster
            tokenized_payers[:, f_ind, :][(~active & cluster_3)] = tok_id
            tok_id += 1
        else:
            cluster_0_7 = out_payers7 == 0
            cluster_0_5 = out_payers5 == 0
            cluster_0_3 = out_payers3 == 0
            
            out_tokenized_payers7[~cluster_0_7] = 1
            out_tokenized_payers5[~cluster_0_5] = 1
            out_tokenized_payers3[~cluster_0_3] = 1


    return tokenized_payers, out_tokenized_payers7, out_tokenized_payers5, out_tokenized_payers3


def create_histories(payers, tokenized_payers, out_tokenized_payers7, out_tokenized_payers5, out_tokenized_payers3, seed,
                     max_inactive_period=30, predicted_period=7, sampling_pct=0.5):
    """
    function for creating input and output for transformer model
    :param payers: 3D tensor of input data (# players, # features, time)
    :param tokenized_payers: tokenized 3D tensor of input data (# players, # features, time)
    :param out_tokenized_payers: tokenized 2D tensor of output data (# players, time - predicted_period + 1)
    :param seed: seed value
    :param max_inactive_period: maximal inactive period in history
    :param predicted_period: # of predicted days in transformer model
    :param sampling_pct: percent of samples from one payer
    :output: X ( for transformer, RF statistics, RF tokens) and Y
    """
    # inputs for transformer and fixed input models
    x_transformer, x_fixed_tokens, x_fixed_statistics = [], [], []
    # outputs for all models
    y7, y5, y3 = [], [], []
    
    # shape of tokenized payers 3D tensor
    np, nf, nt = tokenized_payers.shape
    # we will remove DAU from feature
    nf, all_days = nf - 1, torch.arange(nt)

    for i in range(np):
    
        payer, t_payer, out_t_payer7, out_t_payer5, out_t_payer3 = payers[i], tokenized_payers[i], out_tokenized_payers7[i], out_tokenized_payers5[i], out_tokenized_payers3[i]
        # first transaction day
        first_t_day = max(t_payer[DAU].argmax().item(), 1)
        # vector of active days
        all_active_days = all_days[t_payer[DAU] != 0]
        
        if len(all_active_days) > 0:
        
            # upper bound for sample
            upper_bound = min(max(all_active_days) + max_inactive_period, nt - predicted_period)
            # samples are between first_t_day and upper_bound
            sampling_range = range(first_t_day, upper_bound)
            # number of samples of payer
            num_samples = int(len(sampling_range) * sampling_pct)

            random.seed(seed)
            for idx in random.sample(sampling_range, num_samples):
            
                # part for output (Y)
                label = out_t_payer7[idx + 1]
                y7.append(label.reshape(1))
                label = out_t_payer5[idx + 1]
                y5.append(label.reshape(1))
                label = out_t_payer3[idx + 1]
                y3.append(label.reshape(1))

                # part for X
                temp_tok_payer = t_payer[1:, :idx + 1].T.ravel()

                # part for transformer input (X_transformer)
                # # of time x # of features (without DAU)
                x = torch.zeros(1, nt * nf)
                x[0, :(idx + 1) * nf] = temp_tok_payer
                x_transformer.append(x)

                # part for fixed tokens input (X_fixed_tokens)
                x = torch.zeros(1, 4 * nf)
                for j in range(4 * nf):
                    x[0, j] = sum(temp_tok_payer == j) / idx
                x_fixed_tokens.append(x)

                # part for RF statistics input (X_rf_statistics)
                x = torch.zeros(1, 5 * nf + 2)

                # number of days and percent of active days
                x[0, 0], x[0, 1] = idx, sum(t_payer[0, :idx + 1] == 1) / idx

                temp_payer = payer[1:, :idxd + 1]
                for j in range(nf):
                    # statistics: min, Q_25, medina, Q_75, max 
                    x[0, 2 + j * 5] = torch.min(temp_payer[j])
                    x[0, 3 + j * 5] = torch.quantile(temp_payer[j], 0.25)
                    x[0, 4 + j * 5] = torch.median(temp_payer[j])
                    x[0, 5 + j * 5] = torch.quantile(temp_payer[j], 0.75)
                    x[0, 6 + j * 5] = torch.max(temp_payer[j])
                    
                x_fixed_statistics.append(x)
                
    return torch.cat(x_transformer).int(), torch.cat(x_fixed_tokens), torch.cat(x_fixedf_statistics), \
           torch.cat(y7).int().type(torch.LongTensor), torch.cat(y5).int().type(torch.LongTensor), torch.cat(y5).int().type(torch.LongTensor)


def prepare_data_for_models(payers, seed, history_len=61):
    """
    function for preparing data for models
    :param payers: .pt data
    :param seed: seed value
    :param history_len: maximal history of data per payers
    """
    out_payers7, out_payers5, out_payers3 = make_output_payers(payers)
    
    # train, test and validation split
    payers_tr, payers_te, payers_va, out_payers_tr7, out_payers_te7, out_payers_va7, out_payers_tr5, out_payers_te5, out_payers_va5, out_payers_tr3, out_payers_te3, out_payers_va3 = train_test_val_split(payers, out_payers7, out_payers5, out_payers3, seed=seed)
    # cluster centroids and bounds for outliers
    boundaries = feature_stats(payers_tr, out_payers_tr7, out_payers_tr5, out_payers_tr3, FEATURES, seed)
    
    # tokenize input and output
    payers_tr_tok, out_payers_tr_tok7, out_payers_tr_tok5, out_payers_tr_tok3 = tokenize(payers_tr, out_payers_tr7, out_payers_tr5, out_payers_tr3, FEATURES, boundaries)
    payers_te_tok, out_payers_te_tok7, out_payers_te_tok5, out_payers_te_tok3 = tokenize(payers_te, out_payers_te7, out_payers_te5, out_payers_te3, FEATURES, boundaries)
    payers_va_tok, out_payers_va_tok7, out_payers_va_tok5, out_payers_va_tok3 = tokenize(payers_va, out_payers_va7, out_payers_va5, out_payers_va3, FEATURES, boundaries)
    
    # create history for train, test and validation data
    tr_x_0, tr_x_1, tr_x_2, tr_y_7, tr_y_5, tr_y_3 = create_histories(payers_tr, payers_tr_tok, out_payers_tr_tok7, out_payers_tr_tok5, out_payers_tr_tok3, seed)
    te_x_0, te_x_1, te_x_2, te_y_7, te_y_5, te_y_3 = create_histories(payers_te, payers_te_tok, out_payers_te_tok7, out_payers_te_tok5, out_payers_te_tok3, seed)
    va_x_0, va_x_1, va_x_2, va_y_7, va_y_5, va_y_3 = create_histories(payers_va, payers_va_tok, out_payers_va_tok7, out_payers_va_tok5, out_payers_va_tok3, seed)

    # save output data
    torch.save(te_y_7, f"te_y7_{seed}.pt")
    torch.save(tr_y_7, f"tr_y7_{seed}.pt")
    torch.save(va_y_7, f"va_y7_{seed}.pt")
    torch.save(te_y_5, f"te_y5_{seed}.pt")
    torch.save(tr_y_5, f"tr_y5_{seed}.pt")
    torch.save(va_y_5, f"va_y5_{seed}.pt")
    torch.save(te_y_3, f"te_y3_{seed}.pt")
    torch.save(tr_y_3, f"tr_y3_{seed}.pt")
    torch.save(va_y_3, f"va_y3_{seed}.pt")
    
    # save input data
    torch.save(tr_x_0, f"tr_x_0_{seed}.pt")
    torch.save(tr_x_1, f"tr_x_1_{seed}.pt")
    torch.save(tr_x_2, f"tr_x_2_{seed}.pt")
    torch.save(te_x_0, f"te_x_0_{seed}.pt")
    torch.save(te_x_1, f"te_x_1_{seed}.pt")
    torch.save(te_x_2, f"te_x_2_{seed}.pt")
    torch.save(va_x_0, f"va_x_0_{seed}.pt")
    torch.save(va_x_1, f"va_x_1_{seed}.pt")
    torch.save(va_x_2, f"va_x_2_{seed}.pt")


if __name__ == "__main__":

    # read data about payers
    int_payers = remove_features(torch.load(f"dataOfPayers.pt"), (10, 11))
    # output values
    cm_t, cm_rf_1, cm_rf_2 = {}, {}, {}

    # device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # seed values
    seed_l = [399, 1054, 1175, 1475, 1862, 1903, 1990, 2000, 2023, 2053]
    
    for seed in seed_l:
        prepare_data_for_models(int_payers, seed)
