"""
Showing how to use the model with some time series data.

NB! This is not a full training loop. You have to write the training loop yourself. 

I.e. this code is just a starting point to show you how to initialize the model and provide its inputs

If you do not know how to train a PyTorch model, it is too soon for you to dive into transformers imo :) 

You're better off starting off with some simpler architectures, e.g. a simple feed forward network, in order to learn the basics
"""
from torch import nn
from torch.optim.lr_scheduler import StepLR

import dataset as ds
import utils
from torch.utils.data import DataLoader
import torch
import datetime
import transformer_timeseries as tst
import numpy as np
import torch as th

# Hyperparams
test_size = 0.1
batch_size = 128
target_col_name = "FCR_N_PriceEUR"
timestamp_col = "timestamp"
# Only use data from this date and onwards
cutoff_date = datetime.datetime(2017, 1, 1) 

## Params
n_heads = 8
n_decoder_layers = 4
n_encoder_layers = 4
dec_seq_len = 90 # length of input given to decoder
enc_seq_len = 100 # length of input given to encoder
output_sequence_length = 4 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
window_size = enc_seq_len + output_sequence_length # used to slice data into sub-sequences
step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step
in_features_encoder_linear_layer = 2048
in_features_decoder_linear_layer = 2048
max_seq_len = enc_seq_len
batch_first = True

cuda = "cuda"
# Define input variables 
exogenous_vars = [] # should contain strings. Each string must correspond to a column name
input_variables = [target_col_name] + exogenous_vars
target_idx = 0 # index position of target in batched trg_y

input_size = len(input_variables)

# Read data
data = utils.read_data(timestamp_col_name=timestamp_col)

# Remove test data from dataset
training_data = data[:-(round(len(data)*test_size))]

# Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chunkc. 
# Should be training data indices only
training_indices = utils.get_indices_entire_sequence(
    data=training_data, 
    window_size=window_size, 
    step_size=step_size)

trainint_datasets = training_data[input_variables].values
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))


trainint_datasets = scaler.fit_transform(trainint_datasets)

cuda = "cuda"
training_data = ds.TransformerDataset(
    data=torch.tensor(trainint_datasets, device=cuda).float(),
    indices=training_indices,
    enc_seq_len=enc_seq_len,
    dec_seq_len=dec_seq_len,
    target_seq_len=output_sequence_length
    )
# Making dataloader



i, batch = next(enumerate(training_data))

src, trg, trg_y = batch
# Permute from shape [batch size, seq len, num features] to [seq len, batch size, num features]


model = tst.TimeSeriesTransformer(
    input_size=len(input_variables),
    dec_seq_len=enc_seq_len,
    batch_first=batch_first,
    num_predicted_features=1
    )

# Make src mask for decoder with size:
# [batch_size*n_heads, output_sequence_length, enc_seq_len]
src_mask = utils.generate_square_subsequent_mask(
    dim1=output_sequence_length,
    dim2=enc_seq_len
    ).to(cuda)

# Make tgt mask for decoder with size:
# [batch_size*n_heads, output_sequence_length, output_sequence_length]
tgt_mask = utils.generate_square_subsequent_mask( 
    dim1=output_sequence_length,
    dim2=output_sequence_length
    ).to(cuda)


model = tst.TimeSeriesTransformer(
    input_size=len(input_variables),
    dec_seq_len=enc_seq_len,
    out_seq_len =output_sequence_length,
    num_predicted_features=1,
    batch_first = batch_first
    ).to(cuda)

# i, batch = next(enumerate(training_data))
#
# for batch in training_data:
#     src, trg, trg_y = batch
#     out = model.pred(src, trg)
#     print(out)


#
#
# #
def th_fit(predictor, dataset, loss_func, metrc, optimizer, epoch, batch, validataion_split):
    data_loader = DataLoader(dataset, batch_size=batch)
    valid_size = (dataset.__len__()/batch * validataion_split)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)


    for step in range(epoch+1):
        scheduler.step()
        valid_set = []
        train_set = []

        predictor.train()
        for i, t_set  in enumerate(data_loader):

            src, trg, trg_y = t_set
            if  len(valid_set) < valid_size :
                valid_set.append(t_set)
                continue

            train_set.append(t_set)

            hypothesis = predictor(src, trg, src_mask, tgt_mask)
            loss = loss_func(hypothesis, trg_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        predictor.eval()
        with th.no_grad():
            train = [(predictor(src, trg, src_mask, tgt_mask), trg_y) for src, trg, trg_y in train_set]
            valid = [(predictor(src, trg, src_mask, tgt_mask), trg_y) for src, trg, trg_y in valid_set]
            train_loss =np.mean([loss_func(xb, yb).cpu() for xb, yb in train])
            train_metric = np.mean([metrc(xb, yb) for xb, yb in train])

            valid_loss = np.mean([loss_func(xb, yb).cpu() for xb, yb in valid])
            valid_metric = np.mean([metrc(xb, yb) for xb, yb in valid])
            result = [ str(round(x, 6)) for x in [train_loss, train_metric, valid_loss, valid_metric]]

            print("EPOCH {} : t_loss {}\tt_metrc {}\tval_loss {}\tval_metric {}".format(step, *result))


    return predictor


def ScaledRMSE(scaler):
    def RMSE(yhat,y):
        yhat = yhat.squeeze(-1)
        y = y.squeeze(-1)
        yhat = yhat[:,-1:]
        y = y[:,-1:]
        yhat = scaler.inverse_transform(yhat.cpu())
        y = scaler.inverse_transform(y.cpu())
        return np.sqrt(np.mean((yhat-y)**2))
    return RMSE


optimizer = th.optim.Adam(model.parameters(), lr=0.0002)
# loss_func = nn.BCELoss()
loss_func = nn.MSELoss().to(cuda)
max = 100
validation_split = 0.1
th_fit(model, training_data, loss_func, ScaledRMSE(scaler), optimizer, epoch=max, batch=128, validataion_split=0.1)
