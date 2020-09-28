import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opti

from sklearn.preprocessing import OneHotEncoder



def load_prepare_data():
    # load data
    data = pd.read_csv("data/lfm1b.csv")
    ids = pd.read_csv("data/fold1.csv")
    # subset for debuging
    # REMOVE
    ids = ids.iloc[:100]
    train_data = data.loc[data.sessionId.isin(ids.sessionId.values)]

    items = train_data.filter(['itemId']).drop_duplicates().sort_values(['itemId']).reset_index(drop=True)
    items['new_itemId'] = items.index
    print("items: {}".format(items.shape))
    sessions = train_data.filter(['sessionId']).drop_duplicates().sort_values(['sessionId']).reset_index(drop=True)
    sessions['new_sessionId'] = sessions.index
    print("sessions: {}".format(sessions.shape))
    users = train_data.filter(['userId']).drop_duplicates().sort_values(['userId']).reset_index(drop=True)
    users['new_userId'] = users.index
    print("users: {}".format(users.shape))

    train_data = train_data.merge(items, on="itemId", how="inner")
    train_data = train_data.merge(sessions, on="sessionId", how="inner")
    train_data = train_data.merge(users, on="userId", how="inner")

    train_data = train_data.filter(['new_userId','new_sessionId','new_itemId','time'])
    train_data.rename(columns = {'new_userId':'userId','new_itemId':'itemId','new_sessionId':'sessionId'}, inplace = True)

    return train_data

def sliding_windows(data):
    x = []
    y = []

    # for i in range(len(data)-seq_length-1):
    for sessionId in data.sessionId.unique():
        session_df = data.loc[data.sessionId == sessionId]
        if len(session_df) >= 2:
            for i in range(len(session_df)-1):
                _x = session_df.iloc[i].itemId
                _y = session_df.iloc[i+1].itemId
                x.append(_x)
                y.append(_y)

    return np.array(x),np.array(y)

def main():

    train_data = load_prepare_data()
    print(train_data)

    # Unique items
    alphabet = np.array(train_data.itemId.unique())

    # input and target
    x,y = sliding_windows(train_data)
    print(x.shape)
    print(y.shape)

    # define one-hot encoder and label encoder
    onehot_encoder = OneHotEncoder(sparse=False,categories='auto').fit(alphabet.reshape(-1, 1))
    label_encoder  = {ch: i for i, ch in enumerate(alphabet)}
    
    
    # Transform input and targets
    x = onehot_encoder.transform(x.reshape(-1,1))
    #y = [label_encoder[ch] for ch in y]
    #y = torch.tensor(y)
    y = onehot_encoder.transform(y.reshape(-1,1))
    print(x.shape)
    print(y.shape)


    torch.manual_seed(1)
    len_alphabet = alphabet.shape[0]
    lstm = nn.LSTM(len_alphabet, len_alphabet)
    
    # initialize the hidden state.
    hidden = (torch.randn(1, 1, len_alphabet),
              torch.randn(1, 1, len_alphabet))

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)


    x = torch.tensor(x).type(torch.FloatTensor)
    y = torch.tensor(y).type(torch.FloatTensor)

    cuda = torch.cuda.is_available()

    loss_list=[]
    for epoch in range(10):
        loss_epoch = 0.0
        for i in range(x.shape[0]):
            # Step through the sequence one element at a time.
            # after each step, hidden contains the hidden state.
            out, hidden = lstm(x[i].view(1, 1, -1), hidden)
            optimizer.zero_grad()
            loss = criterion(out.type(torch.FloatTensor), y[i].view(1,1,-1))
    
            loss_epoch += loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()

        print("epoch {} loss: {}".format(epoch, loss_epoch/x.shape[0]))
        loss_list.append(loss_epoch/x.shape[0])

    print(loss_list)

if __name__ == '__main__':
    main()
