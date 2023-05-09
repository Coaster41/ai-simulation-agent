import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
# import autogluon 
# from IPython.display import display

device = torch.device('cuda:0') # change as req'd
df = pd.read_csv('results/results_100_100000_new.csv', header=None)
model_num = 1
# display(df)
results = df.to_numpy()
print(results.shape)
print(results[:,-1].shape)
X = torch.tensor(results[:,:-1]).to(torch.float32).to(device)
if model_num == 1:
    X = torch.reshape(X,(X.shape[0],1,10,10))
y = torch.tensor(np.expand_dims(results[:,-1], axis=1)).to(torch.float32).to(device)
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape, y_train.shape)
# Setting up Dataloader
# training_dataset = TensorDataset(X_train, y_train)
# train_loader = DataLoader(training_dataset, batch_size=24, shuffle=True)

# from autogluon docs
from autogluon.tabular import TabularPredictor, TabularDataset
X_train_flat = X_train.flatten(start_dim=1)

train_data = pd.DataFrame(X_train_flat).astype('float')
train_data['result'] = y_train

X_test_flat = X_test.flatten(start_dim = 1)
test_data = pd.DataFrame(X_test_flat).astype('float')
test_data['result'] = y_test




# need to convert to pandas dataframe and train
# get label
label = 'result'

predictor = TabularPredictor(label=label, problem_type='regression').fit(train_data)


y_pred_auto = predictor.predict(test_data.drop(columns=[label]))
y_pred_auto.head()

predictor.evaluate(test_data, silent=True)