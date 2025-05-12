import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from rna_global import MyLibraryDataset, load_data_from_pkl, split_dataset, RMSELoss, device
import numpy as np

# Dados
file_path = 'data_rna_cobeq_100k.pkl'
library_data = load_data_from_pkl(file_path)

feature_vars = ['p_topo', 'valve1', 'valve2', 'valve3', 'valve4',
                'bcs1_freq', 'bcs2_freq', 'bcs3_freq', 'bcs4_freq', 'booster_freq']
label_vars = ['q_main1', 'q_main2', 'q_main3', 'q_main4', 'q_tr',
              'P_man', 'P_fbhp1', 'P_fbhp2', 'P_fbhp3', 'P_fbhp4',
              'dP_bcs1', 'dP_bcs2', 'dP_bcs3', 'dP_bcs4']

dataset = MyLibraryDataset(library_data, feature_vars, label_vars)
train_dataset, valid_dataset = split_dataset(dataset, train_ratio=0.7)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Otimização
def objective(trial):
    # Hiperparâmetros a otimizar
    hidden_size = trial.suggest_int('hidden_size', 32, 75)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

    # Definir modelo
    class TrialNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(len(feature_vars), hidden_size),
                nn.Tanh(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, len(label_vars))
            )

        def forward(self, x):
            return self.model(x)

    model = TrialNetwork().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossfunc = RMSELoss()

    # Treinamento simples
    def train_epoch():
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = lossfunc(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate():
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for X, y in valid_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = lossfunc(pred, y)
                total_loss += loss.item()
        return total_loss / len(valid_loader)

    for epoch in range(10):  # menor número de épocas p/ acelerar a busca
        train_epoch()

    val_loss = validate()
    return val_loss

# Iniciar otimização
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

# Mostrar melhores parâmetros
print("Melhores parâmetros encontrados:")
print(study.best_params)
