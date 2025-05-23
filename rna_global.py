# -*- coding: utf-8 -*-
"""RNA_global

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dfFRua5xT_5PQqL4WxXYPPnwD7XFoOGX
"""
#%% importações do código
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
from matplotlib.style.core import library
from torch.utils.data import Dataset, DataLoader, random_split
import torch.distributions.uniform as urand
import pickle
import matplotlib.pyplot as plt
# from colorama import Fore, Style


def weighted_mse_loss(output, target, weights):
    """
    output: Tensor com as previsões do modelo (batch_size, n_outputs)
    target: Tensor com os valores reais (batch_size, n_outputs)
    weights: Tensor com pesos por variável (n_outputs)
    """
    return ((weights * (output - target) ** 2)).mean()

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return torch.sqrt(nn.functional.mse_loss(pred, target))

def load_data_from_pkl(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def split_dataset(dataset, train_ratio=0.6):
    total_len = len(dataset)
    assert total_len == 100_000, f"Dataset deve ter 50k amostras (atual: {total_len})"

    train_len = int(total_len * train_ratio)
    test_len = total_len - train_len
    return random_split(dataset, [train_len, test_len])


class MyLibraryDataset(Dataset):
    def __init__(self, library_data, feature_vars, label_vars, transform=None):
        self.library_data = library_data
        self.feature_vars = feature_vars  # Variáveis de entrada
        self.label_vars = label_vars  # Variáveis de saída (inclui a flag)
        self.num_simulations = len(library_data[feature_vars[0]])
        self.transform = transform

        # Calcular limites de normalização
        self.feature_min = {var: min(library_data[var]) for var in feature_vars}
        self.feature_max = {var: max(library_data[var]) for var in feature_vars}
        self.label_min = {var: min(library_data[var]) for var in label_vars if var != 'flag'}
        self.label_max = {var: max(library_data[var]) for var in label_vars if var != 'flag'}

    def normalize(self, value, min_val, max_val):
        if max_val == min_val:
            return 0  # Retorna um valor padrão para evitar a divisão por zero
        return 2 * (value - min_val) / (max_val - min_val) - 1

    def denormalize(self, value, min_val, max_val):
        return (value + 1) * (max_val - min_val) / 2 + min_val # Reverter do intervalo [-1, 1] para o intervalo original

    def __getitem__(self, idx):
        if idx >= 100_000:  # Garantir acesso válido
            raise IndexError(f"Índice {idx} inválido para dataset com 100k amostras!")

        features = [
            self.normalize(self.library_data[var][idx], self.feature_min[var], self.feature_max[var])
            for var in self.feature_vars
        ]


        labels = [
            self.library_data[var][idx] if var == 'flag' else
            self.normalize(self.library_data[var][idx], self.label_min[var], self.label_max[var])
            for var in self.label_vars # Normalizar labels, exceto a flag
        ]

        # Aplicar transformações, se houver
        if self.transform:
            features = self.transform(features)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return self.num_simulations


# Definir a rede neural
class RasmusNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        layers = []  # Lista temporária para criar o Sequential
        hidden_size = 32  # Hiperparâmetro encontrado

        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, output_dim))

        # Definir como um módulo Sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        output = self.model(x)
        return output

    # Treinamento
def train(model, dataloader, optimizer, lossfunc):
    model.train()
    cumloss = 0.0

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        y_labels = y # Saídas contínuas

        pred = model(X)
        loss = lossfunc(pred, y)
        pred_labels = pred

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Acumular a perda
        cumloss += loss.item()

        y_labels_denorm = torch.stack([
            dataset.denormalize(y_labels[:, i], dataset.label_min[var], dataset.label_max[var])
            for i, var in enumerate(dataset.label_vars)
        ], dim=1)
        pred_labels_denorm = torch.stack([
            dataset.denormalize(pred_labels[:, i], dataset.label_min[var], dataset.label_max[var])
            for i, var in enumerate(dataset.label_vars)
        ], dim=1)
        y_labels_list = []
        pred_labels_list = []
        y_labels_list.append(y_labels_denorm)
        pred_labels_list.append(pred_labels_denorm)

    y_labels_all = torch.cat(y_labels_list, dim=0)
    pred_labels_all = torch.cat(pred_labels_list, dim=0)
    avg_loss = cumloss / len(dataloader)
    return avg_loss, y_labels_all, pred_labels_all

def test(model, dataloader, lossfunc):
    model.eval()  # Coloca o modelo em modo de avaliação
    cumloss = 0.0
    y_labels_list = []
    pred_labels_list = []

    with torch.no_grad():  # Desativa o cálculo do gradiente para economizar memória
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            y_labels = y  # Saídas contínuas

            pred = model(X)
            loss = lossfunc(pred, y)
            pred_labels = pred

            # Acumular a perda
            cumloss += loss.item()

            y_labels_denorm = torch.stack([
                dataset.denormalize(y_labels[:, i], dataset.label_min[var], dataset.label_max[var])
                for i, var in enumerate(dataset.label_vars)
            ], dim=1)
            pred_labels_denorm = torch.stack([
                dataset.denormalize(pred_labels[:, i], dataset.label_min[var], dataset.label_max[var])
                for i, var in enumerate(dataset.label_vars)
            ], dim=1)

            y_labels_list.append(y_labels_denorm)
            pred_labels_list.append(pred_labels_denorm)

    y_labels_all = torch.cat(y_labels_list, dim=0)
    pred_labels_all = torch.cat(pred_labels_list, dim=0)
    avg_loss = cumloss / len(dataloader)
    return avg_loss, y_labels_all, pred_labels_all


if __name__ == "__main__":
    file_path = 'data_rna_cobeq_100k.pkl'
    library_data = load_data_from_pkl(file_path)

    # Variáveis selecionadas
    feature_vars = [
        'p_topo', 'valve1', 'valve2', 'valve3', 'valve4',
        'bcs1_freq', 'bcs2_freq', 'bcs3_freq', 'bcs4_freq',
        'booster_freq',
    ]
    label_vars = ['q_main1', 'q_main2', 'q_main3', 'q_main4', 'q_tr',
        'P_man', 'P_fbhp1', 'P_fbhp2',
        'P_fbhp3', 'P_fbhp4', 'dP_bcs1', 'dP_bcs2',
        'dP_bcs3', 'dP_bcs4']

    for var in feature_vars + label_vars:
        if var not in library_data:
            raise ValueError(f"A variável {var} não está presente no dataset!")

    input_dim = len(feature_vars)  # Dimensão da entrada
    output_dim = len(label_vars)  # Dimensão da saída
    dataset = MyLibraryDataset(library_data, feature_vars, label_vars)
    train_dataset, test_dataset = split_dataset(dataset, train_ratio=0.7)

    print(f"Tamanho total do dataset: {len(dataset)}")
    print(f"Tamanho do treino: {len(train_dataset)}")
    print(f"Tamanho do teste: {len(test_dataset)}")
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # Sem shuffle!

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Rodando na {device}")

    # Criar o modelo
    model = RasmusNetwork(input_dim=len(feature_vars), output_dim=len(label_vars)).to(device)
    # state_dict = torch.load('rna_global_model_sbai.pth')
    # Carregue os pesos no modelo inicializado
    # model.load_state_dict(state_dict)
    optimizer = optim.Adam(model.parameters(), lr=0.002207677278010547)
    lossfunc = lambda output, target: weighted_mse_loss(output, target, weights)

saidas =  ['q_main1', 'q_main2', 'q_main3', 'q_main4', 'q_tr',
        'P_man', 'P_fbhp1', 'P_fbhp2',
        'P_fbhp3', 'P_fbhp4', 'dP_bcs1', 'dP_bcs2',
        'dP_bcs3', 'dP_bcs4']

#%%
from colorama import Fore, Style
teste = 0


def SaveNetwork():
    model_path = "rna_global_trained_cobeq_teste.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Modelo completo salvo em {model_path}")

train_list = []
test_list = []
epochs = 150  # Número de épocas
# Define os pesos: 5x mais importante para p_man e q_tr (índices 2 e 3)
weights = torch.tensor([
    3.0, 3.0, 3.0, 3.0,  # q_main1–4
    3.0,                # q_tr
    1.0,                # P_man
    3.0, 3.0, 3.0, 3.0,  # P_fbhp1–4
    1.0, 1.0, 1.0, 1.0   # dP_bcs1–4
], device=device)
if teste == 1:
    for epoch in range(epochs):
        train_loss, y_labels, pred_labels = train(model, train_dataloader, optimizer, lossfunc)
        train_list.append(train_loss)
        test_loss, y_labels, pred_labels = test(model, test_dataloader, lossfunc)
        test_list.append(test_loss)
        y_labels = y_labels.tolist()
        pred_labels = pred_labels.tolist()
        if test_loss < 1e-5 and train_loss < 1e-5:
            SaveNetwork()
            break

        if epoch % 10 == 0:
            print("=" * 58)
            print(f"Epoch {epoch}: Train Loss = {train_loss}")
            print("=" * 58)
            print(f"{'Saída':<15}{'Modelo':<15}{'RNA':<15}{'Diferença (%)':<15}")
            print("-" * 58)

            percent_total = 0
            c = 0

            for i, name in enumerate(saidas):
                # Calcular a diferença percentual
                percent = abs(abs(abs(y_labels[-1][i]) - abs(pred_labels[-1][i])) / abs(y_labels[-1][i])) * 100
                percent_total += percent
                c += 1

                # Cor para a diferença percentual
                if percent > 5:  # Diferença alta
                    color = Fore.RED
                elif percent > 1:  # Diferença média
                    color = Fore.YELLOW
                else:  # Diferença baixa
                    color = Fore.GREEN

                # Print formatado
                print(f"{name:<15}{y_labels[-1][i]:<15.2f}{pred_labels[-1][i]:<15.2f}{color}    {percent:<15.2f}{Style.RESET_ALL}")

            print("-" * 58)
            print(f"{'PERCENTUAL MÉDIO:':<15}{percent_total / c:.2f}%")
            print("=" * 58)
            print(test_loss)

    SaveNetwork()

    plt.figure(dpi=250)
    plt.plot(train_list, 'b')
    plt.plot(test_list, 'r')
    plt.xlabel("'Época", fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.grid()
    plt.show()

#
# import optuna
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import numpy as np
# # Melhores parâmetros encontrados:
# # {'hidden_size': 70, 'lr': 0.0035773984194597216, 'weight_decay': 0.00016610742884793064, 'dropout_rate': 0.17203235031213443}
# # Dados
# file_path = 'data_rna_cobeq_100k.pkl'
# library_data = load_data_from_pkl(file_path)
#
# feature_vars = ['p_topo', 'valve1', 'valve2', 'valve3', 'valve4',
#                 'bcs1_freq', 'bcs2_freq', 'bcs3_freq', 'bcs4_freq', 'booster_freq']
# label_vars = ['q_main1', 'q_main2', 'q_main3', 'q_main4', 'q_tr',
#               'P_man', 'P_fbhp1', 'P_fbhp2', 'P_fbhp3', 'P_fbhp4',
#               'dP_bcs1', 'dP_bcs2', 'dP_bcs3', 'dP_bcs4']
#
# dataset = MyLibraryDataset(library_data, feature_vars, label_vars)
# train_dataset, valid_dataset = split_dataset(dataset, train_ratio=0.7)
#
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# import optuna
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
#
# # Garantir que os loaders estão definidos corretamente
# train_dataset, valid_dataset = split_dataset(dataset, train_ratio=0.7)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
#
# # Função objetivo com estrutura igual à RasmusNetwork
# def objective(trial):
#     lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
#     hidden_size = 32
#     # Rede no estilo do RasmusNetwork
#     class TrialNetwork(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.model = nn.Sequential(
#                 nn.Linear(len(feature_vars), hidden_size),
#                 nn.Tanh(),
#                 nn.Linear(hidden_size, hidden_size),
#                 nn.Tanh(),
#                 nn.Linear(hidden_size, len(label_vars))
#             )
#
#         def forward(self, x):
#             return self.model(x)
#
#     model = TrialNetwork().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#
#     # Pesos como no treino principal
#     weights = torch.tensor([
#         3.0, 3.0, 3.0, 3.0,
#         3.0,
#         1.0,
#         3.0, 3.0, 3.0, 3.0,
#         1.0, 1.0, 1.0, 1.0
#     ], device=device)
#
#     lossfunc = lambda output, target: weighted_mse_loss(output, target, weights)
#
#     def train_epoch():
#         model.train()
#         total_loss = 0
#         for X, y in train_loader:
#             X, y = X.to(device), y.to(device)
#             optimizer.zero_grad()
#             pred = model(X)
#             loss = lossfunc(pred, y)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         return total_loss / len(train_loader)
#
#     def validate():
#         model.eval()
#         total_loss = 0
#         with torch.no_grad():
#             for X, y in valid_loader:
#                 X, y = X.to(device), y.to(device)
#                 pred = model(X)
#                 loss = lossfunc(pred, y)
#                 total_loss += loss.item()
#         return total_loss / len(valid_loader)
#
#     for _ in range(30):
#         train_epoch()
#
#     return validate()
#
# # Executar a otimização
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=100)
#
# print("Melhores parâmetros encontrados:")
# print(study.best_params)


# {'hidden_size': 75, 'lr': 0.0011067057965994955, 'weight_decay': 0.00012247004240116606, 'dropout_rate': 0.11243984096456594}

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Garantir que a rede está com os pesos carregados
# model = RasmusNetwork(input_dim=len(feature_vars), output_dim=len(label_vars)).to(device)
# model.load_state_dict(torch.load("rna_global_trained_cobeq.pth", map_location=device))
# model.eval()
#
# # DataLoader com TODOS os dados
# full_dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
#
# # Rodar a rede em todos os dados
# X_all = []
# with torch.no_grad():
#     for X, _ in full_dataloader:
#         X_all.append(X)
# X_all = torch.cat(X_all, dim=0).to(device)
#
# with torch.no_grad():
#     y_pred = model(X_all)
#
# # Dados reais (do dicionário)
# q_tr_real = np.array(library_data['q_tr'])
# pman_real = np.array(library_data['P_man'])
# flag = np.array(library_data['flag'])
#
# # Índices das variáveis
# i_q_tr = dataset.label_vars.index('q_tr')
# i_pman = dataset.label_vars.index('P_man')
#
# # Denormalizar predições
# q_tr_pred = dataset.denormalize(y_pred[:, i_q_tr], dataset.label_min['q_tr'], dataset.label_max['q_tr']).cpu().numpy()
# pman_pred = dataset.denormalize(y_pred[:, i_pman], dataset.label_min['P_man'], dataset.label_max['P_man']).cpu().numpy()
#
# # Plot estilo Rafael™️
# plt.figure(figsize=(14, 6), dpi=250)
#
# # Subplot 1: Dados reais
# plt.subplot(1, 2, 1)
# plt.plot(q_tr_real[flag == 0], pman_real[flag == 0], 'r.')
# plt.plot(q_tr_real[flag == 1], pman_real[flag == 1], 'b.')
# plt.plot([110, 225], [0, 0], 'k--', linewidth=3)
# plt.xlabel(r"$q_{tr}$ /(m$^3 \cdot$ h$^{-1}$)", fontsize=15)
# plt.ylabel(r"$P_{man}$ /bar", fontsize=15)
# plt.title("Banco de Dados (Real)", fontsize=14)
# plt.grid()
#
# # Subplot 2: Previsão da RNA
# plt.subplot(1, 2, 2)
# plt.plot(q_tr_pred[flag == 0], pman_pred[flag == 0], 'r.')
# plt.plot(q_tr_pred[flag == 1], pman_pred[flag == 1], 'b.')
# plt.plot([110, 225], [0, 0], 'k--', linewidth=3)
# plt.xlabel(r"$q_{tr}$ /(m$^3 \cdot$ h$^{-1}$)", fontsize=15)
# plt.ylabel(r"$P_{man}$ /bar", fontsize=15)
# plt.title("Previsão da RNA", fontsize=14)
# plt.grid()
#
# plt.tight_layout()
# plt.show()
#
#
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Recarregar modelo e pesos treinados
# model = RasmusNetwork(input_dim=len(feature_vars), output_dim=len(label_vars)).to(device)
# model.load_state_dict(torch.load("rna_global_trained_cobeq.pth", map_location=device))
# model.eval()
#
# # Criar DataLoader com TODOS os dados
# full_dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
#
# # Rodar a rede em todos os dados
# X_all = []
# with torch.no_grad():
#     for X, _ in full_dataloader:
#         X_all.append(X)
# X_all = torch.cat(X_all, dim=0).to(device)
#
# with torch.no_grad():
#     y_pred = model(X_all)
#
# # Pegar os 100 mil valores reais diretamente do dicionário
# q_main_real = np.array(library_data['q_main1'])
# dp_bcs_real = np.array(library_data['dP_bcs1'])
# flag = np.array(library_data['flag'])
#
# # Denormalizar predições
# i_q_main = dataset.label_vars.index('q_main1')
# i_dp_bcs = dataset.label_vars.index('dP_bcs1')
#
# q_main_pred = dataset.denormalize(y_pred[:, i_q_main], dataset.label_min['q_main1'], dataset.label_max['q_main1']).cpu().numpy()
# dp_bcs_pred = dataset.denormalize(y_pred[:, i_dp_bcs], dataset.label_min['dP_bcs1'], dataset.label_max['dP_bcs1']).cpu().numpy()
#
# # Plot no seu estilo, com 100 mil pontos
# plt.figure(figsize=(14, 6), dpi=250)
#
# # Real
# plt.subplot(1, 2, 1)
# plt.plot(q_main_real[flag == 0], dp_bcs_real[flag == 0], 'r.')
# plt.plot(q_main_real[flag == 1], dp_bcs_real[flag == 1], 'b.')
# plt.plot([28.55, 20.77], [206.6, 58.07], 'k--', linewidth=3)
# plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
# plt.xlabel(r"$q_{main1}$ /(m$^3 \cdot$ h$^{-1}$)", fontsize=15)
# plt.ylabel(r"$dP_{bcs1}$ /bar", fontsize=15)
# plt.title("Banco de Dados (Real)", fontsize=14)
# plt.grid()
#
# # RNA
# plt.subplot(1, 2, 2)
# plt.plot(q_main_pred[flag == 0], dp_bcs_pred[flag == 0], 'r.')
# plt.plot(q_main_pred[flag == 1], dp_bcs_pred[flag == 1], 'b.')
# plt.plot([28.55, 20.77], [206.6, 58.07], 'k--', linewidth=3)
# plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
# plt.xlabel(r"$q_{main1}$ /(m$^3 \cdot$ h$^{-1}$)", fontsize=15)
# plt.ylabel(r"$dP_{bcs1}$ /bar", fontsize=15)
# plt.title("Previsão da RNA", fontsize=14)
# plt.grid()
#
# plt.tight_layout()
# plt.show()
import torch

# 1. Instanciar o modelo com a mesma arquitetura usada no treino
model.load_state_dict(torch.load("rna_global_trained_cobeq_teste.pth"))
model.to(device)  # envia para GPU se estiver usando
model.eval()

def testar_sem_treinar(model, test_dataloader, lossfunc, saidas):
    """
    Realiza o teste de uma RNA treinada, calcula o desvio padrão médio e a loss.

    Parâmetros:
        model: modelo PyTorch já treinado
        test_dataloader: dataloader com os dados de teste
        lossfunc: função de perda
        saidas: lista de nomes das variáveis de saída

    Retorna:
        media_desvio_padrao: float - média dos desvios padrão dos erros
        test_loss: float - perda no conjunto de teste
    """
    import numpy as np
    from colorama import Fore, Style

    model.eval()  # Coloca em modo avaliação

    # Executa o teste (usa sua função já existente)
    test_loss, y_labels, pred_labels = test(model, test_dataloader, lossfunc)

    # Transforma listas em arrays NumPy
    y_labels = np.array(y_labels)
    pred_labels = np.array(pred_labels)

    # Calcula erros e desvio padrão
    erros = pred_labels - y_labels
    desvios = np.std(erros, axis=0)
    media_desvio_padrao = np.mean(desvios)

    # Impressão formatada
    print("=" * 58)
    print(f"{'Saída':<15}{'Real':<15}{'RNA':<15}{'Erro (%)':<15}")
    print("-" * 58)

    percent_total = 0
    for i, name in enumerate(saidas):
        real = y_labels[-1][i]
        pred = pred_labels[-1][i]
        percent = abs((real - pred) / real) * 100 if real != 0 else 0
        percent_total += percent

        if percent > 5:
            color = Fore.RED
        elif percent > 1:
            color = Fore.YELLOW
        else:
            color = Fore.GREEN

        print(f"{name:<15}{real:<15.2f}{pred:<15.2f}{color}{percent:<15.2f}{Style.RESET_ALL}")

    print("-" * 58)
    print(f"{'PERCENTUAL MÉDIO:':<15}{percent_total / len(saidas):.2f}%")
    print(f"{'DESVIO PADRÃO MÉDIO:':<15}{media_desvio_padrao:.6f}")
    print(f"{'TEST LOSS:':<15}{test_loss:.6f}")
    print("=" * 58)

    return media_desvio_padrao, test_loss

desvio_padrao_medio, test_loss = testar_sem_treinar(model, test_dataloader, lossfunc, saidas)
print(desvio_padrao_medio, test_loss)
