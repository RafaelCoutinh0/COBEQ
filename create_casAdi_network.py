import torch
import numpy as np
from RNA_FLAG import RasmusNetwork as FlagNetwork  # Certifique-se de importar corretamente
from rna_global import RasmusNetwork as GlobalNetwork  # Certifique-se de importar corretamente

def save_model_as_vector(model, path):
    params = []
    for param in model.state_dict().values():
        params.append(param.cpu().numpy().flatten())  # Converter para NumPy e achatar
    param_vector = np.concatenate(params)  # Concatenar em um único vetor
    np.save(path, param_vector)  # Salvar

# Criar modelo FLAG com a arquitetura correta
model_flag = FlagNetwork(input_dim=10, output_dim=1)  # Ajuste os tamanhos conforme necessário
model_flag.load_state_dict(torch.load("rna_flag_model_fbp_vaipf.pth", map_location=torch.device('cpu')))
model_flag.eval()  # Colocar em modo de inferência

# Salvar pesos do modelo FLAG
save_model_as_vector(model_flag, "modelo_flag.npy")

# Criar modelo GLOBAL com a arquitetura correta
model_global = GlobalNetwork(input_dim=10, output_dim=14)  # Ajuste os tamanhos conforme necessário
model_global.load_state_dict(torch.load("rna_global_trained_cobeq.pth", map_location=torch.device('cpu')))
model_global.eval()

# Salvar pesos do modelo GLOBAL
save_model_as_vector(model_global, "modelo_global_cobeq.npy")

print("Pesos das redes convertidos com sucesso para formato NumPy!")
