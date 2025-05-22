import torch
import numpy as np
from torch import optim
import pickle
from rna_global import RasmusNetwork as GlobalNetwork
from RNA_FLAG import RasmusNetwork as FlagNetwork

# Configuração do dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Rodando no dispositivo: {device}")

# ==========================
# 🧠 CARREGANDO OS MODELOS
# ==========================
# Modelo que sinaliza violação de restrições (FlagNetwork)
model_flag = FlagNetwork(10, 1)
state_dict_flag = torch.load('rna_flag_model_fbp_vaipf.pth', map_location=device)
model_flag.load_state_dict(state_dict_flag)
model_flag.eval()

# Modelo global que prediz os estados e variáveis algébricas
model_global = GlobalNetwork(10, 14)
state_dict = torch.load('rna_global_trained_cobeq.pth', map_location=device)
model_global.load_state_dict(state_dict)
model_global.eval()


# ==========================
# 🔄 NORMALIZAÇÃO DOS DADOS
# ==========================
class Normalizer:
    def __init__(self, u_min, u_max, mins_out, maxs_out):
        self.u_min = u_min
        self.u_max = u_max
        self.mins_out = mins_out
        self.maxs_out = maxs_out

    def normalize_inputs(self, x):
        return 2 * (x - self.u_min) / (self.u_max - self.u_min) - 1

    def denormalize_inputs(self, x_norm):
        return (x_norm + 1) * (self.u_max - self.u_min) / 2 + self.u_min

    def normalize_outputs(self, x):
        return 2 * (x - self.mins_out) / (self.maxs_out - self.mins_out) - 1

    def denormalize_outputs(self, x_norm):
        return (x_norm + 1) * (self.maxs_out - self.mins_out) / 2 + self.mins_out


# Limites de normalização para as entradas
u_min = torch.tensor([0.8e5, 0, 0, 0, 0, 35., 35., 35., 35., 35.], device=device)
u_max = torch.tensor([1.2e5, 1, 1, 1, 1, 65., 65., 65., 65., 65.], device=device)

# Limites de saída (22 elementos conforme o modelo global)
mins_out = torch.tensor([
    0.00010670462926302951, 0.0005843963639077903, 3.229948937698398e-05, 0.0005795289243178908, 12.441836086052101,
    -169.49041800700638, 55.55775607071595, 54.919168202220035, 54.70439009544229, 55.95839442254224, 20.580058322525446, 18.215705257048594,
    17.52693465550493, 22.104098174763582
], device=device)

maxs_out = torch.tensor([
    106.42121034906826, 108.02242855969821, 108.56097091664455, 108.56097091664455, 273.6148,
    105.04379583433938, 97.99995744472471, 97.99976693468392, 97.99998711851893, 100.37548991915249, 223.53709772291216, 223.29579330109422,
    223.3463702812387, 223.38355161614396
], device=device)

normalizer = Normalizer(u_min, u_max, mins_out, maxs_out)

# Lista de saídas para exibição (ordem conforme o modelo global)
saidas = ['q_main1', 'q_main2', 'q_main3', 'q_main4', 'q_tr', 'P_man',
          'P_fbhp1', 'P_fbhp2', 'P_fbhp3', 'P_fbhp4',
          'dP_bcs1', 'dP_bcs2', 'dP_bcs3', 'dP_bcs4']


# ==========================
# 🎯 FUNÇÃO OBJETIVO
# ==========================
def objective(u_tensor):
    # Predição dos estados e variáveis algébricas com o modelo global
    x_pred = model_global(u_tensor)
    # Desnormaliza para as unidades físicas
    x_pred_real = normalizer.denormalize_outputs(x_pred)

    # Mapeamento dos índices:
    # • x[1] (vazão de transferência)            -> x_pred_real[4]
    # • x[4] (vazão do poço 1)                     -> x_pred_real[0]
    # • x[7] (vazão do poço 2)                     -> x_pred_real[1]
    # • x[10] (vazão do poço 3)                    -> x_pred_real[2]
    # • x[13] (vazão do poço 4)                    -> x_pred_real[3]
    # • z[1]                                     -> x_pred_real[18]
    # • z[3]                                     -> x_pred_real[19]
    # • z[5]                                     -> x_pred_real[20]
    # • z[7]                                     -> x_pred_real[21]

    # Termos da função objetivo conforme optimize3_model.py
    sale_revenue = -3000 * x_pred_real[4]
    booster_energy = 9653.04 * (x_pred_real[4] / 3600) * (1.0963e3 * (u_tensor[9] / 50) ** 2) * 0.001
    esp_energy = ((x_pred_real[0] / 3600) * (x_pred_real[-4] * 1e2) +
                  (x_pred_real[1] / 3600) * (x_pred_real[-3] * 1e2) +
                  (x_pred_real[2] / 3600) * (x_pred_real[-2] * 1e2) +
                  (x_pred_real[3] / 3600) * (x_pred_real[-1] * 1e2))
    energy_cost = (booster_energy + esp_energy) * 0.91

    base_loss = sale_revenue + energy_cost

    # Penalidade para garantir que a FlagNetwork retorne um valor > 0,5
    flag = model_flag(u_tensor)
    penalty = 1e15 * torch.relu(0.5 - flag)

    return base_loss + penalty


# ==========================
# 🔍 EXECUÇÃO DA OTIMIZAÇÃO
# ==========================

def run_optimization():
    print("Iniciando otimização...")
    # Ponto inicial (vetor de 10 variáveis) e normalização
    u_init = normalizer.normalize_inputs(torch.tensor([0.8e5, 1, 1, 1, 1, 61, 61, 61,61, 53], device=device))
    u_tensor = torch.nn.Parameter(u_init.clone(), requires_grad=True)

    # Limites normalizados para as variáveis manipuláveis
    u_min_norm = normalizer.normalize_inputs(torch.tensor([0.8e5, 0, 0, 0, 0, 35, 35, 35, 35, 35], device=device))
    u_max_norm = normalizer.normalize_inputs(torch.tensor([1.2e5, 1, 1, 1, 1, 65, 65, 65, 65, 65], device=device))

    optimizer = optim.Adam([u_tensor], lr=0.001)

    for epoch in range(200):
        optimizer.zero_grad()
        loss = objective(u_tensor)

        if torch.isnan(loss):
            print(f"Erro: Loss NaN na iteração {epoch}")
            break

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # Garante que os parâmetros permaneçam dentro dos limites
            u_tensor.data = torch.clamp(u_tensor, u_min_norm, u_max_norm)
            flag_output = model_flag(u_tensor).item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.2f}, Flag = {flag_output:.4f}")

    print("\n===== Resultado da Otimização =====")
    u_opt_real = normalizer.denormalize_inputs(u_tensor.detach())
    print(f"f_BP: {u_opt_real[9]:.2f} Hz")
    print(f"p_topside: {u_opt_real[0] / 1e5:.2f} bar")
    for i in range(1, 5):
        print(f"f_ESP{i}: {u_opt_real[i + 4]:.2f} Hz | alpha{i}: {u_opt_real[i]:.2f}")

    x_pred_final = model_global(u_tensor).detach()
    x_pred_real = normalizer.denormalize_outputs(x_pred_final)

    print("\n===== Saídas Finais da Otimização =====")
    for i, name in enumerate(saidas):
        print(f"{name}: {x_pred_real[i].item():.2f}")

run_optimization()



