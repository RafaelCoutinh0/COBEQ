import time
import importlib.util
import numpy as np
import os
import sys
import contextlib
import io
from optimize3_model import mani, fsolve, casadi_to_list
from Rede_Neural_restrições_padrão import nn_global, normalize, denormalize, feature_min, feature_max, output_min, output_max
import pandas as pd
import os
import sys
import contextlib

@contextlib.contextmanager
def suppress_all_output():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(devnull)

def run_module(path, var_name):
    spec = importlib.util.spec_from_file_location("modulo_temp", path)
    modulo = importlib.util.module_from_spec(spec)
    with suppress_all_output():
        spec.loader.exec_module(modulo)
    resultado = getattr(modulo, var_name)
    return resultado



def extrair_escalar(valor):
    try:
        return float(np.array(valor).flatten()[0])
    except Exception as e:
        print(f"Erro ao extrair escalar de {valor}: {e}")
        raise


os.system('cls' if os.name == 'nt' else 'clear')

opt_padrao = run_module("optimize3_model.py", "optimize_padrao")
opt_rna = run_module("Rede_Neural_restrições_padrão.py", "optimize_rna")

campos = [
    "Função Objetivo", "Receita (Venda)", "Energia Total",
    "Energia Booster", "Energia BCS1", "Energia BCS2", "Energia BCS3", "Energia BCS4"
]

print("\n========== COMPARAÇÃO DE RESULTADOS ==========")
for i in range(8):
    nome = campos[i]
    pad = extrair_escalar(opt_padrao[2 + i])
    rna = extrair_escalar(opt_rna[1 + i])
    print(f"{nome:<20}: Padrão = {pad:.2f} | RNA = {rna:.2f}")

variaveis = [
    "q_tr", "q_main1", "q_main2", "q_main3", "q_main4",
    "dP_bcs1", "dP_bcs2", "dP_bcs3", "dP_bcs4",
    "P_fbhp1", "P_fbhp2", "P_fbhp3", "P_fbhp4",
    "p_man"
]

print("\n========== COMPARAÇÃO DAS VARIÁVEIS RELEVANTES ==========")
for i, nome in enumerate(variaveis):
    pad = extrair_escalar(opt_padrao[0][i])
    rna = extrair_escalar(opt_rna[0][i])
    print(f"{nome:<10}: Padrão = {pad:.2f} | RNA = {rna:.2f}")

control_names = [
    "f_BP (Hz)", "p_topside (bar)",
    "f_BCS_1 (Hz)", "alpha_1 (-)",
    "f_BCS_2 (Hz)", "alpha_2 (-)",
    "f_BCS_3 (Hz)", "alpha_3 (-)",
    "f_BCS_4 (Hz)", "alpha_4 (-)"
]

ur_indices_reorder = [9, 0, 5, 1, 6, 2, 7, 3, 8, 4]

print("\n========== VARIÁVEIS MANIPULADAS (u) ==========")
unidades = {
    "p_topside": "bar",
    "alpha_1": "-", "alpha_2": "-", "alpha_3": "-", "alpha_4": "-",
    "f_BP": "Hz", "f_BCS_1": "Hz", "f_BCS_2": "Hz", "f_BCS_3": "Hz", "f_BCS_4": "Hz"
}

for i, nome in enumerate(control_names):
    up = extrair_escalar(opt_padrao[1][i])
    ur = extrair_escalar(opt_rna[0][14 + ur_indices_reorder[i]])
    if "p_topside" in nome:
        up /= 1e5
        ur /= 1e5
        unidade = unidades["p_topside"]
    else:
        unidade = unidades[nome.split()[0]]
    print(f"{nome:<17}: Padrão = {up:.2f} | RNA = {ur:.2f}")

# COMPARAÇÃO DE TEMPOS (extraídos do vetor otimizado)
tempo_padrao = extrair_escalar(opt_padrao[10])
tempo_rna = extrair_escalar(opt_rna[9])
print("\n========== COMPARAÇÃO DE TEMPOS (IPOPT) ==========")
print(f"Tempo solver modelo padrão : {tempo_padrao:.4f} s")
print(f"Tempo solver modelo com RNA: {tempo_rna:.4f} s")

# ========== TESTE CRUZADO ==========
print("\n========== TESTE CRUZADO ==========")

# Entradas da RNA no modelo matemático
u_rna_raw = np.array(opt_rna[0][14:])
ur_indices = [5, 0, 6, 1, 7, 2, 8, 3, 9, 4]  # ordem para modelo
u_rna_for_model = [u_rna_raw[i] for i in ur_indices]
mani_solver_rna = lambda y: np.array([float(i) for i in mani.model(0, y[:14], y[14:], u_rna_for_model)])
x0 = np.copy(opt_padrao[0][:14])
z0 = [30.03625, 209.91713] * 4
start_ref = time.time()
y_ss_rna = fsolve(mani_solver_rna, np.concatenate((x0, z0)))
end_ref = time.time()
time_ref = end_ref - start_ref
print(f"tempo modelo referencia: {time_ref}")

x_ss_matematico_rna = y_ss_rna[:14]
z_ss_matematico_rna = y_ss_rna[14:]

q_tr = x_ss_matematico_rna[1]
energybooster = 9653.04 * (q_tr / 3600) * (1.0963e3 * (u_rna_for_model[0]/50)**2) * 0.001
energybcs = sum((x_ss_matematico_rna[i]/3600) * z_ss_matematico_rna[j] * 1e2 for i, j in zip([4,7,10,13], [1,3,5,7]))
energytot_modelo_rna = (energybooster + energybcs) * 0.91
venda_modelo_rna = 3000 * q_tr
fo_modelo_rna = -(venda_modelo_rna - energytot_modelo_rna)

print(">> Modelo Matemático com entradas da RNA:")
print(f"  FO = {fo_modelo_rna:.2f} | Venda = {venda_modelo_rna:.2f} | Energia Total = {energytot_modelo_rna:.2f}")

# Entradas do modelo matemático na RNA
u_padrao_raw = np.array(opt_padrao[1])
up_indices = [1,3,5,7,9,0,2,4,6,8]
u_padrao_for_rna = [u_padrao_raw[i] for i in up_indices]
start_rna = time.perf_counter()
u_padrao_norm = normalize(u_padrao_for_rna, feature_min, feature_max)
x_ss_rna_padrao = np.array(denormalize(nn_global(u_padrao_norm), output_min, output_max)).flatten()
end_rna =  time.perf_counter()

time_rna = end_rna - start_rna
print(f"tempo modelo rna: {time_rna}")

q_tr = x_ss_rna_padrao[4]
energybooster = 9653.04 * (q_tr/3600) * (1.0963e3 * (u_padrao_for_rna[9]/50)**2) * 0.001
energybcs = sum((x_ss_rna_padrao[i]/3600) * x_ss_rna_padrao[-(4 - i)] * 1e2 for i in range(4))
energytot_rna_padrao = (energybooster + energybcs) * 0.91
venda_rna_padrao = 3000 * q_tr
fo_rna_padrao = -(venda_rna_padrao - energytot_rna_padrao)

print(">> RNA com entradas do Modelo Matemático:")
print(f"  FO = {fo_rna_padrao:.2f} | Venda = {venda_rna_padrao:.2f} | Energia Total = {energytot_rna_padrao:.2f}")

import matplotlib.pyplot as plt

def benchmark_otimizacoes_solver(n=100):
    tempos_padrao = []
    tempos_rna = []

    for i in range(n):
        print(f"Execução {i+1}/{n}", end="\r")

        opt_padrao = run_module("optimize3_model.py", "optimize_padrao")
        opt_rna = run_module("Rede_Neural_restrições_padrão.py", "optimize_rna")

        tempo_padrao = extrair_escalar(opt_padrao[10])
        tempo_rna = extrair_escalar(opt_rna[9])

        tempos_padrao.append(tempo_padrao)
        tempos_rna.append(tempo_rna)

    # Cálculos estatísticos
    media_padrao = np.mean(tempos_padrao)
    desvio_padrao = np.std(tempos_padrao)

    media_rna = np.mean(tempos_rna)
    desvio_rna = np.std(tempos_rna)

    print("\n========== ESTATÍSTICAS DE TEMPO (IPOPT) ==========")
    print(f"Modelo Padrão -> Média: {media_padrao:.4f} s | Desvio padrão: {desvio_padrao:.4f} s")
    print(f"Modelo com RNA -> Média: {media_rna:.4f} s | Desvio padrão: {desvio_rna:.4f} s")

    # Plot do histograma
    plt.figure(figsize=(10, 6))
    plt.hist(tempos_padrao, bins=10, alpha = 0.7, label="Modelo Padrão (IPOPT)")
    plt.hist(tempos_rna, bins=10, alpha = 0.7 ,label="Modelo com RNA (IPOPT)")
    plt.xlabel("Tempo do Solver IPOPT (s)")
    plt.ylabel("Frequência")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Executa o benchmark
benchmark_otimizacoes_solver(n=1000)
#
# # Dados atualizados e convertidos para valores positivos
# referencial_obj = [abs(x) for x in [
#     -7.8387978e+05, -7.8389159e+05, -7.1613273e+05, -7.1614026e+05,
#     -7.1615023e+05, -7.1615078e+05, -7.1615084e+05, -7.1615410e+05,
#     -7.1615461e+05, -7.1615461e+05, -7.1615464e+05, -7.1615467e+05,
#     -7.1615467e+05, -7.1615467e+05, -7.1615467e+05, -7.1615467e+05,
#     -7.1615467e+05
# ]]
#
# rna_obj = [abs(x) for x in [
#     -7.9092301e+05, -7.8976206e+05, -7.8368962e+05, -7.8210626e+05,
#     -7.7960749e+05, -7.7827726e+05, -7.7771928e+05, -7.7756095e+05,
#     -7.7747201e+05, -7.7902436e+05, -7.3397903e+05, -7.3727416e+05,
#     -7.4029366e+05, -7.4137888e+05, -7.4386967e+05, -7.4135743e+05,
#     -7.4154219e+05, -7.4156943e+05, -7.4158222e+05, -7.4158538e+05,
#     -7.4158538e+05, -7.4158543e+05, -7.4158553e+05, -7.4158553e+05
# ]]
#
# # Plotando a convergência
# plt.figure(figsize=(10, 6))
# plt.plot(referencial_obj, 'o-', label='Referencial (Modelo original)')
# plt.plot(rna_obj, 's--', label='RNA (Modelo substituto)')
# plt.xlabel('Iteração')
# plt.ylabel('Função Objetivo')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# dados = {
#     "Parâmetro": [
#         "Iterações totais",
#         "Objetivo final (não escalado)",
#         "Avaliações da função objetivo",
#         "Avaliações do gradiente",
#         "Avaliações da Hessiana",
#         "Avaliações das restrições",
#         "Avaliações do Jacobiano",
#         "Número de variáveis",
#         "Desigualdades",
#         "Igualdades",
#         "Tempo total (ms)",
#         "Precisão (NLP error)",
#         "Tempo nlp_f médio (μs)",
#         "Tempo nlp_g médio (μs)",
#         "Tempo grad_f médio (μs)",
#         "Tempo hess_l médio (μs)",
#         "Tempo jac_g médio (μs)",
#         "Tempo médio por iteração"
#     ],
#     "Referencial (Original)": [
#         16,
#         round(7.1615467e+05, 2),
#         23,
#         17,
#         16,
#         23,
#         17,
#         31,
#         12,
#         22,
#         round(12.24, 2),
#         "{:.2e}".format(7.51e-08),
#         round(1.65, 2),
#         round(20.65, 2),
#         round(2.89, 2),
#         round(69.87, 2),
#         round(53.50, 2),
#         round(12.24/16,2)
#
#     ],
#     "RNA (Substituto)": [
#         23,
#         round(7.4158553e+05, 2),
#         81,
#         24,
#         23,
#         81,
#         24,
#         10,
#         12,
#         0,
#         round(17.15, 2),
#         "{:.2e}".format(2.29e-05),
#         round(4.02, 2),
#         round(5.38, 2),
#         round(11.24, 2),
#         round(74.17, 2),
#         round(31.40, 2),
#         round(17.15/23, 2)
#     ]
# }
#
# df = pd.DataFrame(dados)
#
# # Melhor visualização com alinhamento
# print("\n📋 Comparação de Desempenho e Tempos:\n")
# print(df.to_string(index=False))

