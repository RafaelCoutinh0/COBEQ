import time
import importlib.util
import numpy as np
import os
import sys
import contextlib
import io
from optimize3_model import mani, fsolve, casadi_to_list
from Rede_Neural_restrições_padrão import nn_global, normalize, denormalize, feature_min, feature_max, output_min, output_max

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

opt_padrao = run_module("optimize3_model.py", "optimize_padrao")
opt_rna = run_module("Rede_Neural_restrições_padrão.py", "optimize_rna")
opt_rna2 = run_module("Rede_Neural_restrições_padrão_2.py", "optimize_rna")

campos = [
    "Função Objetivo", "Receita (Venda)", "Energia Total",
    "Energia Booster", "Energia BCS1", "Energia BCS2", "Energia BCS3", "Energia BCS4"
]

print("\n========== COMPARAÇÃO DE RESULTADOS ==========")
for i in range(8):
    nome = campos[i]
    pad = extrair_escalar(opt_padrao[2 + i])
    rna = extrair_escalar(opt_rna[1 + i])
    rna2 = extrair_escalar(opt_rna2[1 + i])
    print(f"{nome:<20}: Padrão = {pad:.2f} | RNA_1 = {rna:.2f} | RNA_2 = {rna2:.2f}")

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
    rna2 = extrair_escalar(opt_rna2[0][i])
    print(f"{nome:<10}: Padrão = {pad:.2f} | RNA_1 = {rna:.2f} | RNA_2 = {rna2:.2f}")

control_names = [
    "f_BP (Hz)", "p_topside (bar)",
    "f_BCS_1 (Hz)", "alpha_1 (-)",
    "f_BCS_2 (Hz)", "alpha_2 (-)",
    "f_BCS_3 (Hz)", "alpha_3 (-)",
    "f_BCS_4 (Hz)", "alpha_4 (-)"
]
ur_indices_reorder = [9, 0, 5, 1, 6, 2, 7, 3, 8, 4]
unidades = {
    "p_topside": "bar",
    "alpha_1": "-", "alpha_2": "-", "alpha_3": "-", "alpha_4": "-",
    "f_BP": "Hz", "f_BCS_1": "Hz", "f_BCS_2": "Hz", "f_BCS_3": "Hz", "f_BCS_4": "Hz"
}

print("\n========== VARIÁVEIS MANIPULADAS (u) ==========")
for i, nome in enumerate(control_names):
    up = extrair_escalar(opt_padrao[1][i])
    ur = extrair_escalar(opt_rna[0][14 + ur_indices_reorder[i]])
    ur2 = extrair_escalar(opt_rna2[0][14 + ur_indices_reorder[i]])
    if "p_topside" in nome:
        up /= 1e5
        ur /= 1e5
        ur2 /= 1e5
        unidade = unidades["p_topside"]
    else:
        unidade = unidades[nome.split()[0]]
    print(f"{nome:<17}: Padrão = {up:.2f} | RNA_1 = {ur:.2f} | RNA_2 = {ur2:.2f}")

tempo_padrao = extrair_escalar(opt_padrao[10])
tempo_rna = extrair_escalar(opt_rna[9])
tempo_rna2 = extrair_escalar(opt_rna2[9])
print("\n========== COMPARAÇÃO DE TEMPOS (IPOPT) ==========")
print(f"Tempo solver modelo padrão  : {tempo_padrao:.4f} s")
print(f"Tempo solver modelo RNA_1   : {tempo_rna:.4f} s")
print(f"Tempo solver modelo RNA_2   : {tempo_rna2:.4f} s")


def benchmark_otimizacoes_solver(n=100):
    tempos_padrao = []
    tempos_rna = []
    tempos_rna2 = []

    for i in range(n):
        print(f"Execução {i+1}/{n}", end="\r")

        opt_padrao = run_module("optimize3_model.py", "optimize_padrao")
        opt_rna = run_module("Rede_Neural_restrições_padrão.py", "optimize_rna")
        opt_rna2 = run_module("Rede_Neural_restrições_padrão_2.py", "optimize_rna")

        tempos_padrao.append(extrair_escalar(opt_padrao[10]))
        tempos_rna.append(extrair_escalar(opt_rna[9]))
        tempos_rna2.append(extrair_escalar(opt_rna2[9]))

    # Cálculos estatísticos
    media_padrao = np.mean(tempos_padrao)
    desvio_padrao = np.std(tempos_padrao)
    media_rna = np.mean(tempos_rna)
    desvio_rna = np.std(tempos_rna)
    media_rna2 = np.mean(tempos_rna2)
    desvio_rna2 = np.std(tempos_rna2)

    print("\n========== ESTATÍSTICAS DE TEMPO (IPOPT) ==========")
    print(f"Modelo Padrão  -> Média: {media_padrao:.4f} s | Desvio padrão: {desvio_padrao:.4f} s")
    print(f"Modelo RNA_1   -> Média: {media_rna:.4f} s | Desvio padrão: {desvio_rna:.4f} s")
    print(f"Modelo RNA_2   -> Média: {media_rna2:.4f} s | Desvio padrão: {desvio_rna2:.4f} s")

    # Plot do histograma
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(tempos_padrao, bins=10, alpha=0.7, label="Modelo Padrão (IPOPT)", edgecolor = 'black')
    plt.hist(tempos_rna, bins=10, alpha=0.7, label="Modelo RNA_1 (IPOPT)", edgecolor = 'black')
    plt.hist(tempos_rna2, bins=10, alpha=0.7, label="Modelo RNA_2 (IPOPT)", edgecolor = 'black')
    plt.xlabel("Tempo do Solver IPOPT (s)")
    plt.ylabel("Frequência")
    plt.title(f"Comparação de Tempo do Solver ({n} Execuções)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Executa o benchmark
benchmark_otimizacoes_solver(n=1000)