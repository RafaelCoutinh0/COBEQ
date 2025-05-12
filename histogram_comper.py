import numpy as np
import matplotlib.pyplot as plt

# Importar os objetos do modelo físico
from optimize3_model import solver_padrao, x0_padrao, lbx as lbx_padrao, ubx as ubx_padrao, lbg as lbg_padrao, ubg as ubg_padrao

# Importar os objetos do modelo com RNA
from Rede_Neural_restrições_padrão import solver_rna, x0_rna, lbx as lbx_rna, ubx as ubx_rna, lbg as lbg_rna, ubg as ubg_rna

def benchmark_solver(solver, x0, lbx, ubx, lbg, ubg, n=10):
    tempos = []
    iteracoes = []
    for i in range(n):
        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        stats = solver.stats()
        tempos.append(stats['t_proc_total'])
        iteracoes.append(stats.get('iter_count', None))  # Usa .get() para evitar erro caso não exista

    return tempos, iteracoes


# === Roda benchmark ===
print("Rodando benchmark do modelo físico...")
tempos_padrao, itter_padrao = benchmark_solver(solver_padrao, x0_padrao, lbx_padrao, ubx_padrao, lbg_padrao, ubg_padrao)

print("Rodando benchmark do modelo com RNA...")
tempos_rna, itter_rna = benchmark_solver(solver_rna, x0_rna, lbx_rna, ubx_rna, lbg_rna, ubg_rna)

# === Estatísticas ===
media_padrao = np.mean(tempos_padrao)
desvio_padrao = np.std(tempos_padrao)
media_rna = np.mean(tempos_rna)
desvio_rna = np.std(tempos_rna)

print("\n=== Estatísticas de Tempo (IPOPT) ===")
print(f"Modelo Físico -> Média: {media_padrao:.4f}s | Desvio: {desvio_padrao:.4f}s")
print(f"Modelo com RNA -> Média: {media_rna:.4f}s | Desvio: {desvio_rna:.4f}s")

# === Plot ===
plt.figure(figsize=(10,6))
plt.hist(tempos_padrao, bins=20, alpha=0.7, label="Modelo Físico", color='blue')
plt.hist(tempos_rna, bins=20, alpha=0.7, label="Modelo com RNA", color='orange')
plt.xlabel("Tempo do Solver IPOPT (s)")
plt.ylabel("Frequência")
plt.title("Comparação de Tempo entre Modelo Físico e RNA (1000 execuções)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot ===
plt.figure(figsize=(10,6))
plt.hist(itter_padrao, bins=20, alpha=0.7, label="Modelo Físico", color='blue')
plt.hist(itter_rna, bins=20, alpha=0.7, label="Modelo com RNA", color='orange')
plt.xlabel("Tempo do Solver IPOPT (s)")
plt.ylabel("Frequência")
plt.title("Comparação de Tempo entre Modelo Físico e RNA (1000 execuções)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



fo_padrao =  [
    -7.9092301e+05,
    -7.8972283e+05,
    -7.4486945e+05,
    -7.4092337e+05,
    -7.3958554e+05,
    -7.3753178e+05,
    -7.4099991e+05,
    -7.4099411e+05,
    -7.4137324e+05,
    -7.4154202e+05,
    -7.4156988e+05,
    -7.4158242e+05,
    -7.4158538e+05,
    -7.4158538e+05,
    -7.4158543e+05,
    -7.4158553e+05,
    -7.4158553e+05
]

fo_rna = [
    -7.8387978e+05,
    -7.8389159e+05,
    -7.1613273e+05,
    -7.1614026e+05,
    -7.1615023e+05,
    -7.1615078e+05,
    -7.1615084e+05,
    -7.1615410e+05,
    -7.1615461e+05,
    -7.1615461e+05,
    -7.1615464e+05,
    -7.1615467e+05,
    -7.1615467e+05,
    -7.1615467e+05,
    -7.1615467e+05,
    -7.1615467e+05
]

plt.figure(figsize=(10,6))
plt.plot(fo_padrao, label="Modelo Físico", linewidth=2)
plt.plot(fo_rna, label="Modelo com RNA", linewidth=2)
plt.xlabel("Iteração IPOPT")
plt.ylabel("Função Objetivo")
plt.title("Evolução da Função Objetivo durante a Otimização")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
