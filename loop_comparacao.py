import importlib.util
import time

import numpy as np
import matplotlib.pyplot as plt

def run_module(path, var_name):
    spec = importlib.util.spec_from_file_location("modulo_temp", path)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return getattr(modulo, var_name)

# Listas com valores para simular diferentes pontos
f_bcs_1_list = [65, 55, 45, 35, 60]
f_bcs_2_list = [65, 55, 45, 35, 60]
f_bcs_3_list = [65, 55, 45, 35, 60]
f_bcs_4_list = [65, 55, 45, 35, 60]
f_booster_list = [65, 55, 45, 35, 50]
valve_list = 1
p_topo = 0.8e5  # constante

n_sim = len(f_bcs_1_list)
n_reps = 10  # número de repetições por ponto

tempos_padrao_medio = []
tempos_rna_medio = []

for i in range(n_sim):
    tempos_padrao_reps = []
    tempos_rna_reps = []

    for rep in range(n_reps):
        f_bcs_1 = f_bcs_1_list[i]
        f_bcs_2 = f_bcs_2_list[i]
        f_bcs_3 = f_bcs_3_list[i]
        f_bcs_4 = f_bcs_4_list[i]
        f_booster = f_booster_list[i]
        valve = valve_list

        # Atualiza o arquivo pontos_iniciais.py
        conteudo = f"""
import numpy as np

p_topo = {p_topo}
f_bcs_1 = {f_bcs_1}
f_bcs_2 = {f_bcs_2}
f_bcs_3 = {f_bcs_3}
f_bcs_4 = {f_bcs_4}
f_booster = {f_booster}
valve = {valve}
u0_ref = np.array([{f_booster}, {p_topo}, {f_bcs_1}, {valve}, {f_bcs_2}, {valve}, {f_bcs_3}, {valve}, {f_bcs_4}, {valve}])
u0_rna = np.array([{p_topo}, {valve}, {valve}, {valve}, {valve}, {f_bcs_1}, {f_bcs_2}, {f_bcs_3}, {f_bcs_4}, {f_booster}])
"""
        with open("pontos_iniciais.py", "w") as f:
            f.write(conteudo)

        # Executa compare_otimização.py e acessa diretamente os tempos
        star_time = time.time()
        tempo_padrao = run_module("compare_otimização.py", "tempo_padrao")
        midle_time = time.time()
        tempo_rna = run_module("compare_otimização.py", "tempo_rna")
        end_time = time.time()
        tempo_padrao = midle_time - star_time
        tempo_rna = end_time - midle_time
        tempos_padrao_reps.append(tempo_padrao)
        tempos_rna_reps.append(tempo_rna)

    # Calcula média das 100 repetições
    tempos_padrao_medio.append(np.mean(tempos_padrao_reps))
    tempos_rna_medio.append(np.mean(tempos_rna_reps))

# ===== PLOT FINAL =====
plt.figure(figsize=(10, 6))
plt.plot(tempos_padrao_medio, 'o-', label='Modelo Padrão (média de 100)')
plt.plot(tempos_rna_medio, 's--', label='Modelo com RNA (média de 100)')
plt.xlabel("Iteração (configuração de entrada)")
plt.ylabel("Tempo de Otimização Médio (s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




