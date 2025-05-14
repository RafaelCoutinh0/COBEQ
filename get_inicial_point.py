import pickle
from violation_verified import violantion
def load_data_from_pkl(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
feature_vars = [
        'p_topo', 'valve1', 'valve2', 'valve3', 'valve4',
        'bcs1_freq', 'bcs2_freq', 'bcs3_freq', 'bcs4_freq',
        'booster_freq',
    ]
file_path = 'data_rna_cobeq_100k.pkl'
library_data = load_data_from_pkl(file_path)
print(library_data['flag'])
entrada = []
for i in range(len(library_data['flag'])):
    if library_data['flag'][i] == 1:
        print(library_data['flag'][i])
        for j in feature_vars:
            print(f"{j}:{library_data[j][i]}")
            entrada.append(library_data[j][i])
        teste = [entrada[9], 8e4,entrada[5],entrada[1],entrada[6],entrada[2],entrada[7],entrada[3],entrada[8],entrada[4]]
        vio = violantion(teste)
        if vio == 1:
            print(vio)
            break



novo_conteudo = f"""import numpy as np

p_topo = {8e4}
f_bcs_1 = {entrada[5]}
f_bcs_2 = {entrada[6]}
f_bcs_3 = {entrada[7]}
f_bcs_4 = {entrada[8]}
f_booster = {entrada[9]}
valve_1 = {entrada[1]}
valve_2 = {entrada[2]}
valve_3 = {entrada[3]}
valve_4 = {entrada[4]}

u0_ref = np.array([
    f_booster,    # booster
    p_topo,       # pressão no topo
    f_bcs_1,      # BCS 1
    valve_1,      # válvula 1
    f_bcs_2,      # BCS 2
    valve_2,      # válvula 2
    f_bcs_3,      # BCS 3
    valve_3,      # válvula 3
    f_bcs_4,      # BCS 4
    valve_4       # válvula 4
])

u0_rna = np.array([
    p_topo,       # pressão no topo
    valve_1,      # válvula 1
    valve_2,      # válvula 2
    valve_3,      # válvula 3
    valve_4,      # válvula 4
    f_bcs_1,      # BCS 1
    f_bcs_2,      # BCS 2
    f_bcs_3,      # BCS 3
    f_bcs_4,      # BCS 4
    f_booster     # booster
])
"""

# Sobrescreve o arquivo
with open("pontos_iniciais.py", "w") as f:
    f.write(novo_conteudo)

print("Arquivo pontos_iniciais.py atualizado com sucesso!")