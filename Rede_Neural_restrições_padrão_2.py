import casadi as ca
import numpy as np
from casadi import vertcat
from pontos_iniciais import u0_rna


def normalize(value, min_val, max_val):
    return 2 * (value - min_val) / (max_val - min_val) - 1

def denormalize(value, min_val, max_val):
    return (value + 1) * (max_val - min_val) / 2 + min_val

# Limites de normalização (mesmos usados no treinamento da RNA)
u0_min = np.array([0.8e5, 0, 0, 0, 0, 35, 35, 35, 35, 35])
u0_max = np.array([0.8000000111e5, 1, 1, 1, 1, 65, 65, 65, 65, 65])
feature_min = np.zeros(14)
output_max = np.array([
    106.42121034906826,      # q_main1
    108.02242855969821,      # q_main2
    108.56097091664455,      # q_main3
    108.56097091664455,      # q_main4
    260.6148,                # q_tr
    105.04379583433938,      # P_man
    97.99995744472471,       # P_fbhp1
    97.99976693468392,       # P_fbhp2
    97.99998711851893,       # P_fbhp3
    100.37548991915249,      # P_fbhp4
    223.53709772291216,      # dP_bcs1
    223.29579330109422,      # dP_bcs2
    223.3463702812387,       # dP_bcs3
    223.38355161614396       # dP_bcs4
])

label_min = np.array([0.8e5, 0, 0, 0, 0, 35, 35, 35, 35, 35])
label_max = np.array([0.8000000111e5, 1, 1, 1, 1, 65, 65, 65, 65, 65])

# Valores mínimos e máximos das saídas da RNA_GLOBAL (22 dimensões)
output_min = np.array([
    0.00010670462926302951, 0.0005843963639077903, 3.229948937698398e-05, 0.0005795289243178908, 12.441836086052101,
    -169.49041800700638, 55.55775607071595, 54.919168202220035, 54.70439009544229, 55.95839442254224, 20.580058322525446, 18.215705257048594,
    17.52693465550493, 22.104098174763582])  # Definir os valores mínimos das saídas da RNA_GLOBAL


def load_model_weights(file_path, layer_sizes):
    param_vector = np.load(file_path)
    params = {}
    index = 0
    for i in range(len(layer_sizes) - 1):
        input_size = layer_sizes[i]
        output_size = layer_sizes[i + 1]
        W = param_vector[index: index + input_size * output_size].reshape((output_size, input_size))
        index += input_size * output_size
        b = param_vector[index: index + output_size].reshape((output_size, 1))
        index += output_size
        params[f"W{i}"] = W
        params[f"b{i}"] = b
    return params

layer_sizes_global = [10, 32, 32, 14]  # RNA_GLOBAL
params_global = load_model_weights("modelo_global_cobeq_close.npy", layer_sizes_global)

def create_nn_function(params, layer_sizes):
    x = ca.MX.sym("x", layer_sizes[0])
    z = x
    for i in range(len(layer_sizes) - 1):
        W = params[f"W{i}"]
        b = params[f"b{i}"]
        z = ca.mtimes(W, z) + b
        if i < len(layer_sizes) - 2:
            z = ca.tanh(z)
    return ca.Function("nn", [x], [z])

nn_global = create_nn_function(params_global, layer_sizes_global)

# Definição da variável de decisão (u - 10 entradas)
u = ca.MX.sym('u', 10)
x = ca.MX.sym('x', 14)

# Normalização das entradas e computo da saída da RNA
u_norm = normalize(u, label_min, label_max)
output_global_denorm = x

objective = -(3000 * output_global_denorm[4]) + (
    (9653.04 * (output_global_denorm[4]/3600) * (1.0963e3 * (u[9]/50) ** 2) * 0.001) +
    ((output_global_denorm[0]/3600) * output_global_denorm[-4] * 1e2) +
    ((output_global_denorm[1]/3600) * output_global_denorm[-3] * 1e2) +
    ((output_global_denorm[2]/3600) * output_global_denorm[-2] * 1e2) +
    ((output_global_denorm[3]/3600) * output_global_denorm[-1] * 1e2)
) * 0.91

g_equality = (normalize(x, output_min,output_max) - nn_global(u_norm))
# Restrições (iguais às de optimize3_model.py, utilizando os elementos da saída da RNA)
restqmain1 = ca.vertcat(
    output_global_denorm[0] - ((output_global_denorm[-4] + 334.2554) / 19.0913),
    ((output_global_denorm[-4] + 193.8028) / 4.4338) - output_global_denorm[0]
)
restqmain2 = ca.vertcat(
    output_global_denorm[1] - ((output_global_denorm[-3] + 334.2554) / 19.0913),
    ((output_global_denorm[-3] + 193.8028) / 4.4338) - output_global_denorm[1]
)
restqmain3 = ca.vertcat(
    output_global_denorm[2] - ((output_global_denorm[-2] + 334.2554) / 19.0913),
    ((output_global_denorm[-2] + 193.8028) / 4.4338) - output_global_denorm[2]
)
restqmain4 = ca.vertcat(
    output_global_denorm[3] - ((output_global_denorm[-1] + 334.2554) / 19.0913),
    ((output_global_denorm[-1] + 193.8028) / 4.4338) - output_global_denorm[3]
)

restfbprest1 = output_global_denorm[6] - 74.14
restfbprest2 = output_global_denorm[7] - 74.14
restfbprest3 = output_global_denorm[8] - 74.14
restfbprest4 = output_global_denorm[9] - 74.14
g_inequality = vertcat(restqmain1,restqmain2,restqmain3,restqmain4,restfbprest1,restfbprest2,restfbprest3,restfbprest4)

# Concatenando todas as restrições
g_constraints = ca.vertcat(g_equality,g_inequality)
num_eq = g_equality.shape[0]  # Número de igualdades
num_ineq = g_inequality.shape[0]  # Número de desigualdades
# Montagem do problema de otimização
nlp = {'x': vertcat( x, u), 'f': objective, 'g': g_constraints}

solver = ca.nlpsol('solver', 'ipopt', nlp)

# Valores iniciais e limites para u
u0 = u0_rna
u0_norm = normalize(u0, label_min, label_max)

# Calcula a saída da RNA com u0 normalizado
x0_casadi = nn_global(u0_norm)  # Saída ainda normalizada
x0 = denormalize(x0_casadi, output_min, output_max)  # Desnormaliza
x0 = np.array(x0).flatten()  # Converte CasADi -> NumPy

# Concatena para formar o vetor inicial do solver
x0_total = np.concatenate([x0, u0])
u_min = np.array([0.8e5, 0, 0, 0, 0, 35, 35, 35, 35, 35])
u_max = np.array([0.8000000111e5, 1, 1, 1, 1, 65, 65, 65, 65, 65])

# output_min e output_max são os limites corretos de x
lbx = feature_min.tolist() + u_min.tolist()
ubx = output_max.tolist() + u_max.tolist()

# Como cada restrição deve ser >= 0, definimos:
lbg = [0.0] * num_eq + [0.0] * num_ineq  # Igualdades: 0 ≤ g ≤ 0 | Desigualdades: 0 ≤ g ≤ ∞
ubg = [0.0] * num_eq + [np.inf] * num_ineq
# Resolver o problema
sol = solver(x0=x0_total, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
solver_stats = solver.stats()
tempo_ipopt = solver_stats['t_proc_total']

solution = sol['x']
# Corrigir nome da variável se necessário

# Converte para array e achata
optimal_u_np = np.array(solution).flatten()

# Pega os últimos 10 valores (u ótimos)
optimal_u_only = optimal_u_np[-10:]
optimal_output_global = optimal_u_np[:-10]

# Separa x e u da solução ótima
x_opt = optimal_u_np[:14]
optimal_u = optimal_u_np[14:]

# Define função com entrada [x, u]
g_func = ca.Function('g_func', [x, u], [g_constraints])

# Avalia com x e u separados
g_values = g_func(x_opt, optimal_u)

# Converte para NumPy para facilitar a análise
g_values_np = np.array(g_values).flatten()

# Verifica quais restrições foram violadas
violadas = np.where(g_values_np < 0)[0]

# if len(violadas) > 0:
#     print("Restrições violadas:")
#     for idx in violadas:
#         print(f"Restrição {idx + 1} violada: valor = {g_values_np[idx]:.6f}")
# else:
#     print("Nenhuma restrição foi violada.")


# Lista de saídas para exibição (ordem conforme o modelo global)
saidas = ['q_main1', 'q_main2', 'q_main3', 'q_main4', 'q_tr', 'P_man',
          'P_fbhp1', 'P_fbhp2', 'P_fbhp3', 'P_fbhp4',
          'dP_bcs1', 'dP_bcs2', 'dP_bcs3', 'dP_bcs4']

# print("===== Resultado da Otimização =====")
# print(f"f_BP: {float(optimal_u[9]):.2f} Hz")  # Converte para float
# print(f"p_topside: {float(optimal_u[0]) / 1e5:.2f} bar")  # Converte antes da divisão

# for i in range(1, 5):
#     print(f"f_ESP{i}: {float(optimal_u[i + 4]):.2f} Hz | alpha{i}: {float(optimal_u[i]):.2f}")
#
# print("\n===== Saídas Finais da Otimização =====")
# for i, name in enumerate(saidas):
#     print(f"{name}: {float(optimal_output_global[i]):.2f}")  # Converte cada saída

# Converte o vetor CasADi para NumPy antes de usar nos cálculos
x_ss = np.array(optimal_output_global)

objective_rna = -(3000 * x_ss[4]) + \
        (((9653.04 * x_ss[4]/3600) * (1.0963e3 * (optimal_u[9]/ 50) ** 2) * 0.001) + \
         ((x_ss[0]/3600) * x_ss[-4] * 1e2) + \
         ((x_ss[1]/3600) * x_ss[-3] * 1e2) + \
         ((x_ss[2]/3600) * x_ss[-2] * 1e2)  + \
         ((x_ss[3]/3600) * x_ss[-1] * 1e2)) * 0.91


# print(f"valor da função objetivo: {objective_rna}")

energybooster = (9653.04 * (x_ss[4]/3600) * (1.0963e3 * (optimal_u[9]/50) ** 2) * 0.001)
energybcs1 =    ((x_ss[0]/3600) * x_ss[-4] * 1e2)
energybcs2 =    ((x_ss[1]/3600) * x_ss[-3] * 1e2)
energybcs3 =    ((x_ss[2]/3600) * x_ss[-2] * 1e2)
energybcs4 =    ((x_ss[3]/3600) * x_ss[-1] * 1e2)
energytot = (energybooster + energybcs1 + energybcs2 + energybcs3 + energybcs4) * 0.91
venda = 3000 * x_ss[4]
funcao = venda - energytot
BOLD = '\033[1m'
RESET = '\033[0m'
CYAN = '\033[36m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
# print(f"{GREEN}{BOLD}{'='*50}{RESET}\n{GREEN}{BOLD}VALORES DA FUNÇÃO OBJETIVO:{RESET}")
# print(f"{YELLOW}{BOLD}Valor da venda do petróleo{RESET}: R${float(venda.item()):.2f}")
# print(energytot, energybooster)
# print(f"{YELLOW}{BOLD}Preço da energia total{RESET}: R${energytot:.2f}")
# print(f"{GREEN}{BOLD}Energia do booster{RESET}: {energybooster:.2f} Kwh")
# print(f"{GREEN}{BOLD}Energia do BCS 1{RESET}: {energybcs1} Kwh")
# print(f"{GREEN}{BOLD}Energia do BCS 2{RESET}: {energybcs2} Kwh")
# print(f"{GREEN}{BOLD}Energia do BCS 3{RESET}: {energybcs3} Kwh")
# print(f"{GREEN}{BOLD}Energia do BCS 4{RESET}: {energybcs4} Kwh")

def casadi_to_list(mx_vector):
    return [float(x) for x in np.array(mx_vector).flatten()]

optimal_u= casadi_to_list(optimal_u)
x_ss_rna_relevante = [
    x_ss[4],                          # q_tr
    x_ss[0], x_ss[1], x_ss[2], x_ss[3],       # q_main1 a q_main4
    x_ss[10], x_ss[11], x_ss[12], x_ss[13],   # dP_bcs1 a dP_bcs4
    x_ss[6], x_ss[7], x_ss[8], x_ss[9],       # P_fbhp_1 a P_fbhp_4
    x_ss[5],                                  # p_man
    *[float(u) for u in np.array(optimal_u).flatten()]  # variáveis manipuladas
]
optimize_rna = [x_ss_rna_relevante, objective_rna, venda, energytot, energybooster, energybcs1, energybcs2, energybcs3, energybcs4, tempo_ipopt]
# print(tempo_ipopt)