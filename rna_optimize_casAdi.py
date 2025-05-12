import casadi as ca
import numpy as np

# ==========================
# 1️⃣ Funções de Normalização e Desnormalização
# ==========================
def normalize(value, min_val, max_val):
    return 2 * (value - min_val) / (max_val - min_val) - 1

def denormalize(value, min_val, max_val):
    return (value + 1) * (max_val - min_val) / 2 + min_val

# Limites de normalização (devem ser os mesmos usados nas RNAs treinadas)
feature_min = np.array([0.8e5, 0, 0, 0, 0, 35, 35, 35, 35, 35])
feature_max = np.array([1.2e5, 1, 1, 1, 1, 65, 65, 65, 65, 65])

output_min = np.array([
    0.00010670462926302951, 0.0005843963639077903, 3.229948937698398e-05, 0.0005795289243178908, 12.441836086052101,
    -169.49041800700638, 55.55775607071595, 54.919168202220035, 54.70439009544229, 55.95839442254224, 20.580058322525446, 18.215705257048594,
    17.52693465550493, 22.104098174763582
])  # Definir os valores mínimos das saídas da RNA_GLOBAL
output_max = np.array([
    106.42121034906826, 108.02242855969821, 108.56097091664455, 108.56097091664455, 273.6148,
    105.04379583433938, 97.99995744472471, 97.99976693468392, 97.99998711851893, 100.37548991915249, 223.53709772291216, 223.29579330109422,
    223.3463702812387, 223.38355161614396
])  # Definir os valores máximos das saídas da RNA_GLOBAL

# ==========================
# 2️⃣ Carregar os pesos das redes neurais treinadas
# ==========================
def load_model_weights(file_path, layer_sizes):
    param_vector = np.load(file_path)
    params = {}
    index = 0
    for i in range(len(layer_sizes) - 1):
        input_size = layer_sizes[i]
        output_size = layer_sizes[i + 1]

        weight_size = input_size * output_size
        bias_size = output_size

        W = param_vector[index: index + weight_size].reshape((output_size, input_size))
        index += weight_size
        b = param_vector[index: index + bias_size].reshape((output_size, 1))
        index += bias_size

        params[f"W{i}"] = W
        params[f"b{i}"] = b

    return params

layer_sizes_flag = [10, 150, 150, 1]  # RNA_FLAG
layer_sizes_global = [10, 32, 32, 14]  # RNA_GLOBAL

params_flag = load_model_weights("modelo_flag.npy", layer_sizes_flag)
params_global = load_model_weights("modelo_global_cobeq_close.npy", layer_sizes_global)

# ==========================
# 3️⃣ Criar Redes no CasADi
# ==========================
def create_nn_function(params, layer_sizes, activation="tanh", output_activation=None):
    x = ca.MX.sym("x", layer_sizes[0])
    z = x
    for i in range(len(layer_sizes) - 1):
        W = params[f"W{i}"]
        b = params[f"b{i}"]
        z = ca.mtimes(W, z) + b
        if i < len(layer_sizes) - 2:
            if activation == "tanh":
                z = ca.tanh(z)
    if output_activation == "sigmoid":
        z = 1 / (1 + ca.exp(-z))
    return ca.Function("nn", [x], [z])

nn_flag = create_nn_function(params_flag, layer_sizes_flag, activation="tanh", output_activation="sigmoid")
nn_global = create_nn_function(params_global, layer_sizes_global, activation="tanh")

# ==========================
# 4️⃣ Definir Problema de Otimização no CasADi
# ==========================
opti = ca.Opti()

# Variáveis de entrada (não normalizadas)
u = opti.variable(10)

# Normalizar entradas antes de passar para a rede
u_norm = (2 * (u - feature_min) / (feature_max - feature_min)) - 1

# Saídas das redes neurais
output_flag = nn_flag(u_norm)
output_global = nn_global(u_norm)

# Desnormalizar saídas da RNA_GLOBAL
output_global_denorm = (output_global + 1) * (output_max - output_min) / 2 + output_min

penalty = 1e3 * ca.fmax(0.5 - output_flag, 0)  # Penaliza quando output_flag < 0.5

# Função objetivo (modificada para usar valores desnormalizados)
objective = -(3000 * output_global_denorm[4]) + ((9653.04 * (output_global_denorm[4]/3600) * (1.0963e3 * (u[9]/ 50) ** 2) * 0.001) + \
        ((output_global_denorm[0]/3600) * output_global_denorm[-4] * 1e2) + \
        ((output_global_denorm[1]/3600) * output_global_denorm[-3] * 1e2) + \
        ((output_global_denorm[2]/3600) * output_global_denorm[-2] * 1e2)  + \
        ((output_global_denorm[3]/3600) * output_global_denorm[-1] * 1e2)) * 0.91 + penalty

# Restrições
opti.subject_to(output_flag >= 0.5)

opti.minimize(objective)

# Definir limites das variáveis de controle
lbx = [0.8e5, 0, 0, 0, 0, 35, 35, 35, 35, 35]
ubx = [0.8e5, 1, 1, 1, 1, 65, 65, 65, 65, 65]
opti.subject_to(opti.bounded(lbx, u, ubx))

opti.solver("ipopt")

# Definir valores iniciais normalizados
u_initial = np.array([0.8e5, 1, 1, 1, 1, 65, 65, 65,65, 53])
opti.set_initial(u, u_initial)

# Resolver otimização
sol = opti.solve()

# Obter resultados
optimal_u = sol.value(u)
optimal_output_flag = sol.value(output_flag)
optimal_output_global = sol.value(output_global_denorm)

print("Entradas Otimizadas:", optimal_u)
print("Saída FLAG:", optimal_output_flag)
print("Saída GLOBAL:", optimal_output_global)


