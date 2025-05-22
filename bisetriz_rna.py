import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
import pickle
from sklearn.metrics import r2_score

# --- Funções auxiliares ---

def load_data_from_pkl(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def normalize(value, min_val, max_val):
    return 2 * (value - min_val) / (max_val - min_val) - 1

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

# --- Configuração do modelo ---

layer_sizes_global = [10, 32, 32, 14]  # RNA_GLOBAL
params_global = load_model_weights("modelo_global_cobeq_close.npy", layer_sizes_global)
nn_global = create_nn_function(params_global, layer_sizes_global)

# --- Carregar dados ---

file_path = 'data_rna_cobeq_100k.pkl'
library_data = load_data_from_pkl(file_path)

feature_vars = [
    'p_topo', 'valve1', 'valve2', 'valve3', 'valve4',
    'bcs1_freq', 'bcs2_freq', 'bcs3_freq', 'bcs4_freq',
    'booster_freq',
]

label_vars = [
    'q_main1', 'q_main2', 'q_main3', 'q_main4', 'q_tr',
    'P_man', 'P_fbhp1', 'P_fbhp2',
    'P_fbhp3', 'P_fbhp4', 'dP_bcs1', 'dP_bcs2',
    'dP_bcs3', 'dP_bcs4'
]

# --- Normalizar features ---

features_data = np.column_stack([library_data[var] for var in feature_vars])
features_min = features_data.min(axis=0)
features_max = features_data.max(axis=0)
features_normalized = normalize(features_data, features_min, features_max)

# --- Normalizar labels ---

labels_data = np.column_stack([library_data[var] for var in label_vars])
labels_min = labels_data.min(axis=0)
labels_max = labels_data.max(axis=0)
labels_normalized = normalize(labels_data, labels_min, labels_max)

# --- Simulação ponto a ponto ---

n_samples = features_data.shape[0]
outputs_nn = np.zeros((n_samples, len(label_vars)))  # Saída já normalizada

for i in range(n_samples):
    x_input = features_normalized[i, :].reshape(-1, 1)
    y_pred = nn_global(x_input)
    outputs_nn[i, :] = np.array(y_pred.full()).flatten()

# --- Concatenar todas as variáveis ---

outputs_flat = outputs_nn.flatten()
labels_flat = labels_normalized.flatten()

# --- Cálculo das métricas ---

r2 = r2_score(labels_flat, outputs_flat)
corr = np.corrcoef(outputs_flat, labels_flat)[0, 1]

print(f"Coeficiente de Determinação (R²): {r2:.4f}")
print(f"Correlação de Pearson: {corr:.4f}")

# --- Plot único: todas variáveis concatenadas ---

plt.figure(figsize=(16, 8))
plt.scatter(outputs_flat, labels_flat, alpha=0.1, s=1)
plt.plot([-1, 1], [-1, 1], 'r--', label='Ideal', linewidth=3)
plt.xlabel('RNA', fontsize=20)
plt.ylabel('FENOMENOLÓGICO', fontsize=20)
plt.legend(fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()
plt.show()