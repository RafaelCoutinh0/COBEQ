from initialization_oil_production_basic import *
from casadi import *
import numpy as np
from scipy.optimize import fsolve
from pontos_iniciais import u0_ref

# Variáveis manipuláveis (entradas do modelo)
u = MX.sym('u', 10)  # [f_BP, p_topside, f_ESP_1, alpha_1, ..., f_ESP_4, alpha_4]

# Variáveis do sistema (saídas do modelo)
x = MX.sym('x', 14)  # Estados (pressões, vazões, etc.)
z = MX.sym('z', 8)   # Variáveis algébricas

rho = 984
g = 9.81
eff = 0.98

objective =  -(3000 * x[1]) + ((9653.04 * (x[1]/3600) * (1.0963e3 * (u[0]/ 50) ** 2) * 0.001) + \
        ((x[4]/3600) * z[1] * 1e2) + \
        ((x[7]/3600) * z[3] * 1e2) + \
        ((x[10]/3600) * z[5] * 1e2)  + \
        ((x[13]/3600) * z[7] * 1e2)) * 0.91

# Certifique-se de que a função mani.model esteja implementada corretamente
mani_model = mani.model(0, x, z, u)

# Restrições de vazão para cada poço com os valores diretos
restqmain1 = vertcat(x[4] - ((z[1] + 334.2554) / 19.0913), ((z[1] + 193.8028) / 4.4338) - x[4])
restqmain2 = vertcat(x[7] - ((z[3] + 334.2554) / 19.0913), ((z[3] + 193.8028) / 4.4338) - x[7])
restqmain3 = vertcat(x[10] - ((z[5] + 334.2554) / 19.0913), ((z[5] + 193.8028) / 4.4338) - x[10])
restqmain4 = vertcat(x[13] - ((z[7] + 334.2554) / 19.0913), ((z[7] + 193.8028) / 4.4338) - x[13])
# Parâmetros do campo (em SI)
Patm = 0.8                # pressão atmosférica em bar              # aceleração da gravidade (m/s²)
TVD = 1029.2            # profundidade vertical (m)

# Cálculo da pressão estática (em bar)
P_static = Patm + (rho * g * TVD) / 1e5   # ≈ 100.35 bar

# Cálculo da queda de pressão por circulação (ΔP_circ)
# Aqui usamos o termo presente no seu código: ((82.1096/3600)/6.9651e-9)
DeltaP_circ = (((82.1096 / 3600) / 6.9651e-9)) / 1e5  # ≈ 32.75 bar

# Aplicando um fator de segurança (por exemplo, 1.1)
P_target = (P_static - DeltaP_circ) * 1.1# ≈ 74.14 bar
# Restrições de pressão de fundo de poço (restfbprest)
restfbprest1 = x[2] - P_target
restfbprest2 = x[5] - P_target
restfbprest3 = x[8] - P_target
restfbprest4 = x[11] - P_target

# Restrições de igualdade (modelo do sistema)
g_equality = vertcat(*mani_model)

# Restrições de desigualdade (limites operacionais)
g_inequality = vertcat(restqmain1, restqmain2, restqmain3, restqmain4,restfbprest1, restfbprest2, restfbprest3, restfbprest4)

# Concatenar todas as restrições
g_constraints = vertcat(g_equality, g_inequality)

# Definir limites para as restrições (lbg e ubg)
num_eq = g_equality.shape[0]  # Número de igualdades
num_ineq = g_inequality.shape[0]  # Número de desigualdades

lbg = [0.0] * num_eq + [0.0] * num_ineq  # Igualdades: 0 ≤ g ≤ 0 | Desigualdades: 0 ≤ g ≤ ∞
ubg = [0.0] * num_eq + [np.inf] * num_ineq

# Configuração do problema de otimização
nlp = {'x': vertcat(x, z, u), 'f': objective, 'g': g_constraints}
solver = nlpsol('solver', 'ipopt', nlp)
solver_caso_a  = solver
u_ptopo = 0.8e5
# Valores iniciais para as variáveis manipuláveis (u), estados (x) e algébricas (z)
u0 = u0_ref
mani_solver = lambda y: np.array([float(i) for i in mani.model(0, y[:14], y[14:], u0)])
x0 = [76.52500, 4 * 85,
      64.11666, 120.91641, 85,
      64.11666, 120.91641, 85,
      64.11666, 120.91641, 85,
      64.11666, 120.91641, 85]

z0 = [30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625]
y_ss = fsolve(mani_solver, np.concatenate((x0, z0)))

# Atualizar x0 e z0 com os resultados do fsolve
x0 = y_ss[:14]
z0 = y_ss[14:]

x0_full = np.concatenate((x0, z0, u0))

# Definir limites das variáveis
lbx = [0] +[0]+ [0] * 12 + [0] * 8 + [35, u_ptopo, 35, 0, 35, 0, 35, 0, 35, 0] #Inferiores
ubx = [np.inf] * 14 + [np.inf] * 8 + [65, u_ptopo, 65, 1, 65, 1, 65,1, 65, 1]  # Superiores
x0_caso_a = x0_full
lbx_caso_a = lbx
ubx_caso_a = ubx
lbg_caso_a = lbg
ubg_caso_a = ubg
# Resolver o problema de otimização
sol = solver(x0=x0_full, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
solver_stats = solver.stats()
tempo_ipopt = solver_stats['t_proc_total']
optimal_solution = sol['x']

# Extrair resultados otimizados
x_opt = optimal_solution[:14]
z_opt = optimal_solution[14:22]
u_opt = optimal_solution[22:]

# Imprimir resultados otimizados

state_names = [
    "p_man (bar)", "q_tr (m^3/h)",
    "P_fbhp_1 (bar)", "P_choke_1 (bar)", "q_mean_1 (m^3/h)",
    "P_fbhp_2 (bar)", "P_choke_2 (bar)", "q_mean_2 (m^3/h)",
    "P_fbhp_3 (bar)", "P_choke_3 (bar)", "q_mean_3 (m^3/h)",
    "P_fbhp_4 (bar)", "P_choke_4 (bar)", "q_mean_4 (m^3/h)"
]

algebraic_names = [
    "P_intake_1 (bar)", "dP_bcs_1 (bar)",
    "P_intake_2 (bar)", "dP_bcs_2 (bar)",
    "P_intake_3 (bar)", "dP_bcs_3 (bar)",
    "P_intake_4 (bar)", "dP_bcs_4 (bar)"
]

control_names = [
    "f_BP (Hz)", "p_topside (Pa)",
    "f_ESP_1 (Hz)", "alpha_1 (-)",
    "f_ESP_2 (Hz)", "alpha_2 (-)",
    "f_ESP_3 (Hz)", "alpha_3 (-)",
    "f_ESP_4 (Hz)", "alpha_4 (-)"
]
# Verificar consistência com fsolve
mani_solver = lambda y: np.array([float(i) for i in mani.model(0, y[:14], y[14:], u_opt)])
y_ss = fsolve(mani_solver, np.concatenate((x0, z0)))

# Separar resultados do fsolve
x_ss = y_ss[:14]
z_ss = y_ss[14:]

# Comparar otimização e fsolve
# Códigos ANSI para negrito e cores
BOLD = '\033[1m'
RESET = '\033[0m'
CYAN = '\033[36m'
GREEN = '\033[32m'
YELLOW = '\033[33m'

energybooster = (9653.04 * (x_ss[1]/3600) * (1.0963e3 * (u_opt[0]/50) ** 2) * 0.001)
energybcs1 = (x_ss[4]/3600) * (z_ss[1]*1e2)
energybcs2 = (x_ss[7]/3600) * (z_ss[3]*1e2)
energybcs3 = (x_ss[10]/3600) * (z_ss[5]*1e2)
energybcs4 = (x_ss[13]/3600) * (z_ss[7]*1e2)
energytot = (energybooster + energybcs1 + energybcs2 + energybcs3 + energybcs4) * 0.91
venda = 3000 * x_ss[1]

objective =  -(3000 * x_ss[1]) + ((9653.04 * (x_ss[1]/3600) * (1.0963e3 * (u_opt[0]/ 50) ** 2) * 0.001) + \
        ((x_ss[4]/3600) * z_ss[1] * 1e2) + \
        ((x_ss[7]/3600) * z_ss[3] * 1e2) + \
        ((x_ss[10]/3600) * z_ss[5] * 1e2)  + \
        ((x_ss[13]/3600) * z_ss[7] * 1e2)) * 0.91

def casadi_to_list(mx_vector):
    return [float(x) for x in np.array(mx_vector).flatten()]

u_opt= casadi_to_list(u_opt)

x_ss_relevante = [
    x_ss[1],
    x_ss[4], x_ss[7], x_ss[10], x_ss[13],
    z_ss[1], z_ss[3], z_ss[5], z_ss[7],
    x_ss[2], x_ss[5], x_ss[8], x_ss[11],
    x_ss[0],
    *[float(val) for val in u_opt]]
solver_padrao = solver
x0_padrao = x0_full  # já montado

optimize_padrao = [x_ss_relevante,u_opt, objective, venda, energytot[0], energybooster[0], energybcs1, energybcs2, energybcs3, energybcs4,tempo_ipopt]