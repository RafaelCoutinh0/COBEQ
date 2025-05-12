# -*- coding: utf-8 -*-

"""
File to simulate a manifold with 4 wells

Adapted from:
Rasmus (2011) - Automatic Start-up and Control of Artificially Lifted Wells

@authors: Rodrigo Lima Meira e Daniel Diniz Santana
"""
import numpy as np
#%% Package import
from numpy import array
import matplotlib.pyplot as plt
import matplotlib.pyplot
from matplotlib import rcParams
from casadi import MX, interpolant, Function, sqrt, vertcat, integrator, jacobian, transpose
# from scipy.special import functions
from plotgraf import *
from bcs_models import *
from manifold import *
from numpy import linspace, array, eye, zeros, repeat, concatenate, delete, diag
from numpy.linalg import inv
from matplotlib.pyplot import plot, figure, title
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
import random
from scipy.optimize import fsolve
from control import ss, tf, sample_system, forced_response
from scipy.signal import ss2tf
#%% Creating functions of BCS, Choke and Pipes

def bcs_functions(f, q):
    """
    BCS Head, Efficiency and Power as function of frequency (f) and flow (q)
    :param f: pump frequency (Hz)
    :param q: flow [m^3/s]
    :return:
    H: head [m]
    eff: efficiency
    pot: power
    """

    f0 = 50
    q0 = q*(f0/f)
    H0 = -29.5615*(q0/0.0353)**4+25.3722*(q0/0.0353)**3-8.7944*(q0/0.0353)**2-8.5706*(q0/0.0353)+21.4278
    H = H0 * (f / f0) ** 2.
    eff = 1
    pot = 1
    return H, eff, pot


def choke_fun(alpha):
    """
    Valve characteristic function
    :param alpha: valve opening  (0 to 1)
    :return: choke characteristic
    """
    # Valve opening
    zc = [0, 13, 15, 17, 19, 22, 25, 29, 30, 32, 34, 36, 39, 41, 44,
          46, 49, 52, 55, 58, 61, 64, 67, 71, 75, 78, 82, 86, 91, 96, 100.01]

    # valve characteristic
    G = [0, 0.011052632, 0.024210526, 0.043157895, 0.067894737, 0.097894737,
         0.133157895, 0.173684211, 0.195789474, 0.219473684, 0.244736842,
         0.271052632, 0.298947368, 0.328421053, 0.358947368, 0.390526316,
         0.423684211, 0.458421053, 0.494210526, 0.531578947, 0.570526316,
         0.610526316, 0.651578947, 0.694210526, 0.738421053, 0.784210526,
         0.830526316, 0.878947368, 0.928421053, 0.979473684, 1]

    fun_choke = interpolant('choke', 'bspline', [zc], G)
    return fun_choke(alpha * 100)


# Pipes creation

# well pipe before BCS
pipe_sec1 = Pipe(0.081985330499706 * 2, 3.078838005940556e3, 1029.2 - 920, 1.5e+9, 984, 0.3, 5.752218216772682e+06,
                 3.903249155428134e+07)
# well pipe after BCS
pipe_sec2 = Pipe(0.0595 * 2, 9.222097306189842e+02, 920 - 126.5400, 1.5e9, 984, 4, 7.455247950618545e+06,
                 6.264914244217266e+07)
# manifold pipe
pipe_mani = Pipe(0.0595 * 2, 500, 0, 8.3022e+6, 984, 4, 5.752218216772682e+06, 3.903249155428134e+07)

#%% Defining the CasADi function for pumps and valves

f_ca = MX.sym('f', 1)
q_ca = MX.sym('q', 1)
alpha_ca = MX.sym('alpha', 1)

H_fun, eff_fun, pot_fun = bcs_functions(f_ca, q_ca)

head_fun = Function('head', [f_ca, q_ca], [64 * bcs_functions(f_ca, q_ca)[0]])
efficiency_fun = Function('efficiency', [f_ca, q_ca], [eff_fun])
power_fun = Function('power', [f_ca, q_ca], [pot_fun])

# Booster pump Head [m]
booster_fun = Function('booster', [f_ca, q_ca], [1.0963e3 * (f_ca / 50) ** 2])

valve_fun = Function('choke', [alpha_ca], [choke_fun(alpha_ca)])

# Defining the BCS of the wells and booster pump

bcs1 = Pump(head_fun, efficiency_fun, power_fun)
bcs2 = Pump(head_fun, efficiency_fun, power_fun)
bcs3 = Pump(head_fun, efficiency_fun, power_fun)
bcs4 = Pump(head_fun, efficiency_fun, power_fun)

booster = Pump(booster_fun, efficiency_fun, power_fun)

# Defining the valves in the wells
# k_choke: valve constant [m^3/s/Pa^0.5]
choke1 = Choke(212.54 / (6e+4 * sqrt(1e+5)), valve_fun)
choke2 = Choke(212.54 / (6e+4 * sqrt(1e+5)), valve_fun)
choke3 = Choke(212.54 / (6e+4 * sqrt(1e+5)), valve_fun)
choke4 = Choke(212.54 / (6e+4 * sqrt(1e+5)), valve_fun)

# Defining the wells and the manifold
#choke1 = [m^3/s/Pa^0.5]

well1 = Well(pipe_sec1, pipe_sec2, bcs1, choke1, 6.9651e-9, 9800000)
well2 = Well(pipe_sec1, pipe_sec2, bcs2, choke2, 6.9651e-9, 9800000)
well3 = Well(pipe_sec1, pipe_sec2, bcs3, choke3, 6.9651e-9, 9800000)
well4 = Well(pipe_sec1, pipe_sec2, bcs4, choke4, 6.9651e-9, 9800000)

mani = Manifold(pipe_mani, booster, 0, 0, [well1, well2, well3, well4])

#%% Defining the simulation variables
#time
t = MX.sym('t')
# Inputs
f_BP = MX.sym('f_BP')  # [Hz] Boost Pump frequency
p_topside = MX.sym('p_topside')  # [Hz] Boost Pump frequency
u = [f_BP, p_topside]

f_ESP_1 = MX.sym('f_ESP_1')  # [Hz] ESP frequency
alpha_1 = MX.sym('alpha_1')  # [%] Choke opening
u += [f_ESP_1, alpha_1]

f_ESP_2 = MX.sym('f_ESP_2')  # [Hz] ESP frequency
alpha_2 = MX.sym('alpha_2')  # [%] Choke opening
u += [f_ESP_2, alpha_2]

f_ESP_3 = MX.sym('f_ESP_3')  # [Hz] ESP frequency
alpha_3 = MX.sym('alpha_3')  # [%] Choke opening
u += [f_ESP_3, alpha_3]

f_ESP_4 = MX.sym('f_ESP_4')  # [Hz] ESP frequency
alpha_4 = MX.sym('alpha_4')  # [%] Choke opening
u += [f_ESP_4, alpha_4]

# States and algebraic variables
p_man = MX.sym('p_man')  # [bar] manifold pressure
q_tr = MX.sym('q_tr')  # [m^3/h] Flow through the transportation line
x = [p_man, q_tr] # states
z = [] # algebraic variables

# Well 1
P_fbhp_1 = MX.sym('P_fbhp_1')  # [bar] Pressure fbhp
P_choke_1 = MX.sym('P_choke_1')  # [bar] Pressure in chokes
q_mean_1 = MX.sym('q_mean_1')  # [m^3/h] Average flow in the wells
P_intake_1 = MX.sym('P_ìntake_1')  # [bar] Pressure intake in ESP's
dP_bcs_1 = MX.sym('dP_bcs_1')  # [bar] Pressure discharge in ESP's
x += [P_fbhp_1, P_choke_1, q_mean_1]
z += [P_intake_1, dP_bcs_1]

# Well 2
P_fbhp_2 = MX.sym('P_fbhp_2')  # [bar] Pressure fbhp in ESP's
P_choke_2 = MX.sym('P_choke_2')  # [bar] Pressure in chokes
q_mean_2 = MX.sym('q_mean_2')  # [m^3/h] Average flow in the wells
P_intake_2 = MX.sym('P_ìntake_2')  # [bar] Pressure intake in ESP's
dP_bcs_2 = MX.sym('dP_bcs_2')  # [bar] Pressure discharge in ESP's
x += [P_fbhp_2, P_choke_2, q_mean_2]
z += [P_intake_2, dP_bcs_2]

# Well 3
P_fbhp_3 = MX.sym('P_fbhp_3')  # [bar] Pressure fbhp in ESP's
P_choke_3 = MX.sym('P_choke_3')  # [bar] Pressure in chokes
q_mean_3 = MX.sym('q_mean_3')  # [m^3/h] Average flow in the wells
P_intake_3 = MX.sym('P_ìntake_3')  # [bar] Pressure intake in ESP's
dP_bcs_3 = MX.sym('dP_bcs_3')  # [bar] Pressure discharge in ESP's
x += [P_fbhp_3, P_choke_3, q_mean_3]
z += [P_intake_3, dP_bcs_3]

# Well 4
P_fbhp_4 = MX.sym('P_fbhp_4')  # [bar] Pressure fbhp in ESP's
P_choke_4 = MX.sym('P_choke_4')  # [bar] Pressure in chokes
q_mean_4 = MX.sym('q_mean_4')  # [m^3/h] Average flow in the wells
P_intake_4 = MX.sym('P_ìntake_4')  # [bar] Pressure intake in ESP's
dP_bcs_4 = MX.sym('dP_bcs_4')  # [bar] Pressure discharge in ESP's
x += [P_fbhp_4, P_choke_4, q_mean_4]
z += [P_intake_4, dP_bcs_4]

# Defining the symbolic manifold model
mani_model = mani.model(0, x, z, u)

# %% Evaluation of steady-state
u0 = [56., 10 ** 5, 50., .5, 50., .5, 50., .5, 50., .5]

x0 = [76.52500, 4 * 85,
      64.11666, 120.91641, 85,
      64.11666, 120.91641, 85,
      64.11666, 120.91641, 85,
      64.11666, 120.91641, 85]

z0 = [30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625,
      30.03625, 239.95338-30.03625]

mani_solver = lambda y: array([float(i) for i in mani.model(0, y[0:-8], y[-8:], u0)])

y_ss = fsolve(mani_solver, x0+z0)

z_ss = y_ss[-8:]

x_ss = y_ss[0:-8]

#%% Dynamic Simulation
dae = {'x': vertcat(*x), 'z': vertcat(*z), 'p': vertcat(*u), 'ode': vertcat(*mani_model[0:-8]),
       'alg': vertcat(*mani_model[-8:])}

tfinal = 1000 # [h]

grid = linspace(0, tfinal, 100)

F = integrator('F', 'idas', dae, 0, grid)

res = F(x0 = x_ss, z0 = z_ss, p = u0)

#%% Novidades

def plotar_graficos(n_pert):
    global res
    global tfinal

    Lista_xf = []
    Lista_zf = []
    Lista_xf.append(res["xf"])
    Lista_zf.append(res["zf"])
    x0 = Lista_xf[-1][:, -1]
    z0 = Lista_zf[-1][:, -1]
    map_est = []
    map_est.append(x0)
    map_est.append(z0)
    Lista_zf = np.array(Lista_zf)
    Lista_xf = np.array(Lista_xf)
    Lista_zf_reshaped = Lista_zf.reshape(8, 100)
    Lista_xf_reshaped = Lista_xf.reshape(14, 100)

    # criando as pertubações de u0
    valve_open1 = np.random.uniform(.42, 1, n_pert)
    valve_open2 = np.random.uniform(.42, 1, n_pert)
    valve_open3 = np.random.uniform(.42, 1, n_pert)
    valve_open4 = np.random.uniform(.42, 1, n_pert)
    # bcs_freq1 = np.random.randint(35., 65., n_pert)
    # bcs_freq2 = np.random.randint(35., 65., n_pert)
    # bcs_freq3 = np.random.randint(35., 65., n_pert)
    # bcs_freq4 = np.random.randint(35., 65., n_pert)
    booster_freq = np.random.uniform(35., 65., n_pert)
    p_topo = np.random.uniform(8, 12, n_pert)

    grid_cont = 1
    for i in range(n_pert):
        grid_cont += 1
        delta = 1000
        grid = linspace(tfinal, tfinal + delta, 100)
        tfinal += delta
        u0 = [booster_freq[i], p_topo[i] ** 5, 50., valve_open1[i], 50., valve_open2[i], 50., valve_open3[i], 50.,
              valve_open4[i]]
        res = F(x0=x0, z0=z0, p=u0)
        x0 = res["xf"][:, -1]
        z0 = res["zf"][:, -1]
        map_est.append(x0)
        map_est.append(z0)

        Lista_xf_reshaped = np.hstack((Lista_xf_reshaped, np.array(res["xf"])))
        Lista_zf_reshaped = np.hstack((Lista_zf_reshaped, np.array(res["zf"])))

        #Plotted Graphs
        rcParams['axes.formatter.useoffset'] = False
        grid = linspace(0, tfinal, 100 * grid_cont)

    def Auto_plot(i, t, xl, yl, c):
        plt.plot(grid, i.transpose(), c)
        matplotlib.pyplot.title(t)
        matplotlib.pyplot.xlabel(xl)
        matplotlib.pyplot.ylabel(yl)
        conc = np.concatenate(i)
        y_min, y_max = np.min(conc), np.max(conc)
        plt.ylim([y_min - 0.1 * abs(y_max), y_max + 0.1 * abs(y_max)])
        plt.grid()
        plt.show()

    Auto_plot(Lista_zf_reshaped[[1, 3, 5, 7], :], "Pressure Discharge in ESP's", 'Time/(h)', 'Pressure/(bar)', 'b')
    Auto_plot(Lista_xf_reshaped[[2, 5, 8, 11], :], "Pressure fbhp in ESP's", 'Time/(h)', 'Pressure/(bar)', 'r')
    Auto_plot(Lista_xf_reshaped[[3, 6, 9, 12], :], 'Pressure in Chokes', 'Time/(h)', 'Pressure/(bar)', 'g')
    Auto_plot(Lista_xf_reshaped[[4, 7, 10, 13], :], 'Average Flow in the Wells', 'Time/(h)', 'Flow Rate/(m^3/h)', 'k')
    Auto_plot(Lista_xf_reshaped[[1], :], 'Flow Through the Transportation Line', 'Time/(h)', 'Flow Rate/(m^3/h)','y')
    Auto_plot(Lista_xf_reshaped[[0], :], 'Manifold Pressure', 'Time/(h)', 'Pressure/(bar)', 'm')

    #p_intake é desnecessário
    # Auto_plot(Lista_zf_reshaped[[0, 2, 4, 6], :],"Pressure Intake in ESP's", 'Time/(h)', 'Pressure/(bar)')




 # %% mapeando estacionários:
def mapping_stationary(n_pert):

    valve_open1 = np.random.uniform(.1, .9, n_pert)
    valve_open2 = np.random.uniform(.1, .9, n_pert)
    valve_open3 = np.random.uniform(.1, .9, n_pert)
    valve_open4 = np.random.uniform(.1, .9, n_pert)
    bcs_freq1 = np.random.uniform(35., 65., n_pert)
    bcs_freq2 = np.random.uniform(35., 65., n_pert)
    bcs_freq3 = np.random.uniform(35., 65., n_pert)
    bcs_freq4 = np.random.uniform(35., 65., n_pert)
    booster_freq = np.random.uniform(35., 65., n_pert)
    p_topo = np.random.uniform(0.8, 1.2, n_pert)


    valves_rand = [valve_open1, valve_open2, valve_open3, valve_open4]
    bcs_rand = [bcs_freq1, bcs_freq2, bcs_freq3, bcs_freq4]
    booster_rand = booster_freq
    p_topo_rand = p_topo

    global x0
    global z0

    est_P_man = []
    est_q_tr = []
    est_P_fbhp1 = []
    est_P_fbhp2 = []
    est_P_fbhp3= []
    est_P_fbhp4 = []
    est_P_choke1 = []
    est_P_choke2 = []
    est_P_choke3 = []
    est_P_choke4 = []
    est_q_main1 = []
    est_q_main2 = []
    est_q_main3 = []
    est_q_main4 = []
    est_P_intake1 = []
    est_P_intake2 = []
    est_P_intake3 = []
    est_P_intake4 = []
    est_dP_bcs1 = []
    est_dP_bcs2 = []
    est_dP_bcs3 = []
    est_dP_bcs4 = []
    flag = []

    i = 0
    contador = 0
    while contador < n_pert :
        u0 = [booster_freq[i], p_topo[i] * 10 ** 5, bcs_freq1[i], valve_open1[i], bcs_freq2[i], valve_open2[i], bcs_freq3[i], valve_open3[i], bcs_freq4[i],valve_open4[i]]
        mani_solver = lambda y: array([float(i) for i in mani.model(0, y[0:-8], y[-8:], u0)])
        y_ss = fsolve(mani_solver, x0+z0)
        z_ss = y_ss[-8:]
        x_ss = y_ss[0:-8]
        if  x_ss[1] == 340:
            valve_open1[i] = np.random.uniform(0, 1)
            valve_open2[i] = np.random.uniform(0, 1)
            valve_open3[i] = np.random.uniform(0, 1)
            valve_open4[i] = np.random.uniform(0, 1)
            bcs_freq1[i] = np.random.uniform(35., 65.)
            bcs_freq2[i] = np.random.uniform(35., 65.)
            bcs_freq3[i] = np.random.uniform(35., 65.)
            bcs_freq4[i] = np.random.uniform(35., 65.)
            booster_freq[i] = np.random.uniform(35., 65.)
            p_topo[i] = np.random.uniform(0.8, 1.2)
            continue
        else:
            i += 1

        a_min = (206.6-58.07)/(28.55-20.77)
        b_min = 58.07-a_min*20.55

        a_max = (170.0983885726676 - 44.8570768595651) / (82.07399108766865 - 53.82716056845215)
        b_max = 44.8570768595651 - a_max * 53.82716056845215

        qmin1 = (z_ss[1] - b_min) / a_min
        qmax1 = (z_ss[1] - b_max) / a_max
        qmin2 = (z_ss[3] - b_min) / a_min
        qmax2 = (z_ss[3] - b_max) / a_max
        qmin3 = (z_ss[5] - b_min) / a_min
        qmax3 = (z_ss[5] - b_max) / a_max
        qmin4 = (z_ss[7] - b_min) / a_min
        qmax4 = (z_ss[7] - b_max) / a_max
        restqmain1 = x_ss[4] >= qmin1 and x_ss[4] <= qmax1
        restqmain2 = x_ss[7] >= qmin2 and x_ss[7] <= qmax2
        restqmain3 = x_ss[10] >= qmin3 and x_ss[10] <= qmax3
        restqmain4 = x_ss[13] >= qmin4 and x_ss[13] <= qmax4


        # Adicionando na lista se for positivo

        if x_ss[0] > 0 and restqmain1 and restqmain2 and restqmain3 and restqmain4:
            Flag = 1
        else:
            Flag = 0
        if x_ss[2] < 74.14291355695116 or x_ss[5] < 74.14291355695116 or \
                x_ss[8] < 74.14291355695116 or x_ss[11] < 74.14291355695116 and Flag == 1:
            Flag = 0
        flag.append(Flag)
        est_P_man.append(x_ss[0])
        est_q_tr.append(x_ss[1])
        est_P_fbhp1.append(x_ss[2])
        est_P_fbhp2.append(x_ss[5])
        est_P_fbhp3.append(x_ss[8])
        est_P_fbhp4.append(x_ss[11])
        est_P_choke1.append(x_ss[3])
        est_P_choke2.append(x_ss[6])
        est_P_choke3.append(x_ss[9])
        est_P_choke4.append(x_ss[12])
        est_q_main1.append(x_ss[4])
        est_q_main2.append(x_ss[7])
        est_q_main3.append(x_ss[10])
        est_q_main4.append(x_ss[13])
        est_P_intake1.append(z_ss[0])
        est_P_intake2.append(z_ss[2])
        est_P_intake3.append(z_ss[4])
        est_P_intake4.append(z_ss[6])
        est_dP_bcs1.append(z_ss[1])
        est_dP_bcs2.append(z_ss[3])
        est_dP_bcs3.append(z_ss[5])
        est_dP_bcs4.append(z_ss[7])
        contador += 1
        print(f'--{contador}/{n_pert}--')
    # Plotando o Gráfico

    plt.figure(dpi=250)
    plt.plot(np.array(est_q_tr)[np.array(flag)==0], np.array(est_P_man)[np.array(flag)==0], 'r.')
    plt.plot(np.array(est_q_tr)[np.array(flag)==1], np.array(est_P_man)[np.array(flag)==1], 'b.')
    plt.plot([110, 225], [0, 0], 'k--', linewidth=3)
    plt.xlabel(r"$q_{tr}$ /(m$^3 \cdot$ h$^{-1}$)", fontsize=15)
    plt.ylabel('$P_{man}$ /bar',fontsize = 15)
    plt.grid()
    plt.show()

    plt.figure(dpi=250)
    plt.plot(np.array(est_q_main1)[np.array(flag)==0], np.array(est_dP_bcs1)[np.array(flag)==0], 'r.')
    plt.plot(np.array(est_q_main1)[np.array(flag)==1], np.array(est_dP_bcs1)[np.array(flag)==1], 'b.')
    plt.plot([28.55, 20.77], [206.6, 58.07], 'k--', linewidth=3)
    plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
    plt.xlabel(r"$q_{main1}$ /(m$^3 \cdot$ h$^{-1}$)", fontsize=15)
    plt.ylabel('$dP_{bcs1}$ /bar',fontsize = 15)
    plt.grid()
    plt.show()

    plt.figure(dpi=250)
    plt.plot(np.array(est_q_main2)[np.array(flag)==0], np.array(est_dP_bcs2)[np.array(flag)==0], 'r.')
    plt.plot(np.array(est_q_main2)[np.array(flag)==1], np.array(est_dP_bcs2)[np.array(flag)==1], 'b.')
    plt.plot([28.55, 20.77], [206.6, 58.07], 'k--', linewidth=3)
    plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
    plt.xlabel(r"$q_{main2}$ /(m$^3 \cdot$ h$^{-1}$)", fontsize=15)
    plt.ylabel('$dP_{bcs2}$ /bar',fontsize = 15)
    plt.grid()
    plt.show()

    plt.figure(dpi=250)
    plt.plot(np.array(est_q_main3)[np.array(flag)==0], np.array(est_dP_bcs3)[np.array(flag)==0], 'r.')
    plt.plot(np.array(est_q_main3)[np.array(flag)==1], np.array(est_dP_bcs3)[np.array(flag)==1], 'b.')
    plt.plot([28.55, 20.77], [206.6, 58.07], 'k--', linewidth=3)
    plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
    plt.xlabel(r"$q_{main3}$ /(m$^3 \cdot$ h$^{-1}$)", fontsize=15)
    plt.ylabel('$dP_{bcs3}$ /bar',fontsize = 15)
    plt.grid()
    plt.show()

    plt.figure(dpi=250)
    plt.plot(np.array(est_q_main4)[np.array(flag)==0], np.array(est_dP_bcs4)[np.array(flag)==0], 'r.')
    plt.plot(np.array(est_q_main4)[np.array(flag)==1], np.array(est_dP_bcs4)[np.array(flag)==1], 'b.')
    plt.plot([28.55, 20.77],[206.6, 58.07], 'k--', linewidth=3)
    plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
    plt.xlabel(r"$q_{main2}$ /(m$^3 \cdot$ h$^{-1}$)", fontsize=15)
    plt.ylabel('$dP_{bcs4}$ /bar',fontsize = 15)
    plt.grid()
    plt.show()

    dados = {

        'p_topo': p_topo_rand.tolist(),
        'valve1': valve_open1,
        'valve2': valve_open2,
        'valve3': valve_open3,
        'valve4': valve_open4,
        'bcs1_freq': bcs_freq1,
        'bcs2_freq': bcs_freq2,
        'bcs3_freq': bcs_freq3,
        'bcs4_freq': bcs_freq4,
        'booster_freq': booster_rand.tolist(),
        'P_man': est_P_man,
        'P_fbhp1': est_P_fbhp1,
        'P_fbhp2': est_P_fbhp2,
        'P_fbhp3': est_P_fbhp3,
        'P_fbhp4': est_P_fbhp4,
        'P_choke1': est_P_choke1,
        'P_choke2': est_P_choke2,
        'P_choke3': est_P_choke3,
        'P_choke4': est_P_choke4,
        'P_intake1': est_P_intake1,
        'P_intake2': est_P_intake2,
        'P_intake3': est_P_intake3,
        'P_intake4': est_P_intake4,
        'dP_bcs1': est_dP_bcs1,
        'dP_bcs2': est_dP_bcs2,
        'dP_bcs3': est_dP_bcs3,
        'dP_bcs4': est_dP_bcs4,
        'q_main1': est_q_main1,
        'q_main2': est_q_main2,
        'q_main3': est_q_main3,
        'q_main4': est_q_main4,
        'q_tr': est_q_tr,
        'flag': flag,
    }
    return dados

