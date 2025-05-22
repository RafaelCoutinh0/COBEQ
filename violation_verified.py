from initialization_oil_production_basic import mani_model, x0, z0, mani
from numpy import array
from scipy.optimize import fsolve
from pontos_iniciais import u0_ref

u0 = u0_ref
def violantion(u0):
    mani_solver = lambda y: array([float(i) for i in mani.model(0, y[0:-8], y[-8:], u0)])

    y_ss = fsolve(mani_solver, x0+z0)

    z_ss = y_ss[-8:]

    x_ss = y_ss[0:-8]

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
        if x_ss[0] <= 0:
            print(f'Pressão do Manifold Violada {x_ss[0]}')
        if x_ss[4] < qmin1:
            print(f'Downthrust bcs 1 Violada {-x_ss[4] + qmin1}')
        if x_ss[4] < qmax1:
            print(f'Upthrust bcs 1 Violada {x_ss[4] - qmax1}')
        if x_ss[7] < qmin2:
            print(f'Downthrust bcs 1 Violada {-x_ss[7] + qmin2}')
        if x_ss[7] < qmax2:
            print(f'Upthrust bcs 1 Violada {x_ss[7] - qmax2}')
        if x_ss[10] < qmin3:
            print(f'Downthrust bcs 1 Violada {-x_ss[10] + qmin3}')
        if x_ss[10] < qmax3:
            print(f'Upthrust bcs 1 Violada {x_ss[19] - qmax3}')
        if x_ss[13] < qmin4:
            print(f'Downthrust bcs 1 Violada {-x_ss[13] + qmin4}')
        if x_ss[13] < qmax4:
            print(f'Upthrust bcs 1 Violada {x_ss[13] - qmax4}')
        Flag = 0

    if x_ss[2] < 74.14291355695116 or x_ss[5] < 74.14291355695116 or \
            x_ss[8] < 74.14291355695116 or x_ss[11] < 74.14291355695116 and Flag == 1:
        if x_ss[2] < 74.14291355695116:
            diferença = 74.14291355695116 - x_ss[2]
            print(f'Pressão de Topo 1 violada {x_ss[2]} - {diferença/74.14291355695116}% - {diferença}')
        if x_ss[5] < 74.14291355695116:
            diferença = 74.14291355695116 - x_ss[5]
            print(f'Pressão de Topo 2 violada {x_ss[5]} - {diferença/74.14291355695116}% - {diferença}')
        if x_ss[8] < 74.14291355695116:
            diferença = 74.14291355695116 - x_ss[8]
            print(f'Pressão de Topo 3 violada {x_ss[8]} - {diferença/74.14291355695116}% - {diferença}')
        if x_ss[11] < 74.14291355695116:
            diferença = 74.14291355695116 - x_ss[11]
            print(f'Pressão de Topo 4 violada {x_ss[11]} - {diferença/74.14291355695116}% - {diferença}')
        Flag = 0

    return Flag