from initialization_oil_production_basic import *

def plotar(n):
    if n == 1:
        plt.figure(dpi=250)
        plt.plot(np.array(dados_unidos['q_tr'])[np.array(dados_unidos['flag']) == 0],
                 np.array(dados_unidos['P_man'])[np.array(dados_unidos['flag']) == 0], 'r.')
        plt.plot(np.array(dados_unidos['q_tr'])[np.array(dados_unidos['flag']) == 1],
                 np.array(dados_unidos['P_man'])[np.array(dados_unidos['flag']) == 1], 'b.')
        plt.plot([110, 225], [0, 0], 'k--', linewidth=3)
        plt.xlabel(r'$q_{tr}$ /(m$^3 \ cdot$ h$^{-1}$)', fontsize=15)
        plt.ylabel('$P_{man}$ /bar', fontsize=15)
        plt.grid()
        plt.show()

        plt.figure(dpi=250)
        plt.plot(np.array(dados_unidos['q_main1'])[np.array(dados_unidos['flag']) == 0],
                 np.array(dados_unidos['dP_bcs1'])[np.array(dados_unidos['flag']) == 0], 'r.')
        plt.plot(np.array(dados_unidos['q_main1'])[np.array(dados_unidos['flag']) == 1],
                 np.array(dados_unidos['dP_bcs1'])[np.array(dados_unidos['flag']) == 1], 'b.')
        plt.plot([28.55, 20.77], [206.6, 58.07], 'k--', linewidth=3)
        plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
        plt.xlabel(r'$q_{main1}$ /(m$^3 \ cdot$ h$^{-1}$)', fontsize=15)
        plt.ylabel('$dP_{bcs1}$ /bar', fontsize=15)
        plt.grid()
        plt.show()

        plt.figure(dpi=250)
        plt.plot(np.array(dados_unidos['q_main2'])[np.array(dados_unidos['flag']) == 0],
                 np.array(dados_unidos['dP_bcs2'])[np.array(dados_unidos['flag']) == 0], 'r.')
        plt.plot(np.array(dados_unidos['q_main2'])[np.array(dados_unidos['flag']) == 1],
                 np.array(dados_unidos['dP_bcs2'])[np.array(dados_unidos['flag']) == 1], 'b.')
        plt.plot([28.55, 20.77], [206.6, 58.07], 'k--', linewidth=3)
        plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
        plt.xlabel(r'$q_{main2}$ /(m$^3 \ cdot$ h$^{-1}$)', fontsize=15)
        plt.ylabel('$dP_{bcs2}$ /bar', fontsize=15)
        plt.grid()
        plt.show()

        plt.figure(dpi=250)
        plt.plot(np.array(dados_unidos['q_main3'])[np.array(dados_unidos['flag']) == 0],
                 np.array(dados_unidos['dP_bcs3'])[np.array(dados_unidos['flag']) == 0], 'r.')
        plt.plot(np.array(dados_unidos['q_main3'])[np.array(dados_unidos['flag']) == 1],
                 np.array(dados_unidos['dP_bcs3'])[np.array(dados_unidos['flag']) == 1], 'b.')
        plt.plot([28.55, 20.77], [206.6, 58.07], 'k--', linewidth=3)
        plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
        plt.xlabel(r'$q_{main3}$ /(m$^3 \ cdot$ h$^{-1}$)', fontsize=15)
        plt.ylabel('$dP_{bcs3}$ /bar', fontsize=15)
        plt.grid()
        plt.show()

        plt.figure(dpi=250)
        plt.plot(np.array(dados_unidos['q_main4'])[np.array(dados_unidos['flag']) == 0],
                 np.array(dados_unidos['dP_bcs4'])[np.array(dados_unidos['flag']) == 0], 'r.')
        plt.plot(np.array(dados_unidos['q_main4'])[np.array(dados_unidos['flag']) == 1],
                 np.array(dados_unidos['dP_bcs4'])[np.array(dados_unidos['flag']) == 1], 'b.')
        plt.plot([28.55, 20.77], [206.6, 58.07], 'k--', linewidth=3)
        plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
        plt.xlabel(r'$q_{main4}$ /(m$^3 \ cdot$ h$^{-1}$)', fontsize=15)
        plt.ylabel('$dP_{bcs4}$ /bar', fontsize=15)
        plt.grid()
        plt.show()
    if n == 2:
        plt.figure(dpi=250)
        plt.plot(np.array(dados_anteriores['q_tr'])[np.array(dados_anteriores['flag']) == 0],
                 np.array(dados_anteriores['P_man'])[np.array(dados_anteriores['flag']) == 0], 'r.')
        plt.plot(np.array(dados_anteriores['q_tr'])[np.array(dados_anteriores['flag']) == 1],
                 np.array(dados_anteriores['P_man'])[np.array(dados_anteriores['flag']) == 1], 'b.')
        plt.plot([110, 225], [0, 0], 'k--', linewidth=3)
        plt.xlabel('$q_{tr}$ /(m$^3 \\cdot$ h$^{-1}$)', fontsize=15)
        plt.ylabel('$P_{man}$ /bar', fontsize=15)
        plt.grid()
        plt.show()

        plt.figure(dpi=250)
        plt.plot(np.array(dados_anteriores['q_main1'])[np.array(dados_anteriores['flag']) == 0],
                 np.array(dados_anteriores['dP_bcs1'])[np.array(dados_anteriores['flag']) == 0], 'r.')
        plt.plot(np.array(dados_anteriores['q_main1'])[np.array(dados_anteriores['flag']) == 1],
                 np.array(dados_anteriores['dP_bcs1'])[np.array(dados_anteriores['flag']) == 1], 'b.')
        plt.plot([28.55, 20.77], [206.6, 58.07], 'k--', linewidth=3)
        plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
        plt.xlabel('$q_{main1}$ /(m$^3 \\cdot$ h$^{-1}$)', fontsize=15)
        plt.ylabel('$dP_{bcs1}$ /bar', fontsize=15)
        plt.grid()
        plt.show()

        plt.figure(dpi=250)
        plt.plot(np.array(dados_anteriores['q_main2'])[np.array(dados_anteriores['flag']) == 0],
                 np.array(dados_anteriores['dP_bcs2'])[np.array(dados_anteriores['flag']) == 0], 'r.')
        plt.plot(np.array(dados_anteriores['q_main2'])[np.array(dados_anteriores['flag']) == 1],
                 np.array(dados_anteriores['dP_bcs2'])[np.array(dados_anteriores['flag']) == 1], 'b.')
        plt.plot([28.55, 20.77], [206.6, 58.07], 'k--', linewidth=3)
        plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
        plt.xlabel('$q_{main2}$ /(m$^3 \\cdot$ h$^{-1}$)', fontsize=15)
        plt.ylabel('$dP_{bcs2}$ /bar', fontsize=15)
        plt.grid()
        plt.show()

        plt.figure(dpi=250)
        plt.plot(np.array(dados_anteriores['q_main3'])[np.array(dados_anteriores['flag']) == 0],
                 np.array(dados_anteriores['dP_bcs3'])[np.array(dados_anteriores['flag']) == 0], 'r.')
        plt.plot(np.array(dados_anteriores['q_main3'])[np.array(dados_anteriores['flag']) == 1],
                 np.array(dados_anteriores['dP_bcs3'])[np.array(dados_anteriores['flag']) == 1], 'b.')
        plt.plot([28.55, 20.77], [206.6, 58.07], 'k--', linewidth=3)
        plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
        plt.xlabel('$q_{main3}$ /(m$^3 \\cdot$ h$^{-1}$)', fontsize=15)
        plt.ylabel('$dP_{bcs3}$ /bar', fontsize=15)
        plt.grid()
        plt.show()

        plt.figure(dpi=250)
        plt.plot(np.array(dados_anteriores['q_main4'])[np.array(dados_anteriores['flag']) == 0],
                 np.array(dados_anteriores['dP_bcs4'])[np.array(dados_anteriores['flag']) == 0], 'r.')
        plt.plot(np.array(dados_anteriores['q_main4'])[np.array(dados_anteriores['flag']) == 1],
                 np.array(dados_anteriores['dP_bcs4'])[np.array(dados_anteriores['flag']) == 1], 'b.')
        plt.plot([28.55, 20.77], [206.6, 58.07], 'k--', linewidth=3)
        plt.plot([82.1, 53.6], [170.1, 44.7], 'k--', linewidth=3)
        plt.xlabel('$q_{main4}$ /(m$^3 \\cdot$ h$^{-1}$)', fontsize=15)
        plt.ylabel('$dP_{bcs4}$ /bar', fontsize=15)
        plt.grid()
        plt.show()


import pickle
import os
def ler():
    filename = 'rna_training.pkl'
    if os.path.getsize(filename) > 0:
        with open(filename, "rb") as f:
            unpickler = pickle.Unpickler(f)
            dados_anteriores = unpickler.load()
            return dados_anteriores
def juntar_escrever(n, dados_anteriores, dados_novos):
    if n == 1:
        dados_unidos = {chave: dados_anteriores[chave] + dados_novos[chave] for chave in dados_anteriores}
        filename = 'dados_training100k.pkl'
        import pickle
        with open(filename, "wb") as f:
            pickle.dump(dados_unidos, f)
            print("Dados Unidos")
            return dados_unidos

    elif n == 2:
        filename = 'dados_training100k.pkl'
        with open(filename, "wb") as f:
            pickle.dump(dados_novos, f)

from initialization_oil_production_basic import *
