import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from optimize3_model import x0_caso_a,lbx_caso_a,ubx_caso_a, lbg_caso_a, ubg_caso_a, solver_caso_a
from Rede_Neural_restrições_padrão_2 import x0_caso_b,lbx_caso_b,ubx_caso_b,lbg_caso_b, ubg_caso_b, solver_caso_b
from Rede_Neural_restrições_padrão import x0_caso_c,lbx_caso_c,ubx_caso_c, lbg_caso_c, ubg_caso_c, solver_caso_c

# Substitui todos os prints por gravação em arquivo
output_file = "histogram_data.txt"
with open(output_file, "w", encoding="utf-8") as f:

    def log(msg=""):
        f.write(str(msg) + "\n")

    def benchmark_otimizacoes_solver(n=100):
        caso_a = []
        caso_b = []
        caso_c = []

        for i in range(n):
            print(f"Execução {i + 1}/{n}", end="\r")
            # Caso A
            sol = solver_caso_a(x0=x0_caso_a, lbx=lbx_caso_a, ubx=ubx_caso_a, lbg=lbg_caso_a, ubg=ubg_caso_a)
            solver_stats = solver_caso_a.stats()
            tempo_caso_a = solver_stats['t_proc_total']
            caso_a.append(tempo_caso_a)
            # Caso B
            sol = solver_caso_b(x0=x0_caso_b, lbx=lbx_caso_b, ubx=ubx_caso_b, lbg=lbg_caso_b, ubg=ubg_caso_b)
            solver_stats = solver_caso_b.stats()
            tempo_caso_b = solver_stats['t_proc_total']
            caso_b.append(tempo_caso_b)
            # Caso C
            sol = solver_caso_c(x0=x0_caso_c, lbx=lbx_caso_c, ubx=ubx_caso_c, lbg=lbg_caso_c, ubg=ubg_caso_c)
            solver_stats = solver_caso_c.stats()
            tempo_caso_c = solver_stats['t_proc_total']
            caso_c.append(tempo_caso_c)

        # Estatísticas
        media_caso_a = np.mean(caso_a)
        desvio_caso_a = np.std(caso_a)
        media_caso_b = np.mean(caso_b)
        desvio_caso_b = np.std(caso_b)
        media_caso_c = np.mean(caso_c)
        desvio_caso_c = np.std(caso_c)

        log("========== ESTATÍSTICAS DE TEMPO (IPOPT) ==========")
        log(f"CASO A    -> Média: {media_caso_a:.4f} s | Desvio padrão: {desvio_caso_a:.4f} s")
        log(f"CASO B    -> Média: {media_caso_b:.4f} s | Desvio padrão: {desvio_caso_b:.4f} s")
        log(f"CASO C    -> Média: {media_caso_c:.4f} s | Desvio padrão: {desvio_caso_c:.4f} s")

        print("========== ESTATÍSTICAS DE TEMPO (IPOPT) ==========")
        print(f"CASO A  -> Média: {media_caso_a:.4f} s | Desvio padrão: {desvio_caso_a:.4f} s")
        print(f"CASO B   -> Média: {media_caso_b:.4f} s | Desvio padrão: {desvio_caso_b:.4f} s")
        print(f"CASO C   -> Média: {media_caso_c:.4f} s | Desvio padrão: {desvio_caso_c:.4f} s")
        log(f"Caso A: {caso_a}")
        log(f"Caso B: {caso_b}")
        log(f"Caso C: {caso_c}")

        # Plot
        plt.figure(figsize=(10, 6))
        plt.hist(caso_a, bins=20, alpha=0.7, label="Caso A", edgecolor='black')
        plt.hist(caso_b, bins=20, alpha=0.7, label="Caso B", edgecolor='black')
        plt.hist(caso_c, bins=20, alpha=0.7, label="Caso C", edgecolor='black')
        plt.xlabel("Tempo do Solver IPOPT / s", fontsize=20)
        plt.ylabel("Frequência", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        ax = plt.gca()
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))  # ativa notação para valores fora desse intervalo
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.get_offset_text().set_fontsize(20)
        plt.tight_layout()
        plt.savefig("comparacao_tempos_solver.png")  # <-- SALVA o plot
        plt.close()
        plt.show()


    # Executar benchmark com 100 execuções (ou o valor que quiser)
    benchmark_otimizacoes_solver(n=1000)