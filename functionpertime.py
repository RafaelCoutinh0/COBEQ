import pandas as pd
import matplotlib as plt

dados = {
    "Parâmetro": [
        "Iterações totais",
        "Objetivo final (não escalado)",
        "Avaliações da função objetivo",
        "Avaliações do gradiente",
        "Avaliações da Hessiana",
        "Avaliações das restrições",
        "Avaliações do Jacobiano",
        "Número de variáveis",
        "Desigualdades",
        "Igualdades",
        "Tempo total (ms)",
        "Precisão (NLP error)",
        "Tempo nlp_f médio (μs)",
        "Tempo nlp_g médio (μs)",
        "Tempo grad_f médio (μs)",
        "Tempo hess_l médio (μs)",
        "Tempo jac_g médio (μs)",
        "Tempo médio por iteração"
    ],
    "Caso A": [
        16,
        round(7.1615467e+05, 2),
        23,
        17,
        16,
        23,
        17,
        31,
        12,
        22,
        round(12.24, 2),
        "{:.2e}".format(7.51e-08),
        round(1.65, 2),
        round(20.65, 2),
        round(2.89, 2),
        round(69.87, 2),
        round(53.50, 2),
        round(12.24/16,2)
    ],
    "Caso B": [
        14,
        round(7.4158553e+05, 2),
        16,
        15,
        14,
        16,
        15,
        24,
        12,
        14,
        round(32.96, 2),
        "{:.2e}".format(2.75e-07),
        round(2.81, 2),
        round(14.19, 2),
        round(6.44, 2),
        round(105.07, 2),
        round(61.12, 2),
        round(32.96/14, 2)
    ],
    "Caso C": [
        23,
        round(7.4158553e+05, 2),
        81,
        24,
        23,
        81,
        24,
        10,
        12,
        0,
        round(17.15, 2),
        "{:.2e}".format(2.29e-05),
        round(4.02, 2),
        round(5.38, 2),
        round(11.24, 2),
        round(74.17, 2),
        round(31.40, 2),
        round(17.15/23, 2)
    ]
}

df = pd.DataFrame(dados)

# Melhor visualização com alinhamento
print("\n📋 Comparação de Desempenho e Tempos:\n")
print(df.to_string(index=False))


import matplotlib.pyplot as plt

# Dados atualizados e convertidos para valores positivos
caso_a = [
    783879.78, 783891.59, 716132.73, 716140.26, 716150.23, 716150.78, 716150.84,
    716154.10, 716154.61, 716154.61, 716154.64, 716154.67, 716154.67, 716154.67,
    716154.67, 716154.67, 716154.67
]


caso_b = [
    771854.98, 772021.38, 771939.86, 741220.18, 741205.13, 741403.47,
    741557.40, 741562.04, 741580.21, 741580.15, 741585.37, 741585.50,
    741585.53, 741585.53, 741585.53]



caso_c = [
    790923.01, 789762.06, 783689.62, 782106.26, 779607.49, 778277.26,
    777719.28, 777560.95, 777472.01, 779024.36, 733979.03, 737274.16,
    740293.66, 741378.88, 743869.67, 741357.43, 741542.19, 741569.43,
    741582.22, 741585.38, 741585.38, 741585.43, 741585.53, 741585.53
]

# Plotando a convergência
plt.figure(figsize=(10, 6))
plt.plot(caso_a, 'o-', label='Caso A', linewidth=4, markersize=10)
plt.plot(caso_b, 's--', label='Caso B', linewidth=4, markersize=10)
plt.plot(caso_c, marker='^', linestyle='-', label='Caso C', linewidth=4, markersize=10, markeredgewidth=2)

# Aumentando o tamanho dos rótulos e da legenda
plt.xlabel('Iteração', fontsize=20)
plt.ylabel('Função Objetivo', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.show()




