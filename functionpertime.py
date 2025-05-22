import matplotlib.pyplot as plt

# Dados atualizados e convertidos para valores positivos
caso_a = [
    5.0660185e+05,
    5.1989150e+05,
    5.8059968e+05,
    6.7172403e+05,
    7.0579546e+05,
    7.1568948e+05,
    7.1603367e+05,
    7.1616144e+05,
    7.1615946e+05,
    7.1613704e+05,
    7.1613705e+05,
    7.1615091e+05,
    7.1615403e+05,
    7.1615410e+05,
    7.1615462e+05,
    7.1615463e+05,
    7.1615467e+05,
    7.1615467e+05,
    7.1615467e+05,
    7.1615467e+05,
    7.1615467e+05,
    7.1615467e+05,
    7.1615467e+05
]



caso_c = [
    5.1292344e+05,
    5.1267765e+05,
    5.4651045e+05,
    7.0462188e+05,
    7.3465400e+05,
    7.3850027e+05,
    7.3579010e+05,
    7.4044368e+05,
    7.4077583e+05,
    7.4114218e+05,
    7.4140363e+05,
    7.4145699e+05,
    7.4156724e+05,
    7.4157132e+05,
    7.4158386e+05,
    7.4158536e+05,
    7.4158553e+05,
    7.4158553e+05,
    7.4158553e+05
]




caso_b = [
    5.1292344e+05,
    5.1324398e+05,
    5.2684569e+05,
    5.7834506e+05,
    6.7601456e+05,
    7.1912856e+05,
    7.4006327e+05,
    7.4138631e+05,
    7.4118441e+05,
    7.4123609e+05,
    7.4130725e+05,
    7.4155632e+05,
    7.4156232e+05,
    7.4157920e+05,
    7.4158487e+05,
    7.4158553e+05,
    7.4158550e+05,
    7.4158553e+05,
    7.4158553e+05
]

print(len(caso_a))
print(len(caso_b))
print(len(caso_c))
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Plotando a convergência
plt.figure(figsize=(10, 6))
plt.plot(caso_a, 'o-', label='Caso A', linewidth=4, markersize=10)
plt.plot(caso_b, 's--', label='Caso B', linewidth=4, markersize=10)
plt.plot(caso_c, marker='^', linestyle='-', label='Caso C', linewidth=4, markersize=10, markeredgewidth=2)

# Aumentando o tamanho dos rótulos e da legenda
plt.xlabel('Iteração', fontsize=20)
plt.ylabel('Função Objetivo', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)

# Aplicando notação científica no eixo Y
ax = plt.gca()
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-3, 3))  # ativa notação para valores fora desse intervalo
ax.yaxis.set_major_formatter(formatter)
ax.yaxis.get_offset_text().set_fontsize(20)
plt.grid(True)
plt.tight_layout()
plt.show()





