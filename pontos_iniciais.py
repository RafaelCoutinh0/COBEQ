import numpy as np

p_topo = 80000.0
f_bcs_1 = 60
f_bcs_2 = 60
f_bcs_3 = 60
f_bcs_4 = 60
f_booster = 65
valve = 1

u0_ref = np.array([
    f_booster,    # booster
    p_topo,       # pressão no topo
    f_bcs_1,      # BCS 1
    valve,        # válvula 1
    f_bcs_2,      # BCS 2
    valve,        # válvula 2
    f_bcs_3,      # BCS 3
    valve,        # válvula 3
    f_bcs_4,      # BCS 4
    valve         # válvula 4
])

u0_rna = np.array([
    p_topo,       # pressão no topo
    valve,        # válvula 1
    valve,        # válvula 2
    valve,        # válvula 3
    valve,        # válvula 4
    f_bcs_1,      # BCS 1
    f_bcs_2,      # BCS 2
    f_bcs_3,      # BCS 3
    f_bcs_4,      # BCS 4
    f_booster     # booster
])

