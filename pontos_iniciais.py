import numpy as np

p_topo = 80000.0
f_bcs_1 = 44.210034809928004
f_bcs_2 = 59.898654215689234
f_bcs_3 = 46.962365911971986
f_bcs_4 = 45.54753856315394
f_booster = 36.79580857861887
valve_1 = 0.8656801935771609
valve_2 = 0.6308261391068003
valve_3 = 0.6985952136651123
valve_4 = 0.4391766674050762

u0_ref = np.array([
    f_booster,    # booster
    p_topo,       # pressão no topo
    f_bcs_1,      # BCS 1
    valve_1,      # válvula 1
    f_bcs_2,      # BCS 2
    valve_2,      # válvula 2
    f_bcs_3,      # BCS 3
    valve_3,      # válvula 3
    f_bcs_4,      # BCS 4
    valve_4       # válvula 4
])

u0_rna = np.array([
    p_topo,       # pressão no topo
    valve_1,      # válvula 1
    valve_2,      # válvula 2
    valve_3,      # válvula 3
    valve_4,      # válvula 4
    f_bcs_1,      # BCS 1
    f_bcs_2,      # BCS 2
    f_bcs_3,      # BCS 3
    f_bcs_4,      # BCS 4
    f_booster     # booster
])
