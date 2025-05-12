# -*- coding: utf-8 -*-

from numpy import pi

class Manifold:
    """
    Class to describe the manifold model
    Adapted from:
    Rasmus (2011) - Automatic Start-up and Control of Artificially Lifted Wells
    """

    def __init__(self, pipe, booster, q_wv, avrWC, wells):
        """
        Manifold setup

        :param pipe: pipe class of the manifold [pipe class]
        :param booster: pump class of the manifold booster pump [pump class]
        :param q_wv: [%] Initial water cut
        :param avrWC: [%] Initial, average water cut
        :param wells: list containing the wells objects [class] connect to the manifold
        """
        self.wells = wells
        self.pipe = pipe
        self.booster = booster

        # not in use
        self.q_wv = q_wv
        self.avrWC = avrWC

    def model(self, t, x, z, u):
        """
        Manifold complete model, including the associated wells
        :param t: time
        :param x: list with the states
        [P_manifold, flow through the transportation line (q_tr), P_fbhp_1, P_choke_1, q_average_1 ...
        P_fbhp_n, P_choke_n, q_average_n]
        :param z: list with algebraic variables [P_intake_1, P_discharge_1, ... P_intake_n, P_discharge_n]
        :param u: inputs [booster pump frequency, P_topside, f_ESP_1, choke opening 1, ... f_ESP_n, choke opening n]
        :return: 2 manifold ODE, n*3 well ODE, n*2 well algebraic equations
        """

        # inputs
        f_BP = u[0]
        p_topside = u[1]

        f_ESP = []
        alpha = []
        cont = 1
        for i in enumerate(self.wells):
            f_ESP.append(u[cont + 1])
            alpha.append(u[cont + 2])
            cont = cont + 2

        # States
        p_man = x[0]  # [Pa] manifold pressure
        q_tr = x[1]  # [m^3/s] Flow through the transportation line

        P_fbhp = []
        P_choke = []
        P_intake = []
        P_discharge = []
        q_mean = []
        q_choke = []
        dot_x = []
        g_z = []
        qin_man = 0

        contx = 1
        contz = -1
        for i, well in enumerate(self.wells):
            P_fbhp.append(x[contx + 1])
            P_choke.append(x[contx + 2])
            q_mean.append(x[contx + 3])
            P_intake.append(z[contz + 1])
            P_discharge.append(z[contz + 2])

            # for each well
            q_choke.append(well.flow_rate_choke(P_choke[i] * 1e5, p_man * 1e5, alpha[i]))

            qin_man = qin_man + q_choke[i]

            dot_p_fbhp, dot_p_choke, dot_q_ave, g_p_in, g_dp_bcs = well.model(t,
                                                                             [x[contx + 1], x[contx + 2], x[contx + 3]],
                                                                             [z[contz + 1], z[contz + 2]],
                                                                             [f_ESP[i], alpha[i]], p_man)

            contx = contx + 3
            contz = contz + 2

            dot_x += [dot_p_fbhp, dot_p_choke, dot_q_ave]
            g_z += [g_p_in, g_dp_bcs]

        fric_manifold = self.friction(q_tr / 3600, self.pipe)

        dp_boost = self.booster.head_fun(f_BP, q_tr / 3600) * self.pipe.rho * 9.81

        height = self.pipe.h * self.pipe.rho * 9.81

        dot_p_man = (self.pipe.beta / self.pipe.V) * (qin_man - q_tr / 3600) / 1e5
        dot_qtr = (self.pipe.A / self.pipe.rho / self.pipe.L) * (p_man * 1e5 - p_topside - height - fric_manifold + dp_boost) * 3600

        return [dot_p_man, dot_qtr] + dot_x + g_z

    def friction(self, q, pipe):
        """
        Friction model of manifold
        :param q: flow [m^3/s]
        :param pipe: pipe class
        :return:
        """
        Re = (2 * pipe.rho * q) / (pipe.r * pi * pipe.mu)
        fric = 0.36 * Re ** (-0.25)
        return (pipe.B0 + pipe.B1 * fric) * q ** 2 * pipe.rho / 2.
