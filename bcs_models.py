# -*- coding: utf-8 -*-
"""
Editor Spyder

Este é um arquivo de script temporário.
"""

from casadi import sign, fabs
from numpy import pi


class Pipe:
    def __init__(self, d, L, h, beta, rho, mu, B0, B1):
        """
        Define pipe parameters

        :param d: pipe's axial section diameter [m]
        :param L: pipe's length [m]
        :param h: pipe's height [m]
        :param beta: fluid's dP/drho [Pa m^3 /kg]
        :param rho: fluid density [kg/m^3]
        :param mu: fluid's viscosity [Pa s]
        :param B0: friction parameter 1
        :param B1: friction parameter 2
        """
        self.d = d
        self.r = d / 2  # [m] pipe's axial section radius
        self.L = L
        self.h = h
        self.beta = beta
        self.rho = rho
        self.V = pi * self.r ** 2. * L  # [m^3] pipe's volume
        self.A = pi * self.r ** 2.  # [m^2] pipe's axial section area
        self.mu = mu
        self.B0 = B0
        self.B1 = B1


class Choke:

    def __init__(self, k_choke, choke_fun):
        """
        Choke setup
        :param k_choke: valve constant [m^3/s/Pa^0.5]
        :param choke_fun: [adm -> adm] valve opening function in CasADi function format
        """
        self.k_choke = k_choke
        self.choke_fun = choke_fun

    def model(self, opening):
        """
        Multiplication of k_choke and choke characteristic as function of valve opening. Equation needed to evaluate
        the flow rate through the valve.
        :param opening: valve opening (0 to 1)
        :return: k_choke*choke characteristic
        """
        return self.k_choke * self.choke_fun(opening)


class Pump:

    def __init__(self, head_fun, eff_fun, pot_fun):
        """
        Pump setup
        :param head_fun: head of BCS in function of frequency and volumn flow in CasADi function format [Hz, m^3/s -> m]
        :param eff_fun: efficiency of BCS in function of frequency and volumn flow in CasADi function format [Hz, m^3/s -> adm]
        :param pot_fun: power of BCS in function of frequency and volumn flow in CasADi function format [Hz, m^3/s -> W]
        """
        self.head_fun = head_fun
        self.eff_fun = eff_fun
        self.pot_fun = pot_fun


class Well:

    def __init__(self, pipe1, pipe2, bcs, choke, PI, P_reservoir):
        """
        Well setup
        :param pipe1: pipe class of the first section of well (before BCS)
        :param pipe2: pipe class of the second section of well (after BCS)
        :param bcs: pump class of the bcs of the well
        :param choke: choke class of the choke valve of well
        :param PI: production index of well reservoir [m^3/s/Pa]
        :param P_reservoir: pressure of the reservoir in the well [Pa]
        """
        self.pipe1 = pipe1  # [pipe class]
        self.pipe2 = pipe2  # [pipe class]
        self.bcs = bcs  # [pump class]
        self.choke = choke  # [choke class]
        self.PI = PI
        self.P_reservoir = P_reservoir

    def model(self, t, x, z, u, bound_conditions):
        """
        Differential-algebraic system to describe the Well dynamics

        :param t: time [s]
        :param x: state vector of differential ordinary functions (differential variables)
        [P_fbhp, P_choke, q_average]
        :param z: algebraic variables - [P_intake_BCS, P_discharge_BCS]
        :param u: inputs [f_BCS, choke_opening]
        :param bound_conditions: bound conditions - P_manifold
        :return: 3 ODE, 2 Algebraic
        """
        P_fbhp = x[0]
        P_choke = x[1]
        q_average = x[2]
        P_intake = z[0]
        bcs_dP = z[1]

        frequency = u[0]
        opening = u[1]

        P_manifold = bound_conditions

        q_reservoir = self.PI * (self.P_reservoir - P_fbhp * 1e5)

        friction_1 = self.friction(q_average / 3600, self.pipe1)
        friction_2 = self.friction(q_average / 3600, self.pipe2)

        q_choke = self.flow_rate_choke(P_choke * 1e5, P_manifold * 1e5, opening)

        l_bar, r_bar, A_bar, rho_bar = self.average_properties()

        dp = self.bcs.head_fun(frequency, q_average / 3600) * 9.81 * rho_bar

        height_1 = self.pipe1.h * self.pipe1.rho * 9.81

        height_2 = self.pipe2.h * self.pipe2.rho * 9.81

        # ODE's

        dpfbdt = self.pipe1.beta / self.pipe1.V * (q_reservoir - q_average / 3600) / 1e5

        dpchokedt = self.pipe2.beta / self.pipe2.V * (q_average / 3600 - q_choke) / 1e5

        dqaveragedt = A_bar / rho_bar / l_bar * (P_fbhp * 1e5 - P_choke * 1e5 - friction_1 - friction_2 - height_1 - height_2 + dp) * 3600

        # ALGEBRAIC
        
        P_discharge = P_choke * 1e5 + friction_2 + height_2
        
        g_pin = P_fbhp - (friction_1 + height_1) / 1e5 - P_intake

        g_bcs = bcs_dP - (P_discharge / 1e5 - P_intake)

        return dpfbdt, dpchokedt, dqaveragedt, g_pin, g_bcs

    def friction(self, q, pipe):
        """
        Evaluation of pipe friction

        :param q: flow [m^3/s]
        :param pipe: pipe class
        :return: pressure variation due to friction in pipe
        """
        Re = (2 * pipe.rho * q) / (pipe.r * pi * pipe.mu)
        fric = 0.36 * Re ** (-0.25)
        return (pipe.B0 + pipe.B1 * fric) * q ** 2 * pipe.rho / 2.

    def average_properties(self):
        """
        Average properties of Well, evaluated from pipes
        :return: average length, average radius, average cross-section area, average density
        """
        l_bar = (self.pipe1.L + self.pipe2.L) / 2.
        r_bar = (self.pipe1.r * self.pipe1.L + self.pipe2.r * self.pipe2.L) * 2. / l_bar
        A_bar = pi * r_bar ** 2.
        rho_bar = (self.pipe1.rho * self.pipe1.V + self.pipe2.rho * self.pipe2.V) / (self.pipe1.V + self.pipe2.V)

        return l_bar, r_bar, A_bar, rho_bar

    def flow_rate_choke(self, p_in, p_out, opening):
        """
        Flow rate through choke
        :param p_in: pressure in valve intake [Pa]
        :param p_out: pressure in valve discharge [Pa]
        :param opening: choke opening (0 to 1)
        :return: flow [m^3/s]
        """
        choke_factor = self.choke.model(opening)
        return sign(p_in - p_out) * choke_factor * (fabs(p_in - p_out)) ** 0.5
