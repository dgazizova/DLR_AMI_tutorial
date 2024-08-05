import numpy as np
from scipy.integrate import quad
from pydlr import dlr

def dyson(G, SE):
    """
    Calculation bold GF in Dyson equation
    :param G: GF function usually bare
    :param SE: Self energy
    :return: bold GF
    """
    res = G / (1.0 + 0.0*1j - G*SE)
    return res

class multiple_DLR:
    """
    Class that creates few different DLR representations
    """
    def __init__(self, n_dlr, beta, E_max, eps, delta_range: tuple):
        """
        :param n_dlr: number of DLR representations needed
        :param beta: beta temperature
        :param E_max: range for DLR
        :param eps: DLR eps
        :param delta_range: tuple of boarders for additional delta to the E_max
        """
        self.n_dlr = n_dlr
        self.beta = beta
        self.E_max = E_max
        self.eps = eps
        self.delta_range = delta_range

        self.d_list = []
        for n_dlr_ in range(self.n_dlr):
            delta = np.random.uniform(self.delta_range[0], self.delta_range[1])
            self.d_list.append(dlr(lamb=self.beta * (self.E_max + delta), eps=self.eps))

        self.w_list = []
        self.iw_q_list = []
        self.iw_q_re_list = []
        self.r_list = []
        self.r_list_un = []
        for d in self.d_list:
            self.w_list.append(d.dlrrf / beta)
            self.r_list.append(len(d.dlrrf))
            self.iw_q_list.append(d.get_matsubara_frequencies(beta))
            self.iw_q_re_list.append(np.imag(d.get_matsubara_frequencies(beta)))

    def get_ac_list(self, G_iwq_list):
        """
        Creates list of poles
        :param G_iwq_list: list of green functions should be length of n_dlr and every GF in its own DLR matsubara freq
        :return: return list of n_dlr number of poles
        """
        ac_list = []
        for G_iwq, d, w in zip(G_iwq_list, self.d_list, self.w_list):
            ac_list.append(myac(d.dlr_from_matsubara(G_iwq, self.beta) * (-1), w))
        return ac_list


    def get_ac(self, G_iwq, number_of_DLR):
        """
        Creates poles for exact nth number of Green function
        :param G_iwq: Green function in its own matsubara freq
        :param number_of_DLR: nth number of DLR to create the poles
        :return: poles of the GF
        """
        ac = myac(self.d_list[number_of_DLR].dlr_from_matsubara(G_iwq, self.beta) * (-1), self.w_list[number_of_DLR])
        return ac

class myac:
    def __init__(self, pole_weight, pole_location):
        self.pole_weight = pole_weight
        self.pole_location = pole_location


def get_GF_from_DLR_iw(iw, g_k, w_k):
    '''
    Function that recover Green function from pole representation
    :param iw: matsubara frequencies of Green Function
    :param g_k: pole weights
    :param w_k: pole locations
    :return: Green function in matsubara frequency, works same if using iw as real frequencies with small imaginary part
    '''
    res = 0
    for pole_weight, pole_location in zip(g_k, w_k):
        res = res + pole_weight/(iw - pole_location)
    return res

def get_G_iw_indv(iw, a):
    '''
    Green function through spectral function
    :param iw: one single matsubara frequency
    :param a: range for the bethe spectral function
    :return: Green function in one individual matsubara frequency
    '''
    I = lambda x: bethe_lattice_dos(x) / (iw - x)
    real_part = quad(lambda x: I(x).real, -a, a)[0]
    imag_part = quad(lambda x: I(x).imag, -a, a)[0]
    return real_part + 1j * imag_part

def get_G_iw(iw, a):
    '''
    Vectorized version of get_G_iw for array of matsubara frequencies
    :param iw: numpy array of matsubara frequencies
    :param a: range for the bethe spectral function
    :return: Green function in the array of matsubara frequencies
    '''
    return np.vectorize(get_G_iw_indv)(iw, a)

def matsubara(statistics: str, n, beta):
    '''
    Function to create matsubara frequencies
    :param statistics: "F" or "B" for Fermi or Bose statistics
    :param n: range of int n to create matsubara frequencies
    :param beta: temperature value
    :return imaginary part of matsubara frequencies
    '''
    omega = 0
    if statistics == "F":
        omega = (2*n + 1) * np.pi / beta
    if statistics == "B":
        omega = 2*n * np.pi / beta
    return omega

def get_n_from_matsubara(w, beta):
    '''Function to get number of matsubara frequency'''
    return np.real((w*beta / np.pi - 1) / 2)

def bethe_lattice_dos_indiv(x, t = 1.0):
    '''
    Intermediate function of bethe_lattice_dos(w, t) which only works for a single point.
    '''
    if np.abs(x) > 2 * t:
        return 0.0
    else:
        return 1.0 / (2.0 * np.pi * t**2.0) * np.sqrt(4 * t**2.0 - x**2.0)

def bethe_lattice_dos(w, t = 1.0):
    '''
    Density of states for the tight-binding model in Bethe lattice with interaction t.
    '''
    return np.vectorize(lambda x: bethe_lattice_dos_indiv(x, t))(w)


class SelfEnergy:

    def __init__(self, beta):
        self.beta = beta


    def get_fermi_bose(self, statistics: str, dispersion):
        '''
        Definition of the Fermi and Bose functions
        '''
        if statistics == "F":
            return 1 / (np.exp(dispersion * self.beta) + 1)
        elif statistics == "B":
            return 1 / (np.exp(dispersion * self.beta) - 1)
        else:
            return

    def get_self_energy(self, iomega, z1, z2, z3, A1=1, A2=1, A3=1, U=1):
        '''
        Analytical expression for the self-energy
        :param iomega: matsubara frequencies for calculation
        :param z1: poles locations of first Green function
        :param z2: poles locations of second Green function
        :param z3: poles locations of third Green function
        :param A1: poles weights of first Green function (can be spectral functions)
        :param A2: poles weights of second Green function (can be spectral functions)
        :param A3: poles weights of third Green function (can be spectral functions)
        :param U: U values for self energy calculation (defaults to 1)
        :return: second order self energy calculated as a function of matsubara frequencies
        '''
        num = ((self.get_fermi_bose("F", z1) - self.get_fermi_bose("F", z2))
               * (self.get_fermi_bose("B", z2 - z1) + self.get_fermi_bose("F", -z3)))
        den = iomega + z1 - z2 - z3
        prefactor = A1 * A2 * A3
        return prefactor * num / den * U**2

    def get_self_energy_from_poles(self, ac_list: list, iw, U=1):
        '''
        Function to calculate self energy from poles
        :param ac_list: list of 3 ac that has pole weight and location
        :param iw: matsubara frequencies
        :param U: U values for self energy
        :return self energy calculated as a function of matsubara frequencies (can be replaced with real freq with
        small imaginary part)
        '''
        SE_dlr = 0
        for A1, z1 in zip(ac_list[0].pole_weight, ac_list[0].pole_location):
            for A2, z2 in zip(ac_list[1].pole_weight, ac_list[1].pole_location):
                for A3, z3 in zip(ac_list[2].pole_weight, ac_list[2].pole_location):
                    local = self.get_self_energy(iw, z1, z2, z3, A1=A1, A2=A2, A3=A3, U=U)
                    SE_dlr = SE_dlr + local
        return SE_dlr


    def SE_full_spectral(self, n, a, iw):
        '''
        Calculation of full spectral Self energy
        :param n: number of real frequencies for integration
        :param a: range where spectral energy is defined in bethe lattice
        :param iw: matsubara frequencies
        :return: second order self energy results for spectral functions
        '''
        x = []
        # create random points of shape n * 3
        for i in range(3):
            local = np.random.uniform(-a, a, n) + 0.0 * 1j
            x.append(local)
        SE_res = 0
        for x1, x2, x3 in zip(x[0], x[1], x[2]):
            A1, A2, A3 = bethe_lattice_dos(w=x1), bethe_lattice_dos(w=x2), bethe_lattice_dos(w=x3)
            local = self.get_self_energy(iw, x1, x2, x3, A1, A2, A3)
            SE_res = SE_res + local
        SE_res = SE_res / n * (2 * a) ** 3
        return SE_res