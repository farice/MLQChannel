from enum import Enum
import types
from qutip import *
from scipy import integrate
import numpy as np

class P(Enum):
    X = 1
    Y = 2
    Z = 3
    I = 4
    

class Pulse:
    def __init__(self, t: P):
        self.t = t
        self.op = sigmax()
        if t is  P.Y:
            self.op = sigmay()
        elif t is P.Z:
            self.op = sigmaz()
        elif t is P.I:
            self.op = identity(2)
            
    def __str__(self):
        return str(self.t)
    
class Noise:
    def __init__(self, f: types.LambdaType, g: types.LambdaType, h: types.LambdaType=None):
        self.X_fn = f
        self.Z_fn = g
        self.Y_fn = h

class Evolution:
    def __init__(self, p: Pulse, n: Noise, eps=1e-3):
        self.pulse = p
        self.noise = n
        
        self.eps = eps #Has to be less 1/cutoff_freq
        
        self.first_order_fn = n.X_fn
        self.opp_fn = n.Z_fn
        self.opp_pulse = sigmaz()
        if p.t is P.Z:
            self.first_order_fn = n.Z_fn
            self.opp_fn = n.X_fn
            self.opp_pulse = sigmax()
    
    def u(self, t_1, t_2):
        
        M = int((t_2-t_1)/self.eps )
        series = identity(2)
        
        _incremental_time_op = lambda t: (
            -1.j* self.eps * (self.noise.X_fn(t)*sigmax() + self.noise.Z_fn(t) * sigmaz())
        ).expm()

        for i in range(M+1):
            series *= _incremental_time_op(t_1 + i * self.eps)

        return series

    def seq(self, t_1, t_2):
        return self.pulse.op * self.u(t_1, t_2)
    
    def first_order_exp(self, t_final, t_0=0): 
    # find the first order exponential integral of a give Pauli operator
    # valid only if opp_fn(t) is slow-varying relative to the pulse frequency
        
        # TODO:- why is this factor of 10 needed???
        return (  -10j * integrate.quad(self.first_order_fn, t_0, t_final)[0] * self.pulse.op).expm()
    
    def first_order_exp_exact(self, tau, t_final, t_0=0):
        opp_fn_modulated = lambda t: self.opp_fn(t) if int(t - t_0) % 2 == 0 else -1 * self.opp_fn(t)
        return (
                -10j * ( integrate.quad(self.first_order_fn, t_0, t_final)[0] * self.pulse.op +
                        integrate.quad(opp_fn_modulated, t_0, t_final)[0] * self.opp_pulse
                      )
                    ).expm()

    #@jit
    def get_states_optimized(self, N, tau):
        state = basis(2, 0)
        dd = identity(2)

        state_list = [state]

        for i, seq in enumerate([self.seq(i * tau, (i+1) * tau) for i in range(N)]):
            state = seq * state
            if i % 2 == 0:
                state_list += [state]
            dd = seq * dd

        return state_list, dd


class DistanceStats:
    @staticmethod
    def unitary_res(u, up):
        return u @ np.array(up).transpose().conj()

    @staticmethod
    def unitary_diff_proj(u, up):
        '''get projection of difference between two unitaries '''

        unitary_res = DistanceStats.unitary_res(u, up)
        return [(l, (unitary_res * b).trace()) for l, b in [('id', identity(2)), ('X', sigmax()), ('Y', sigmay()), ('Z', sigmaz())] ]

    @staticmethod
    def unitary_diff_fidelity(u, up):
        '''get projection of difference between two unitaries '''

        unitary_res = DistanceStats.unitary_res(u, up)
        return (np.abs(np.trace(unitary_res))/2)**2

    @staticmethod
    def unitary_diff_stats(u, up):
        return DistanceStats.unitary_diff_fidelity(u, up), DistanceStats.unitary_diff_proj(u, up)
