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
def get_states_optimized_helper(seqs):
    state = basis(2, 0)
    dd = identity(2)
    
    state_list = [state]
    
    for i, seq in enumerate(seqs):
        state = seq * state
        if i % 2 == 0:
            state_list += [state]
        dd = seq * dd
        
    return state_list, dd
    
#@jit
def get_states_optimized(N, ev: Evolution, tau):
    seqs = [ev.seq(i * tau, (i+1) * tau) for i in range(N)]
    return get_states_optimized_helper(seqs)

def get_unitary_res(u, up):
    return u @ np.array(up).transpose().conj() # should approach id

def get_unitaries_diff_proj(u, up):
    '''get projection of difference between two unitaries '''

    unitary_res = get_unitary_res(u, up)
    return [(l, (unitary_res * b).trace()) for l, b in [('id', identity(2)), ('X', sigmax()), ('Y', sigmay()), ('Z', sigmaz())] ]
    
def get_unitaries_diff_fidelity(u, up):
    '''get projection of difference between two unitaries '''
    
    unitary_res = get_unitary_res(u, up)
    return (np.abs(np.trace(unitary_res))/2)**2
    
    
def compute_complete_unitary(ev: Evolution, tau, exact=False):
        
    print("Compute "+ str(ev.pulse) + " Series")
    #rn = np.linspace(100, 1000000, 6, dtype=int)
    rn = [100]
    print(rn, "\n")

    for N in rn:
        print("N=", N)
        _, DD = get_states_optimized(N, ev, tau )
        
        print("true mat: \n", DD)
        print("\n")
        
        fo = ev.first_order_exp(N * tau, 0)
        
        print("1st ord approx (slow varying):\n", fo)
        fid = get_unitaries_diff_fidelity(DD, fo)
        print("fidelity: ", fid)
        proj = get_unitaries_diff_proj(DD, fo)
        print("projection traces: ", proj)
        print("\n")
        
        if exact:
            fo_ex = ev.first_order_exp_exact(tau, N * tau, 0)

            print("1st ord approx:\n", fo_ex)
            get_unitaries_diff_fidelity(DD, fo_ex)
            get_unitaries_diff_proj(DD, fo_ex)
            print("\n")