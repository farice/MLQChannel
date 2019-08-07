from .core import Evolution, DistanceStats


def compute_complete_unitary(ev: Evolution, tau, rn, exact=False):
    def outpt(DD, fo, txt):
        print(txt, fo)
        fid, proj = DistanceStats.unitary_diff_stats(DD, fo)
        print("fidelity: ", fid)
        print("projection traces: ", proj)
        print("\n")
        
    print(str(ev.pulse) + " Series")
    
    print(rn, "\n")

    for N in rn:
        print("N=", N)
        _, DD = ev.get_states_optimized(N, tau )
        
        print("true mat: \n", DD)
        print("\n")
        
        fo = ev.first_order_exp(N * tau, 0)
        outpt(DD, fo, "1st ord approx (slow varying):")
        
        if exact:
            fo_ex = ev.first_order_exp_exact(tau, N * tau, 0)
            outpt(DD, fo_ex, "1st ord approx (slow varying):")