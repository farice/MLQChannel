from .core import Evolution, DistanceStats


def compute_complete_unitary(ev: Evolution, tau, rn, verbose=False, exact=False):
    def outpt(DD, fo, txt):
        if verbose:
            print(txt, fo)
        fid, proj = DistanceStats.unitary_diff_stats(DD, fo)
        
        if verbose:
            print("fidelity: ", fid)
            print("projection traces: ", proj)
            print("\n")
        
        return fid, proj
        
    if verbose:
        print(str(ev.pulse) + " Series")
        print(rn, "\n")

    fids, projs = [], []
    for N in rn:
        if verbose: print("N=", N)
        _, DD = ev.get_states_optimized(N, tau )
        
        if verbose:
            print("true mat: \n", DD)
            print("\n")
        
        fo = ev.first_order_exp(N * tau, 0)
        fid, proj = outpt(DD, fo, "1st ord approx (slow varying):")
        fids.append(fid)
        fids.append(proj)
        
        if exact:
            fo_ex = ev.first_order_exp_exact(tau, N * tau, 0)
            outpt(DD, fo_ex, "1st ord approx (slow varying):")
            
    return fids, projs