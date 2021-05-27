
import scipy as sp
import scipy.stats
def compute_multi_info(sampling_p_marg,jt):
    fact = 0
    N = sampling_p_marg.size
    for r in range(N):
        fact = fact + sp.stats.entropy([sampling_p_marg[r],1-sampling_p_marg[r]])
    
    multi_info = (fact - sp.stats.entropy(jt))/fact
    return multi_info#,fact,sp.stats.entropy(jt)
