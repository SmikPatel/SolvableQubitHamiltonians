import numpy as np
import networkx as nx
from openfermion import (
    QubitOperator as Q,
    commutator,
    anticommutator,
    get_sparse_operator as gso,
    get_ground_state as ggs,
    variance,
    bravyi_kitaev
)
from utils_basic import (
    is_commuting,
    is_anticommuting,
    copy_hamiltonian
)
from utils_ff import (
    obtain_ac_graph,
    is_line_graph
)

import pickle
import sys

N_QUBITS = {
    'h2'   : 4,
    'lih'  : 12,
    'beh2' : 14,
    'h2o'  : 14,
    'nh3'  : 16
}

#
#    first set of functions concerns creation of electronic Hamiltonian
#

def abs_of_dict_value(x):
    return np.abs(x[1])

def load_hamiltonian(moltag):
    filename = f'ham_lib/{moltag}_fer.bin'
    with open(filename, 'rb') as f:
        Hfer = pickle.load(f)
    Hqub = bravyi_kitaev(Hfer)
    Hqub -= Hqub.constant
    Hqub.compress()
    Hqub.terms = dict(sorted(Hqub.terms.items(), key=abs_of_dict_value, reverse=True))
    return Hqub

#
#    second set of functions concerns is_X for X in FC, AC, TWC_AC
#

def is_empty_graph(G):
    return len(G.edges) == 0

def is_complete_graph(G):
    return len(G.edges) == len(G.nodes) * (len(G.nodes) - 1) // 2

def is_ccs_are_cliques(G):

    cc_nodes  = nx.connected_components(G)
    cc_graphs = [G.subgraph(nodes) for nodes in cc_nodes]

    for graph in cc_graphs:
        if not is_complete_graph(graph):
            return False
    return True

def is_fc_hamiltonian(H):
    AC = obtain_ac_graph(H)
    return is_empty_graph(AC)

def is_ac_hamiltonian(H):
    AC = obtain_ac_graph(H)
    return is_complete_graph(AC)

def is_twc_ac_hamiltonian(H):
    AC = obtain_ac_graph(H)
    return is_ccs_are_cliques(AC)

#
#    third set of functions concerns is_X for X in ff, twc_ff
#

def is_ccs_are_line_graphs(G):

    cc_nodes  = nx.connected_components(G)
    cc_graphs = [G.subgraph(nodes) for nodes in cc_nodes]

    for graph in cc_graphs:
        if not is_line_graph(graph):
            return False
    return True

def is_ff_hamiltonian(H):
    AC = obtain_ac_graph(H)
    return is_line_graph(AC)

def is_twc_ff_hamiltonian(H):
    AC = obtain_ac_graph(H)
    return is_ccs_are_line_graphs(AC)

#
#    fourth set of functions concerns obtaining Hamiltonian with TFQG of H
#

def obtain_adjacency_list(H):
    
    adjacency_list = dict()
    for op in H.terms.keys():

        op_adjacencies = Q().zero()
        
        for t, s in H.terms.items():
            if is_anticommuting(op, t):
                op_adjacencies += s * Q(t)

        adjacency_list[op] = op_adjacencies

    return adjacency_list

def remove_fc_operators(H, adjacency_list):
    
    for op in H.terms.keys():
        if adjacency_list[op] == Q().zero():
            del adjacency_list[op]

def obtain_twin_equivalence_classes(H, adjacency_list):
    
    equiv_classes = []
    while adjacency_list != dict():

        current_class = Q().zero()

        initial_term        = list(adjacency_list.keys())[0]
        initial_coef        = H.terms[initial_term]
        initial_adjacencies = adjacency_list[initial_term]
        current_class      += initial_coef * Q(initial_term)
        
        for term, term_adjacencies in adjacency_list.items():
            if (term != initial_term) and (term_adjacencies  == initial_adjacencies):
                current_class += H.terms[term] * Q(term)

        equiv_classes.append(current_class)
        for term in current_class.terms.keys():
            del adjacency_list[term]
    
    return equiv_classes

def obtain_sampling_operator(H):
    adj     = obtain_adjacency_list(H)
    remove_fc_operators(H, adj)
    classes = obtain_twin_equivalence_classes(H, adj)
    
    sampler = Q().zero()
    for c in classes:
        sampler += Q(list(c.terms.keys())[0])
    return sampler

#
#    fifth set of functions is is_X functions for NC SYMTWCAC SYMFF SYMTWCFF
#

def is_nc_hamiltonian(H):
    sampler = obtain_sampling_operator(H)
    AC = obtain_ac_graph(sampler)
    return is_complete_graph(AC)

def is_symtwcac_hamiltonian(H):
    sampler = obtain_sampling_operator(H)
    AC = obtain_ac_graph(sampler)
    return is_ccs_are_cliques(AC)

def is_symff_hamiltonian(H):
    sampler = obtain_sampling_operator(H)
    AC = obtain_ac_graph(sampler)
    return is_line_graph(AC)

def is_symtwcff_hamiltonian(H):
    sampler = obtain_sampling_operator(H)
    AC = obtain_ac_graph(sampler)
    return is_ccs_are_line_graphs(AC)

#
#    sixth set of functions are for implementing sorted insertion
#

def methodtag_to_verifier(methodtag):
    if methodtag == 'fc':
        return is_fc_hamiltonian
    elif methodtag == 'ac':
        return is_ac_hamiltonian
    elif methodtag == 'twc_ac':
        return is_twc_ac_hamiltonian
    elif methodtag == 'sym_twc_ac':
        return is_symtwcac_hamiltonian
    elif methodtag == 'nc':
        return is_nc_hamiltonian
    elif methodtag == 'ff':
        return is_ff_hamiltonian
    elif methodtag == 'twc_ff':
        return is_twc_ff_hamiltonian
    elif methodtag == 'sym_ff':
        return is_symff_hamiltonian
    elif methodtag == 'sym_twc_ff':
        return is_symtwcff_hamiltonian
    else:
        print('nooo')
        return is_fc_hamiltonian
    
def inclusion_criterion(H, A, verifier):
    if isinstance(A, tuple):
        A = Q(A)

    Hcopy  = copy_hamiltonian(H)
    Hcopy += A
    return verifier(Hcopy)

def single_sorted_insertion_step(H, verifier):

    fragment = Q().zero()

    for term, coef in H.terms.items():
        
        if fragment == Q().zero():
            fragment += coef * Q(term)

        else:
            if inclusion_criterion(fragment, term, verifier):
                fragment += coef * Q(term)

    return fragment

def remove_fragment(H, fragment):
    H -= fragment
    H.compress()

def sorted_insertion_decomposition(H, methodtag):
    verifier = methodtag_to_verifier(methodtag)
    
    decomp = []
    while H != Q().zero():
        fragment = single_sorted_insertion_step(H, verifier)
        remove_fragment(H, fragment)
        decomp.append(fragment)
    return decomp

#
#    seventh set of functions is for metric calculations
#

def l1_norm(operator):
    l1 = 0
    for term, coef in operator.terms.items():
        l1 += np.abs(coef)
    return l1

def l2_norm(operator):
    l2 = 0
    for term, coef in operator.terms.items():
        l2 += coef**2
    return np.sqrt(l2)

def operator_length(operator):
    return len(operator.terms)

def variance_metric(H, decomp, N):
    psi = ggs(gso(H, N))[1]

    vars = np.zeros(len(decomp), dtype=np.complex128)
    for i, frag in enumerate(decomp):
        vars[i] = variance( gso(frag, N), psi )
    return np.sum((vars)**(1/2))**2

def optimal_fragment_metrics(decomp):
    l1     = -1
    l2     = -1
    length = -1

    for frag in decomp:
        fragl1     = l1_norm(frag)
        fragl2     = l2_norm(frag)
        fraglength = operator_length(frag)

        if fragl1 > l1:
            l1     = fragl1
            l2     = fragl2
            length = operator_length(frag)
        
    return l1, l2, length

def save_decomposition(decomp, moltag, methodtag):
    filename = f'decompositions/{moltag}_{methodtag}'
    with open(filename, 'wb') as f:
        pickle.dump(decomp, f)
    return None

if __name__ == '__main__':
    moltag    = sys.argv[1]
    methodtag = sys.argv[2]

    N      = N_QUBITS[moltag]
    H      = load_hamiltonian(moltag)
    Hcopy  = copy_hamiltonian(H)
    decomp = sorted_insertion_decomposition(Hcopy, methodtag)
    var    = variance_metric(H, decomp, N)
    
    l1     = l1_norm(H)
    l2     = l2_norm(H)
    length = operator_length(H)

    optl1, optl2, optlength = optimal_fragment_metrics(decomp)

    print(f'''
        molecule  : {moltag}
        method    : {methodtag}
        variance  : {var}
          
        l1norm    : {optl1}
        l2norm    : {optl2}
        length    : {optlength}
          
        Haml1     : {l1}
        Haml2     : {l2}
        Hamlength : {length}

    ''')

    save_decomposition(decomp, moltag, methodtag)

