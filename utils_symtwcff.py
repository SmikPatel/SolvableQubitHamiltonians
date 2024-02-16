import numpy as np
import networkx as nx
from openfermion import (
    QubitOperator as Q
)
from utils_basic import (
    is_commuting,
    is_anticommuting,
    copy_hamiltonian,
    random_pauli_term
)
from utils_fc import (
    obtain_generators,
    obtain_polynomial_representation_of_fc_hamiltonian,
    solve_multiple_independent_commuting_generators
)
from utils_ff import (
    obtain_ac_graph,
    obtain_cycle_symmetries,
    is_line_graph, 
    solve_ff_qubit_hamiltonian
)
from utils_nc import (
    move_z_string_to_polynomial
)
from numpy.random import uniform

def obtain_fc_clique(H):

    fc_clique = Q().zero()

    for term, coef in H.terms.items():

        is_fc = True
        for r, _ in H.terms.items():
            if not is_commuting(term, r):
                is_fc = False
                break
        
        if is_fc:
            fc_clique += coef * Q(term)

    return fc_clique

def remove_operator(H, operator):
    H -= operator
    H.compress()

def obtain_connected_component_operators(H):
    """
    return operators Hccs which correspond to connected components in anti-compatibility graph of H

    this means: all Pauli terms in Hcc[i] commute with all Pauli terms in Hcc[j] for i != j
    """
    AC        = obtain_ac_graph(H)
    cc_nodes  = nx.connected_components(AC)
    cc_graphs = [AC.subgraph(nodes) for nodes in cc_nodes]

    Hccs = []
    for graph in cc_graphs:
        Hcc = Q().zero()
        for node in graph:
            Hcc += H.terms[node] * Q(node)
        Hccs.append(Hcc)
    return Hccs

def verify_connected_component_operators(Hccs):
    """
    verify that operators in Hccs[i] all commute with operators in Hccs[j] for i != j
    """
    for i, Hcc_i in enumerate(Hccs):
        for j, Hcc_j in enumerate(Hccs):
            if i < j:
                for op_alph in Hcc_i.terms.keys():
                    for op_beta in Hcc_j.terms.keys():
                        if not is_commuting(op_alph, op_beta):
                            return False
    return True

def obtain_adjacency_list(H):
    """
    return dictionary of {op : Q[op]} where Q[op] is subhamiltonian of H consisting of ops which anticommute with op, and 
    are therefore adjacent in the AC graph
    """
    adjacency_list = dict()
    for term in H.terms.keys():
        
        term_adjacencies = Q().zero()
        
        for t, s in H.terms.items():
            if is_anticommuting(term, t):
                term_adjacencies += s * Q(t)
        
        adjacency_list[term] = term_adjacencies
    
    return adjacency_list

def obtain_twin_equivalence_class_operators(H):
    """
    obtain list of ops which sum to H, where each op is a linear-combination of twin-Paulis in H

    the property of being twins is an equiv-relation so this decomp is unique up to permutation of the output list

    the twin Pauli operators have identical adjacencies and so they have identical values in the adjacency list 
    """
    adjacency_list = obtain_adjacency_list(H)

    twin_classes = []
    while adjacency_list != dict():
        current_class = Q().zero()

        seed_term        = list(adjacency_list.keys())[0]
        seed_coef        = H.terms[seed_term]
        seed_adjacencies = adjacency_list[seed_term]
        current_class   += seed_coef * Q(seed_term)

        for term, term_adjacencies in adjacency_list.items():
            if (term != seed_term) and (term_adjacencies == seed_adjacencies):
                current_class += H.terms[term] * Q(term)

        twin_classes.append(current_class)

        for term in current_class.terms:
            del adjacency_list[term]

    return twin_classes

def obtain_twin_classes_of_symtwcff_hamiltonian(H, verify=False):
    fc_clique    = obtain_fc_clique(H)
    remove_operator(H, fc_clique)
    Hccs         = obtain_connected_component_operators(H)

    if verify:
        assert verify_connected_component_operators(Hccs)

    twin_classes = [obtain_twin_equivalence_class_operators(h) for h in Hccs]

    return fc_clique, twin_classes

def factorize_cliques_of_symtwcff_hamiltonian(fc_clique, twin_classes):
    Alist          = [ [Q("")] ]
    operator_coefs = [ [fc_clique] ]

    for connected_component in twin_classes:
        cc_Alist   = []
        cc_opcoefs = []
        for twin_class in connected_component:
            representative = Q(list(twin_class.terms.keys())[0])
            shifted_class  = twin_class * representative
            cc_Alist.append(representative)
            cc_opcoefs.append(shifted_class)
        Alist.append(cc_Alist)
        operator_coefs.append(cc_opcoefs)
    
    return Alist, operator_coefs

def is_symtwcff_hamiltonian(H):
    fc_clique, twin_classes = obtain_twin_classes_of_symtwcff_hamiltonian(H)
    Alist, operator_coefs   = factorize_cliques_of_symtwcff_hamiltonian(fc_clique, twin_classes)

    if len(Alist) < 2:
        return True
    else:
        Htwinfree = sum([sum(component) for component in Alist])
        AC        = obtain_ac_graph(Htwinfree)
        cc_nodes  = nx.connected_components(AC)
        cc_graphs = [AC.subgraph(nodes) for nodes in cc_nodes]
        for graph in cc_graphs:
            if not is_line_graph(graph):
                return False
        return True
    
def symtwcff_inclusion_criterion(H, A):
    if isinstance(A, tuple):
        A = Q(A)

    Hcopy = copy_hamiltonian(H)
    Hcopy += A
    return is_symtwcff_hamiltonian(Hcopy)

def random_symtwcff_hamiltonian(Nqubits, Nterms):

    H = Q()

    while len(H.terms) < Nterms:

        _, A = random_pauli_term(Nqubits)
        if symtwcff_inclusion_criterion(H, A):
            H += uniform(-2, 2) * A
            print(len(H.terms), end=' ')

    assert is_symtwcff_hamiltonian(H)
    return H

def verify_that_operator_coefs_are_symmetries(H, operator_coefs):
    """
    every Pauli term in every op in operator_coefs is a product of two-twins; therefore, they should all be symmetries of the 
    initial Hamiltonian, which is what this function verifies
    """
    for component in operator_coefs:
        for op in component:
            for term in op.terms.keys():
                for Hterm in H.terms.keys():
                    if not is_commuting(term, Hterm):
                        return False
    return True

def simultaneously_solve_operator_coefs_and_cycle_symmetries_symtwcff(operator_coefs, N, syms=[], verify_syms=True):
    
    combined_operator_total_list = [Q(op) for component in operator_coefs for H in component for op in H.terms.keys()]
    combined_operator_list = []
    for op in combined_operator_total_list:
        if op not in combined_operator_list:
            combined_operator_list.append(op)

    if verify_syms:
        for sym in syms:
            for op in combined_operator_list:
                assert is_commuting(sym, op)

    for sym in syms:
        if sym not in combined_operator_list:
            combined_operator_list.append(sym)

    C = obtain_generators(combined_operator_list, N)
    p = [
        [obtain_polynomial_representation_of_fc_hamiltonian(fc, N, generators=C) for fc in component] 
        for component in operator_coefs
        ]
    U = solve_multiple_independent_commuting_generators(C)

    return C, p, U

def factorize_symtwcff_hamiltonian(H, N, verify_twin_free_quotients=True):
    """
    obtain factorization of symtwcff Hamiltonian into Lie algebra generators with polynomial coefficients, and the Clifford
    that solves the polynomial coefficients
    """
    Hcopy                   = copy_hamiltonian(H)
    fc_clique, twin_classes = obtain_twin_classes_of_symtwcff_hamiltonian(Hcopy, verify=True)
    Alist, operator_coefs   = factorize_cliques_of_symtwcff_hamiltonian(fc_clique, twin_classes)

    assert verify_that_operator_coefs_are_symmetries(H, operator_coefs)
    
    if len(Alist) > 1:
        Htwinfrees = [sum(component) for component in Alist[1:]]
        if verify_twin_free_quotients:
            for Htf in Htwinfrees:
                AC = obtain_ac_graph(Htf)
                assert is_line_graph(AC)

        cycle_syms = [sym for Htf in Htwinfrees for sym in obtain_cycle_symmetries(Htf)]
    
    else:
        cycle_syms = []

    C, p, U                 = simultaneously_solve_operator_coefs_and_cycle_symmetries_symtwcff(operator_coefs,
                                                                                                N,
                                                                                                syms=cycle_syms,
                                                                                                verify_syms=True)
    
    return Alist, C, p, U

def convert_to_tilde_representation(p, A, K):

    ptilde = []
    Atilde = []

    Lam    = len(p)
    d      = [len(p[lam]) for lam in range(Lam)]

    for lam in range(Lam):
        component_ptilde = []
        component_Atilde = []
        for a in range(d[lam]):
            cur_ptilde, cur_Atilde = move_z_string_to_polynomial(p[lam][a], A[lam][a], K)
            component_ptilde.append(cur_ptilde)
            component_Atilde.append(cur_Atilde)
        ptilde.append(component_ptilde)
        Atilde.append(component_Atilde)

    return ptilde, Atilde

def obtain_symtwcff_hamiltonian_blocks(ptilde, Atilde):
    Lam = len(ptilde)
    d   = [len(ptilde[lam]) for lam in range(Lam)]

    def Ham(v):
        return [sum([ptilde[lam][a](v) * Atilde[lam][a] for a in range(d[lam])]) for lam in range(Lam)]
    
    return Ham

def obtain_symtwcff_generator_blocks(ptilde, Atilde):
    Lam = len(ptilde)
    d   = [len(ptilde[lam]) for lam in range(Lam)]
    Ham = obtain_symtwcff_hamiltonian_blocks(ptilde, Atilde)

    def Gen(v):
        Hv = Ham(v)
        Gv = [Q("")]
        for H in Hv[1:]:
            H -= H.constant
            H.compress()
            Gv.append(solve_ff_qubit_hamiltonian(H))
        return Gv
    
    return Gen