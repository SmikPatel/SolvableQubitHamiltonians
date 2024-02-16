import numpy as np
import networkx as nx
from openfermion import QubitOperator as Q
from openfermion import MajoranaOperator as M
from openfermion import hermitian_conjugated as dagger
from openfermion import commutator, anticommutator, get_fermion_operator
from openfermion import get_sparse_operator as gso
import random
from numpy.random import uniform
from math import ceil

def copy_hamiltonian(H):
    H_copy = Q().zero()

    for t, s in H.terms.items():
        H_copy += s * Q(t)

    assert (H - H_copy) == Q().zero()
    return H_copy

def random_pauli_term(Nqubits):
    
    letters = ['X', 'Y', 'Z', 'I']

    term_tuple = []

    for i in range(Nqubits):
        current_letter = random.sample(letters, 1)[0]
        if current_letter != 'I':
            term_tuple.append( (i, current_letter) )

    return tuple(term_tuple), Q(tuple(term_tuple))

def is_commuting(A, B):
    if isinstance(A, tuple):
        A = Q(A)

    if isinstance(B, tuple):
        B = Q(B)

    return commutator(A, B) == Q().zero()

def is_anticommuting(A, B):
    if isinstance(A, tuple):
        A = Q(A)

    if isinstance(B, tuple):
        B = Q(B)

    return anticommutator(A, B) == Q().zero()

def apply_unitary_product(H, U_list):
    for U in U_list:
        H = U * H * dagger(U)
        H.compress()
    return H

def compute_product_of_unitaries(U_list):
    U_prod = Q("")
    for U in U_list:
        U_prod = U * U_prod
    return U_prod

def is_unitary(U):
    return U * dagger(U) == Q("")

def clifford(A, B):
    assert is_anticommuting(A, B)
    return (A + B) / np.sqrt(2)

def random_hamiltonian_with_specified_terms(op_list):
    H = Q().zero()
    for op in op_list:
        H += uniform(-2, 2) * op
    return H

def goto_matrix(X, N):
    if isinstance(X, M):
        return gso(get_fermion_operator(X), ceil(N/2)).toarray()
    else:
        return gso(X, N).toarray()

def obtain_spectrum(X, N):
    Xmat = goto_matrix(X, N)
    return sorted(np.round(np.linalg.eig(Xmat)[0], 8))

def obtain_spectrum_no_degeneracies(X, N):
    spectrum = obtain_spectrum(X, N)
    return sorted(list(set(spectrum)))

def is_termwise_symmetry(H, S):
    for term in H.terms.keys():
        if not is_commuting(term, S):
            return False
    return True

def introduce_symmetries(H, syms):
    
    Hsym = Q()

    for term, coef in H.terms.items():

        current_sym = Q("")
        for S in syms:
            if uniform(0, 1) < 0.5:
                current_sym = current_sym * S
        
        Hsym += coef * Q(term) * current_sym

    return Hsym

def shift_hamiltonian_qubits(H, K):
    """
    H is an N qubit Hamiltonian which acts as I on the first K qubits

    return is an N - K qubit Hamiltonian Hc which is the same as H but acts on the first N - K qubits
    """
    Hc = Q()

    for term, coef in H.terms.items():
        shifted_term = []
        for op in term:
            shifted_term.append( (op[0] - K, op[1]) )
        shifted_term = tuple(shifted_term)
        Hc += coef * Q(shifted_term)

    return Hc

def obtain_ac_graph(H):
    G = nx.Graph()
    G.add_nodes_from(H.terms.keys())

    for i, t in enumerate(H.terms.keys()):
        for j, r in enumerate(H.terms.keys()):
            if (i < j) and is_anticommuting(t, r):
                G.add_edges_from( [(t,r)] )

    return G

#
#    Lie algebra closure functions in Majorana and Qubit spaces
#

def close_majorana_algebra(oplist):

    modified = True
    while modified:

        new_ops = []

        for i, A in enumerate(oplist):
            for j, B in enumerate(oplist):
                if (i < j) and (not commutator(A, B) == M()):
                    AB = M(list(commutator(A, B).terms.keys())[0])
                    if (not AB in oplist) and (not AB in new_ops):
                        new_ops.append(AB)

        if new_ops != []:
            modified = True
            oplist = oplist + new_ops

        else:
            modified = False

    return oplist

def close_majorana_hamiltonian_algebra(H):
    op_list       = [M(x) for x in H.terms]
    op_list       = close_majorana_algebra(op_list)
    op_list_terms = [list(G.terms.keys())[0] for G in op_list]

    for term in op_list_terms:
        if term not in H.terms:
            H.terms[term] = 0.0

    return H

def close_qubit_algebra(oplist):

    modified = True
    while modified:

        oplist_oned = [Q(list(op.terms.keys())[0]) for op in oplist]
        new_ops     = []

        for i, A in enumerate(oplist):
            for j, B in enumerate(oplist):
                if (i < j) and (not is_commuting(A, B)):
                    AB = Q(list(commutator(A, B).terms.keys())[0])
                    if (not AB in oplist_oned) and (not AB in new_ops):
                        new_ops.append(AB)

        if new_ops != []:
            modified = True
            oplist = oplist + new_ops

        else:
            modified = False

    return oplist

def close_qubit_hamiltonian_algebra(H):
    op_list       = [Q(P) for P in H.terms]
    op_list       = close_qubit_algebra(op_list)
    op_list_terms = [list(P.terms.keys())[0] for P in op_list]

    for term in op_list_terms:
        if term not in H.terms:
            H.terms[term] = 0.0

    return H