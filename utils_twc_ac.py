import numpy as np
from openfermion import QubitOperator as Q
from numpy.random import uniform

from utils_basic import (
    is_commuting, 
    is_anticommuting,
    copy_hamiltonian,
    apply_unitary_product,
    random_pauli_term
)

from utils_ac import (
    obtain_SO_solving_unitary
)

from utils_fc import (
    solve_fc_hamiltonian
)

def obtain_fc_clique(H):

    fc_clique = Q().zero()

    for t, s in H.terms.items():

        is_fc = True
        for r, _ in H.terms.items():
            if not is_commuting(t, r):
                is_fc = False
                break
        
        if is_fc:
            fc_clique += s * Q(t)

    return fc_clique

def obtain_ac_cliques(H):
    H.compress()
    ac_cliques = []

    while H != Q().zero():
        current_clique = Q().zero()

        op              = list(H.terms.keys())[0]
        current_clique += H.terms[op] * Q(op)

        for t, s in H.terms.items():
            if is_anticommuting(op, t):
                current_clique += s * Q(t)

        ac_cliques.append(current_clique)
        H -= current_clique
        H.compress()
    
    return ac_cliques

def obtain_cliques_of_twc_ac_hamiltonian(H):
    fc_clique  = obtain_fc_clique(H)
    H         -= fc_clique
    ac_cliques = obtain_ac_cliques(H)

    return fc_clique, ac_cliques

def verify_twc_ac_property_for_cliques(cliques):

    for i, clique1 in enumerate(cliques):
        for j, clique2 in enumerate(cliques):

            if i == j:
                for t, _ in clique1.terms.items():
                    for s, _ in clique2.terms.items():
                        if t != s:
                            if not is_anticommuting(t, s):
                                return False
                            
            elif i > j:
                for t, _ in clique1.terms.items():
                    for s, _ in clique2.terms.items():
                        if not is_commuting(t, s):
                            return False
                        
    return True

def is_twc_ac_hamiltonian(H):
    Hc          = copy_hamiltonian(H)
    fc_clique   = obtain_fc_clique(Hc)
    Hc         -= fc_clique
    ac_cliques  = obtain_ac_cliques(Hc)

    return verify_twc_ac_property_for_cliques(ac_cliques)

def twc_ac_inclusion_criterion(H, A):
    if isinstance(A, tuple):
        A = Q(A)

    Hcopy  = copy_hamiltonian(H)
    Hcopy += A
    return is_twc_ac_hamiltonian(Hcopy)

def random_twc_ac_hamiltonian(Nqubits, Nterms):
    H = Q().zero()

    while len(H.terms) < Nterms:
        _, A = random_pauli_term(Nqubits)
        if twc_ac_inclusion_criterion(H, A):
            H += uniform(-2, 2) * A

    return H

def obtain_SO_direct_sum_solving_unitary(ac_cliques):
    SO_solver      = []
    SO_solver_flat = []
    for clique in ac_cliques:
        _, _, U = obtain_SO_solving_unitary(clique)
        SO_solver.append(U)
        SO_solver_flat += U
    return SO_solver, SO_solver_flat

def solve_twc_ac_hamiltonian(H, N, first_active_qubit=0, detailed=False):
    Hc = copy_hamiltonian(H)

    fc_clique, ac_cliques     = obtain_cliques_of_twc_ac_hamiltonian(Hc)
    p, C, cliff_U             = solve_fc_hamiltonian(fc_clique, N, C_list=None, start_idx=first_active_qubit)
    K                         = len(C)
    tilde_ac_cliques          = [apply_unitary_product(clique, cliff_U) for clique in ac_cliques]
    SO_solver, SO_solver_flat = obtain_SO_direct_sum_solving_unitary(tilde_ac_cliques)
    solved_tilde_ac_cliques   = apply_unitary_product(sum(tilde_ac_cliques), SO_solver_flat)
    q, D, cliff_V             = solve_fc_hamiltonian(solved_tilde_ac_cliques, N, C_list=None, start_idx=first_active_qubit+K)

    if detailed:
        return p, C, cliff_U, SO_solver, SO_solver_flat, q, D, cliff_V
    else:
        return cliff_U + SO_solver_flat + cliff_V

def restricted_solve_twc_ac_hamiltonian(H, N):
    """
    returns oplus_\lam SO(L_lam + 1) unitary that maps twc_ac to fc, where twc_ac has no intrinsic fc part
    """
    Hc = copy_hamiltonian(H)

    fc_clique, ac_cliques = obtain_cliques_of_twc_ac_hamiltonian(Hc)
    assert fc_clique     == Q().zero()
    _, SO_solver_flat     = obtain_SO_direct_sum_solving_unitary(ac_cliques)

    return SO_solver_flat

def twc_ac_hamiltonian_analytical_solution(H, N):
    Hc = copy_hamiltonian(H)
    
    fc_clique, ac_cliques = obtain_cliques_of_twc_ac_hamiltonian(Hc)
    p, C, cliff_U         = solve_fc_hamiltonian(fc_clique, N, C_list=None, start_idx=0)
    K                     = len(C) 
    Zlist                 = [Q(f'Z{k}') for k in range(K)]

    twc_ac_solution = Q()
    for lmbda, Hlmbda in enumerate(ac_cliques):
        c = 0
        for _, s in Hlmbda.terms.items():
            c += s**2
        c = np.sqrt(c)
        twc_ac_solution += c * Q(f'Z{K + lmbda}')

    return p(Zlist) + twc_ac_solution