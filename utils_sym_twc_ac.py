import numpy as np
from openfermion import QubitOperator as Q

from utils_basic import (
    is_commuting,
    is_anticommuting,
    copy_hamiltonian
)

from utils_fc import (
    obtain_generators,
    obtain_polynomial_representation_of_fc_hamiltonian,
    solve_multiple_independent_commuting_generators,
)

from utils_twc_ac import (
    restricted_solve_twc_ac_hamiltonian
)

from utils_nc import (
    move_z_string_to_polynomial,
)


##############################################################################################################################
#
#    Subroutines for solving Sym-TWC-AC Hamiltonians
#
##############################################################################################################################

#
#    Factorization and verification
#

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

def obtain_cluster_subgraphs(H):

    cluster_graphs = []
    while H != Q().zero():
        current_cluster  = Q().zero()
        op               = list(H.terms.keys())[0]
        current_cluster += H.terms[op] * Q(op)

        for t, s in H.terms.items():
            if is_anticommuting(t, op):
                current_cluster += s * Q(t)
                second_op        = t
                break

        for t, s in H.terms.items():
            distinctness_check  = (t != op) and (t != second_op)
            anticommuting_check = is_anticommuting(t, op) or is_anticommuting(t, second_op)
            if distinctness_check and anticommuting_check:
                current_cluster += s * Q(t)

        H -= current_cluster
        H.compress()
        cluster_graphs.append(current_cluster)

    return cluster_graphs

def obtain_cliques_from_cluster_graphs(cluster_graphs):
    cliques = []

    for cluster_op in cluster_graphs:
        current_cluster_list = []

        while cluster_op != Q().zero():
            current_clique  = Q().zero()
            op              = list(cluster_op.terms.keys())[0]
            current_clique += cluster_op.terms[op] * Q(op)

            for t, s in cluster_op.terms.items():
                if (t != op) and is_commuting(t, op):
                    current_clique += s * Q(t)

            cluster_op -= current_clique
            cluster_op.compress()
            current_cluster_list.append(current_clique)

        cliques.append(current_cluster_list)
    
    return cliques

def obtain_cliques_of_sym_twc_ac_hamiltonian(H):
    fc_clique = obtain_fc_clique(H)
    H -= fc_clique
    H.compress()

    clusters = obtain_cluster_subgraphs(H)
    cliques  = obtain_cliques_from_cluster_graphs(clusters)

    return fc_clique, cliques

def verify_sym_twc_ac_property_for_cliques(cliques):
    for lam, T_lam in enumerate(cliques):
        for sig, T_sig in enumerate(cliques):

            for i, clique_i in enumerate(T_lam):
                for j, clique_j in enumerate(T_sig):

                    for op1, _ in clique_i.terms.items():
                        for op2, _ in clique_j.terms.items():

                            if lam == sig:
                                if i == j:
                                    if not is_commuting(op1, op2):
                                        return False
                                elif i > j:
                                    if not is_anticommuting(op1, op2):
                                        return False
                                    
                            elif lam > sig:
                                if not is_commuting(op1, op2):
                                    return False
                                
    return True

def is_sym_twc_ac_hamiltonian(H):
    Hc = copy_hamiltonian(H)

    fc_clique = obtain_fc_clique(Hc)
    Hc -= fc_clique
    Hc.compress()

    clusters = obtain_cluster_subgraphs(Hc)
    cliques  = obtain_cliques_from_cluster_graphs(clusters)
    return verify_sym_twc_ac_property_for_cliques(cliques)

def sym_twc_ac_inclusion_criterion(H, A):
    if isinstance(A, tuple):
        A = Q(A)

    Hcopy  = copy_hamiltonian(H)
    Hcopy += A
    return is_sym_twc_ac_hamiltonian(Hcopy)

def factorize_cliques_of_sym_twc_ac_hamiltonian(fc_clique, cliques):
    A_list = [[Q("")]]
    operator_coefs = [[fc_clique]]

    for T_lam in cliques:
        lam_As    = []
        lam_coefs = []

        for clique in T_lam:
            representative = Q(list(clique.terms.keys())[0])
            shifted_clique = clique * representative
            lam_As.append(representative)
            lam_coefs.append(shifted_clique)

        A_list.append(lam_As)
        operator_coefs.append(lam_coefs)

    return A_list, operator_coefs

def solve_operator_coefs_of_sym_twc_ac_hamiltonian(operator_coefs, N, C_list=None, start_idx=0):

    if C_list is None:
        return_C_list = True
        combined_operator_total_list = [Q(op) for cluster in operator_coefs for fc in cluster for op in fc.terms]
        combined_operator_list = []
        for op in combined_operator_total_list:
            if not op in combined_operator_list:
                combined_operator_list.append(op)
        C_list = obtain_generators(combined_operator_list, N)
    
    else:
        return_C_list = False

    p = [
        [obtain_polynomial_representation_of_fc_hamiltonian(fc, N, generators=C_list) for fc in cluster] 
        for cluster in operator_coefs
        ]
    
    U = solve_multiple_independent_commuting_generators(C_list, start_idx=start_idx)

    if return_C_list:
        return C_list, p, U
    return p, U

def factorize_sym_twc_ac_hamiltonian(H, N):
    fc_clique, sym_twc_ac_cliques = obtain_cliques_of_sym_twc_ac_hamiltonian(H)

    assert verify_sym_twc_ac_property_for_cliques(sym_twc_ac_cliques)

    A_list, operator_coefs = factorize_cliques_of_sym_twc_ac_hamiltonian(fc_clique, sym_twc_ac_cliques)

    C, p, U = solve_operator_coefs_of_sym_twc_ac_hamiltonian(operator_coefs, N, C_list=None, start_idx=0)

    return A_list, C, p, U

#
#    move z-terms to polynomial
#

def convert_sym_twc_ac_to_tilde_representation(p, A, Lam, K):
    """
    note that A should be cliff_U transformed operators
    """
    ptilde = []
    Atilde = []

    for lam in range(Lam):
        cluster_ptilde = []
        cluster_Atilde = []

        for i in range(len(A[lam])):
            cur_ptilde, cur_Atilde = move_z_string_to_polynomial(p[lam][i], A[lam][i], K)
            cluster_ptilde.append(cur_ptilde)
            cluster_Atilde.append(cur_Atilde)

        ptilde.append(cluster_ptilde)
        Atilde.append(cluster_Atilde)

    return ptilde, Atilde

#
#    obtain operators defining symmetry blocks, eigenvalues, and final diagonalizing clifford
#

def obtain_sym_twc_ac_hamiltonian_blocks(ptilde, Atilde, Lam, K, N):
    Ls = [len(Atilde[lam]) for lam in range(Lam)]

    def Ham(v):
        return sum([sum([ptilde[lam][i](v)*Atilde[lam][i] for i in range(Ls[lam])]) for lam in range(Lam)])
    
    return Ham

def obtain_sym_twc_ac_unitary_blocks(ptilde, Atilde, Lam, K, N):
    
    Ham = obtain_sym_twc_ac_hamiltonian_blocks(ptilde, Atilde, Lam, K, N)

    def Unit(v):
        Hv = Ham(v)
        Hv -= Hv.constant
        Hv.compress()
        Uv = restricted_solve_twc_ac_hamiltonian(Hv, N)
        return Uv
    
    return Unit

def obtain_sym_twc_ac_diagonal_blocks(ptilde, Atilde, Lam, K, N):
    Ls = [len(Atilde[lam]) for lam in range(Lam)]

    def Diag(v):
        return (
            ptilde[0][0](v) + 
            sum([np.linalg.norm([ptilde[lam][i](v) for i in range(Ls[lam])]) * Q(f'Z{(K-1)+lam}') for lam in range(1, Lam)])
        )
    
    return Diag

def obtain_sym_twc_ac_eigenvalues(ptilde, Atilde, Lam, K, N):
    Ls = [len(Atilde[lam]) for lam in range(Lam)]

    def E_H(v, w):
        return (
            ptilde[0][0](v) + 
            sum([np.linalg.norm([ptilde[lam][i](v) for i in range(Ls[lam])]) * w[lam-1] for lam in range(1, Lam)])
        )
    
    return E_H

def obtain_sym_twc_ac_hamiltonian_blocks_clifford_and_eigenvalues(ptilde, Atilde, Lam, K, N):
    Ham    = obtain_sym_twc_ac_hamiltonian_blocks(ptilde, Atilde, Lam, K, N)
    Unit   = obtain_sym_twc_ac_unitary_blocks(ptilde, Atilde, Lam, K, N)
    Diag   = obtain_sym_twc_ac_diagonal_blocks(ptilde, Atilde, Lam, K, N)
    cliffV = solve_multiple_independent_commuting_generators([Q(list(Atilde[lam][-1].terms.keys())[0]) for lam in range(1, Lam)], start_idx=K)
    E_H    = obtain_sym_twc_ac_eigenvalues(ptilde, Atilde, Lam, K, N)

    return Ham, Unit, Diag, cliffV, E_H
