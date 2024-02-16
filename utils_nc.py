import numpy as np
from openfermion import QubitOperator as Q

from utils_basic import (
    is_commuting,
    is_anticommuting,
    copy_hamiltonian,
    compute_product_of_unitaries
)

from utils_ac import (
    obtain_SO_solving_unitary
)

from utils_fc import (
    decimal_to_binary_list,
    obtain_generators,
    obtain_polynomial_representation_of_fc_hamiltonian,
    solve_multiple_independent_commuting_generators,
    instantiate_one,
    z_proj
)

#
#    first set of functions: for determining if a Hamiltonian is non-contextual and factorizing it into sum_i p_i(C)A_i
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

def obtain_twac_fc_cliques(H):

    twac_fc_cliques = []

    while H != Q().zero():

        current_clique = Q().zero()
        op = list(H.terms.keys())[0]
        current_clique += H.terms[op] * Q(op)

        for t, s in H.terms.items():
            if t != op and is_commuting(t, op):
                current_clique += s * Q(t)

        twac_fc_cliques.append(current_clique)
        H -= current_clique
        H.compress()

    return twac_fc_cliques

def obtain_cliques_of_nc_hamiltonian(H):
    fc_clique = obtain_fc_clique(H)
    H -= fc_clique
    H.compress()
    twac_fc_cliques = obtain_twac_fc_cliques(H)
    return fc_clique, twac_fc_cliques

def verify_twac_fc_property_for_cliques(twac_fc_cliques):

    for i, clique_i in enumerate(twac_fc_cliques):
        for j, clique_j in enumerate(twac_fc_cliques):

            if i == j:
                for t, _ in clique_i.terms.items():
                    for s, _ in clique_j.terms.items():
                        if not is_commuting(t, s):
                            return False

            elif i > j:
                for t, _ in clique_i.terms.items():
                    for s, _ in clique_j.terms.items():
                        if not is_anticommuting(t, s):
                            return False
                        
    return True

def is_nc_hamiltonian(H):
    Hc = copy_hamiltonian(H)
    fc_clique = obtain_fc_clique(Hc)
    Hc -= fc_clique
    Hc.compress()
    twac_fc_cliques = obtain_twac_fc_cliques(Hc)
    return verify_twac_fc_property_for_cliques(twac_fc_cliques)

def nc_inclusion_criterion(H, A):
    if isinstance(A, tuple):
        A = Q(A)

    Hcopy  = copy_hamiltonian(H)
    Hcopy += A
    return is_nc_hamiltonian(Hcopy)

def factorize_cliques_of_nc_hamiltonian(fc_clique, twac_fc_cliques):
    A_list         = [Q("")]
    operator_coefs = [fc_clique]

    for clique in twac_fc_cliques:
        representative = Q(list(clique.terms.keys())[0])
        shifted_clique = clique * representative

        A_list.append(representative)
        operator_coefs.append(shifted_clique)

    return A_list, operator_coefs

def solve_multiple_twc_fc_hamiltonians(H_list, N, C_list=None, start_idx=0):

    if C_list is None:
        return_C_list = True
        combined_operator_total_list = [Q(op) for H in H_list for op in H.terms]
        combined_operator_list = []
        for op in combined_operator_total_list:
            if not op in combined_operator_list:
                combined_operator_list.append(op)
        C_list = obtain_generators(combined_operator_list, N)

    else:
        return_C_list = False
    
    p = [obtain_polynomial_representation_of_fc_hamiltonian(fc, N, generators=C_list) for fc in H_list]
    U = solve_multiple_independent_commuting_generators(C_list, start_idx=start_idx)

    if return_C_list:
        return C_list, p, U
    return p, U

def factorize_nc_hamiltonian(H, N):
    """
    obtain representation of non-contextual Hamiltonian as a linear combination of anti-commuting Pauli operators, where 
    coefficients are twc-fc Hamiltonians, along with the tapering clifford which maps the twc-fc Hamiltonians to ising Hamiltonians
    """
    fc_clique = obtain_fc_clique(H)
    H -= fc_clique
    H.compress()

    twac_fc_cliques = obtain_twac_fc_cliques(H)
    assert verify_twac_fc_property_for_cliques(twac_fc_cliques)

    A_list, operator_coefs = factorize_cliques_of_nc_hamiltonian(fc_clique, twac_fc_cliques)

    C, p, U = solve_multiple_twc_fc_hamiltonians(operator_coefs, N)

    return A_list, C, p, U


#
#    second set of functions: {p, A} --> {ptilde, Atilde}
#

def obtain_z_string(term, K):
    z_string = Q("")
    term.compress()
    term_tuple = list(term.terms.keys())[0]

    for op in term_tuple:
        if op[0] < K:
            if op[1] != 'Z':
                print(f"{term} does not have trailing z string on qubits 0,...,{K-1}")
            else:
                z_string *= Q(f'Z{op[0]}')

    return z_string

def evaluate_product_based_on_z_string_support(C, z_string):
    valid_indices = [op[0] for op in list(z_string.terms.keys())[0]]

    one_type = type(C[0])
    product  = instantiate_one(one_type)

    for k in range(len(C)):
        if k in valid_indices:
            product *= C[k]

    return product

def move_z_string_to_polynomial(p, A, K):
    z_string = obtain_z_string(A, K)
    Atilde   = z_string * A

    def ptilde(C):
        return p(C) * evaluate_product_based_on_z_string_support(C, z_string)
    
    return ptilde, Atilde

def convert_to_tilde_representation(p, A, K):
    """
    note: it is assumed that A is already clifford transformed so that H = sum_i p_i(Z)*A_i
    """
    ptilde = []
    Atilde = []

    for i in range(len(A)):
        cur_ptilde, cur_Atilde = move_z_string_to_polynomial(p[i], A[i], K)
        ptilde.append(cur_ptilde)
        Atilde.append(cur_Atilde)

    return ptilde, Atilde

#
#    third set of functions: Hv, Uv, Dv, Pv and R = clifford(Atilde_L, z_K)
#

def obtain_nc_hamiltonian_blocks(ptilde, Atilde):

    def Ham(v):
        return sum([ptilde[i](v)*Atilde[i] for i in range(len(Atilde))])
    
    return Ham

def obtain_nc_unitary_blocks(ptilde, Atilde):

    Ham = obtain_nc_hamiltonian_blocks(ptilde, Atilde)

    def Unit(v):
        Hv  = Ham(v)
        Hv -= Hv.constant
        Hv.compress()
        _, _, Uv = obtain_SO_solving_unitary(Hv)
        return Uv
    
    return Unit

def obtain_nc_diagonal_blocks(ptilde, Atilde, K):

    def Diag(v):
        L = len(Atilde) - 1
        return ptilde[0](v) + np.linalg.norm([ptilde[i](v) for i in range(1, L + 1)]) * Q(f'Z{K}')
    
    return Diag

def obtain_nc_hamiltonian_eigenvalues(ptilde, K):

    def E_H(v, w):
        L = len(ptilde) - 1
        return ptilde[0](v) + np.linalg.norm([ptilde[i](v) for i in range(1, L + 1)]) * w
    
    return E_H

def obtain_nc_hamiltonian_blocks_rotation_and_eigenvalues(ptilde, Atilde, K):
    Ham  = obtain_nc_hamiltonian_blocks(ptilde, Atilde)
    Unit = obtain_nc_unitary_blocks(ptilde, Atilde)
    Diag = obtain_nc_diagonal_blocks(ptilde, Atilde, K)
    R    = solve_multiple_independent_commuting_generators([Atilde[-1]], start_idx=K)
    E_H  = obtain_nc_hamiltonian_eigenvalues(ptilde, K)

    return Ham, Unit, Diag, R, E_H

def decimal_to_parity_list(n, K):
    return 1 - 2 * np.array(decimal_to_binary_list(n, K))

def evaluate_block_sum_over_binaries(func, K):
    total_operator = Q().zero()

    for n in range(2 ** K):
        v               = decimal_to_parity_list(n, K)
        total_operator += func(v) * z_proj(v)

    return total_operator

def obtain_product_form_of_unit(Unit):

    def Unit_product(v):
        return compute_product_of_unitaries(Unit(v))
    
    return Unit_product