import numpy as np
from openfermion import QubitOperator as Q
from numpy.random import uniform
from numpy import sin, cos
from numpy import arctan2 as atan2
from utils_basic import (
    is_anticommuting, 
    random_pauli_term, 
    clifford
)

def is_ac_hamiltonian(H):
    for t, _ in H.terms.items():
        for s, _ in H.terms.items():
            if (t != s) and (not is_anticommuting(t, s)) and (Q(t) != Q("")) and (Q(s) != Q("")):
                return False
    return True

def ac_inclusion_criterion(H, A):
    for t, _ in H.terms.items():
        if (not is_anticommuting(t, A)):
            return False
    return True

def random_ac_hamiltonian(Nqubits, Nterms):
    assert Nterms <= (2 * Nqubits - 1)

    H = Q().zero()

    while len(H.terms) < Nterms:

        _, A = random_pauli_term(Nqubits)
        if ac_inclusion_criterion(H, A):
            H += uniform(-2, 2) * A

    assert is_ac_hamiltonian(H)
    return H

def obtain_SO_solving_unitary(H):

    assert is_ac_hamiltonian(H)

    c = []
    A = []

    for t, s in H.terms.items():
        c.append(s)
        A.append(Q(t))

    L = len(A)

    if (L == 1):
        return [0.0], [Q()], [Q("")]

    theta0 = -atan2(c[0], c[-1])
    G0     = -theta0 * A[-1] * A[0] / 2
    U0     = cos(theta0/2) - sin(theta0/2) * A[-1] * A[0]

    theta_list = [theta0]
    G_list     = [G0]
    U_list     = [U0]

    for j in range(1, L - 1):
        second_arg = np.sqrt(c[-1]**2 + sum([c[k]**2 for k in range(j)]))
        thetaj     = -atan2(c[j], second_arg)
        Gj         = -thetaj * A[-1] * A[j] / 2
        Uj         = cos(thetaj/2) - sin(thetaj/2) * A[-1] * A[j]

        theta_list.append(thetaj)
        G_list.append(Gj)
        U_list.append(Uj)

    return theta_list, G_list, U_list

def map_pauli_to_z0(P):
    P.compress()
    P = Q(list(P.terms.keys())[0])

    if P == Q('Z0'):
        return [Q("")]
    
    elif (P == Q('X0')) or (P == Q('Y0')):
        return [clifford(Q('Z0'), P)]
    
    else:
        Pterm     = list(P.terms.keys())[0]
        op        = Pterm[0]
        i, letter = op[0], op[1] 

        if letter == 'Z':
            ac_op = Q(f'X{i}')
        
        elif letter in ['X', 'Y']:
            ac_op = Q(f'Z{i}')

        U_list = [clifford(P, ac_op)]
        
        if ac_op == Q('Z0'):
            return U_list
        
        elif ac_op == Q('X0'):
            x_to_z = clifford(Q('X0'), Q('Z0'))
            U_list.append(x_to_z)
            return U_list
        
        else:
            if list(ac_op.terms.keys())[0][0][1] == 'X':
                pairing_U = clifford(
                    ac_op, Q(f'X{0} Z{i}')
                )
                x_to_z = clifford(
                    Q('Z0'), Q(f'X{0} Z{i}')
                )
                U_list.append(pairing_U)
                U_list.append(x_to_z)
                return U_list

            elif list(ac_op.terms.keys())[0][0][1] == 'Z':
                pairing_U = clifford(
                    ac_op, Q(f'X{0} X{i}')
                )
                x_to_z = clifford(
                    Q('Z0'), Q(f'X{0} X{i}')
                )
                U_list.append(pairing_U)
                U_list.append(x_to_z)
                return U_list
            
def ac_hamiltonian_solver(H):

    assert is_ac_hamiltonian(H)

    c = []
    A = []

    for t, s in H.terms.items():
        c.append(s)
        A.append(Q(t))

    _, _, SO_unitary = obtain_SO_solving_unitary(H)
    clifford_unitary = map_pauli_to_z0(A[-1])

    return SO_unitary + clifford_unitary