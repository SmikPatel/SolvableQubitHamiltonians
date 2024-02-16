import numpy as np
import networkx as nx
from numpy.random import uniform
from openfermion import QubitOperator as Q
from openfermion import MajoranaOperator as M
from openfermion import jordan_wigner, get_fermion_operator
from utils_la import (
    SO_real_schur,
    obtain_orthogonal_generator
)
from utils_basic import (
    is_anticommuting,
    close_qubit_hamiltonian_algebra,
    apply_unitary_product,
    copy_hamiltonian,
    random_pauli_term,
    obtain_ac_graph
)
from utils_fc import (
    obtain_generators,
    obtain_polynomial_representation_of_fc_hamiltonian,
    solve_multiple_independent_commuting_generators
)
from utils_nc import (
    obtain_z_string
)

def is_line_graph(G):
    try:
        R = nx.inverse_line_graph(G)
        return True
    except:
        return False

def is_ff_hamiltonian(H):
    AC = obtain_ac_graph(H)
    return is_line_graph(AC)

def ff_inclusion_criterion(H, A):
    if isinstance(A, tuple):
        A = Q(A)

    Hcopy  = copy_hamiltonian(H)
    Hcopy += A
    return is_ff_hamiltonian(Hcopy)

def random_ff_hamiltonian(Nqubits, Nterms):

    H = Q()

    while len(H.terms) < Nterms:

        _, A = random_pauli_term(Nqubits)
        if ff_inclusion_criterion(H, A):
            H += uniform(-2, 2) * A

    assert is_ff_hamiltonian(H)
    return H

#
#    obtain root graph of line graph functions
#

K3 = nx.complete_graph(3)

def K3_correct_root(G):
    """
    G is a graph isomorphic to K3
    return is root graph R of G that is isomorphic to G

    this function is needed since nx.inverse_line_graph will give the claw graph which is 
    not what I want 
    """
    Gnodes = list(G.nodes)
    R      = nx.Graph()
    R.add_nodes_from([
        (Gnodes[0], Gnodes[1]),
        (Gnodes[0], Gnodes[2]),
        (Gnodes[1], Gnodes[2])
    ])
    return nx.complete_graph(R)

def obtain_root_graph(G):
    if len(G.nodes) == 3 and nx.is_isomorphic(G, K3):
        return K3_correct_root(G)
    else:
        return nx.inverse_line_graph(G)

#
#    Homogeneous-quadratic Majorana solver
#

def random_majorana_ff_hamiltonian(N):
    """
    N is the number of Majorana modes
    return is a random homogeneous quadratic Hamiltonian over N Majorana modes
    """
    H = M()
    for i in range(N):
        for j in range(i+1, N):
            H += 1j * uniform(-2, 2) * M((i,j))
    return H

def obtain_ff_coef_matrix(H, N):
    h = np.zeros([N,N])
    for term, coef in H.terms.items():
        h[term[0], term[1]] =  np.imag(coef)
        h[term[1], term[0]] = -np.imag(coef)
    return h

def obtain_ff_solver(H, N):
    """
    return 
        1. diagonalized single particle Hamiltonian from which eigenvalues can be derived
        2. special orthogonal single-particle-basis transformation
        3. and MajoranaOperator form of generator of diagonalizing unitary over the Fock space
    """
    h      = obtain_ff_coef_matrix(H, N)
    lam, o = SO_real_schur(h)
    x      = obtain_orthogonal_generator(o)

    X = M()
    for i in range(N):
        for j in range(i+1, N):
            X += 0.5 * x[i,j] * M((i,j))

    return lam, o, X

#
#    Qubit-ff solvers ignoring cycle-symmetries
#

def obtain_fundamental_cycles(G):
    T           = nx.minimum_spanning_tree(G)
    r           = list(T.nodes)[0]
    paths       = nx.single_source_shortest_path(T, r)
    non_T_edges = list(set(G.edges) - set(T.edges))

    to_paths   = {}
    from_paths = {}
    cycles     = {}

    for node in T.nodes:

        if node == r:
            to_paths[node]   = []
            from_paths[node] = []

        else:
            
            to_path    = []
            path_nodes = paths[node]
            for i in range(len(path_nodes) - 1):
                to_path.append( (path_nodes[i], path_nodes[i+1]) )
            to_paths[node] = to_path

            from_path  = []
            path_nodes = paths[node][::-1]
            for i in range(len(path_nodes) - 1):
                from_path.append( (path_nodes[i+1], path_nodes[i]) )
            from_paths[node] = from_path

    for edge in non_T_edges:
        cycles[edge] = to_paths[edge[0]] + [edge] + from_paths[edge[1]]

    return T, non_T_edges, cycles

def edge_intersection(edge):
    return list(set.intersection(set(edge[0]), set(edge[1])))[0]

def obtain_node_to_index_dictionary(G):
    D = dict()
    for i, node in enumerate(G.nodes):
        D[node] = i
    return D

def obtain_index_to_node_dictionary(G):
    D = dict()
    for i, node in enumerate(G.nodes):
        D[i] = node
    return D

def obtain_edge_to_index_pair_dictionary(G):
    node2index = obtain_node_to_index_dictionary(G)
    D          = dict()
    for edge in G.edges:
        i1 = node2index[edge[0]]
        i2 = node2index[edge[1]]
        D[edge] = (i1, i2)
    return D

def obtain_index_pair_to_edge_dictionary(G):
    node2index = obtain_node_to_index_dictionary(G)
    D          = dict()
    for edge in G.edges:
        i1 = node2index[edge[0]]
        i2 = node2index[edge[1]]
        D[(i1, i2)] = edge
    return D

def obtain_operator_to_index_pair_dictionary(G):
    edge2index = obtain_edge_to_index_pair_dictionary(G)
    D          = dict()
    for edge, index_pair in edge2index.items():
        op    = edge_intersection(edge)
        D[op] = index_pair 
    return D

def obtain_index_pair_to_operator_dictionary(G):
    edge2index = obtain_edge_to_index_pair_dictionary(G)
    D          = dict()
    for edge, index_pair in edge2index.items():
        op = edge_intersection(edge)
        D[index_pair] = op
    return D

def obtain_edge_to_operator_dictionary(G):
    edge2index = obtain_edge_to_index_pair_dictionary(G)
    index2op   = obtain_index_pair_to_operator_dictionary(G)
    D          = dict()
    for edge, index_pair in edge2index.items():
        op      = index2op[index_pair]
        D[edge] = op
    return D

def obtain_operator_to_edge_dictionary(G):
    op2index   = obtain_operator_to_index_pair_dictionary(G) 
    index2edge = obtain_index_pair_to_edge_dictionary(G)
    D          = dict()
    for op, index_pair in op2index.items():
        edge  = index2edge[index_pair]
        D[op] = edge
    return D

def obtain_nonTedge_to_cycleopQ_dictionary(G):
    _, _, cycles = obtain_fundamental_cycles(G)
    edge2op      = obtain_edge_to_operator_dictionary(G)

    D = dict()
    for nonTedge, edgelist in cycles.items():
        Op = Q("")
        for edge in edgelist:
            Op = Op * Q(edge2op[edge])
        Op.compress()
        D[nonTedge] = Op
    return D

def obtain_nonTedge_to_cycleopM_dictionary(G):
    _, _, cycles = obtain_fundamental_cycles(G)
    edge2index   = obtain_edge_to_index_pair_dictionary(G)

    D = dict()
    for nonTedge, edgelist in cycles.items():
        Op = M("")
        for edge in edgelist:
            Op = Op * 1j * M(edge2index[edge])
        D[nonTedge] = Op
    return D

def obtain_correct_op2index_ordering(op2index, edge2op, cycleQ, cycleM):
    op2indexC = dict(op2index)
    for edge in cycleQ:
        qubitized_M_op = jordan_wigner(get_fermion_operator(cycleM[edge]))
        if cycleQ[edge] != qubitized_M_op:
            op2indexC[edge2op[edge]] = op2indexC[edge2op[edge]][::-1]
    return op2indexC

def obtain_all_root_graph_dictionaries(G):
    d1  = obtain_node_to_index_dictionary(G)
    d2  = obtain_index_to_node_dictionary(G)
    d3  = obtain_edge_to_index_pair_dictionary(G)
    d4  = obtain_index_pair_to_edge_dictionary(G)
    d5  = obtain_operator_to_index_pair_dictionary(G)
    d6  = obtain_index_pair_to_operator_dictionary(G)
    d7  = obtain_edge_to_operator_dictionary(G)
    d8  = obtain_operator_to_edge_dictionary(G)
    d9  = obtain_nonTedge_to_cycleopQ_dictionary(G)
    d10 = obtain_nonTedge_to_cycleopM_dictionary(G)
    d11 = obtain_correct_op2index_ordering(d5, d7, d9, d10)
    return d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11

def obtain_fermionized_hamiltonian(H, N):
    G         = obtain_ac_graph(H)
    R         = obtain_root_graph(G)
    op2indexC = obtain_all_root_graph_dictionaries(R)[-1]

    Hm = M()
    for op, index_pair in op2indexC.items():
        Hm += H.terms[op] * 1j * M(index_pair)
    return Hm

def solve_ff_qubit_hamiltonian(H):
    H         = close_qubit_hamiltonian_algebra(H)
    G         = obtain_ac_graph(H)
    R         = obtain_root_graph(G)
    Nmodes    = len(R.nodes) 
    op2indexC = obtain_all_root_graph_dictionaries(R)[-1]

    Hm = M()
    for op, index_pair in op2indexC.items():
        Hm += 1j * H.terms[op] * M(index_pair)

    _, _, X = obtain_ff_solver(Hm, Nmodes)

    Xq = Q()
    for op, index_pair in op2indexC.items():
        if index_pair[0] < index_pair[1]:
            Xq += -1j * X.terms[index_pair] * Q(op)
        else:
            Xq -= -1j * X.terms[index_pair[::-1]] * Q(op)
    return Xq

#
#    functions for factorizing cycle symmetries
#

def obtain_cycle_symmetries(H):
    AC     = obtain_ac_graph(H)
    R      = obtain_root_graph(AC)
    cycleD = obtain_all_root_graph_dictionaries(R)[-3]

    cycle_symmetries = []
    for op in cycleD.values():
        op.compress()
        op_oned = Q(list(op.terms.keys())[0])
        if op_oned not in cycle_symmetries and op_oned not in [Q(), Q("")]:
            cycle_symmetries.append(op_oned)

    return cycle_symmetries

def obtain_and_solve_cycle_symmetries(H, N):
    AC     = obtain_ac_graph(H)
    R      = obtain_root_graph(AC)
    cycleD = obtain_all_root_graph_dictionaries(R)[-3]

    cycle_symmetries = []
    for op in cycleD.values():
        op.compress()
        op_oned = Q(list(op.terms.keys())[0])
        if op_oned not in cycle_symmetries and op_oned not in [Q(), Q("")]:
            cycle_symmetries.append(op_oned)

    cycle_generators = obtain_generators(cycle_symmetries, N)
    Uc               = solve_multiple_independent_commuting_generators(cycle_generators)

    return cycle_symmetries, cycle_generators, Uc

def factorize_cycle_symmetries(H, N):
    """
    returns four objects, analogous to `factorize_nc_hamiltonian` function:
        1. generators of the Lie algebra, obtained by factorizing cycle symmetry prefixes
        2. independent generators of the cycle symmetry group
        3. expression of all cycle symmetry prefixes as polynomial (monomial) of generators
        4. clifford that maps cycle symmetry generators to z0,...,z{K-1}
    """
    cycle_syms, C, Uc = obtain_and_solve_cycle_symmetries(H, N)
    K                 = len(C)
    Hr                = apply_unitary_product(H, Uc)
    z_gens            = [Q(f'Z{k}') for k in range(K)]

    z_strings     = []
    alg_ops_tilde = []
    for term, coef in Hr.terms.items():
        z_string     = obtain_z_string(Q(term), K)
        alg_op_tilde = z_string * Q(term)
        
        z_strings.append(coef * z_string)
        alg_ops_tilde.append(alg_op_tilde)

    p = [
        obtain_polynomial_representation_of_fc_hamiltonian(z_string, N, generators=z_gens)
        for z_string in z_strings
    ]
    alg_ops = [apply_unitary_product(A, Uc[::-1]) for A in alg_ops_tilde]

    return alg_ops, C, p, Uc

def obtain_ff_hamiltonian_blocks(p, Atilde):
    
    def Ham(v):
        return sum([p[i](v)*Atilde[i] for i in range(len(p))])
    
    return Ham

def obtain_ff_generator_blocks(p, Atilde):

    Ham = obtain_ff_hamiltonian_blocks(p, Atilde)

    def Gen(v):
        return solve_ff_qubit_hamiltonian(Ham(v))

    return Gen