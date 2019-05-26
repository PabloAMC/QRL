from qiskit.aqua.input import LinearSystemInput
from qiskit.quantum_info import state_fidelity
from qiskit.aqua.algorithms.classical import ExactLSsolver
import numpy as np

from qiskit import BasicAer, QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms.single_sample import HHL

def build_circuit(matrix, vector_u, vector_p):
    n_io = int(np.log2(np.size(vector_u)))
    # vector_p /= np.linalg.norm(vector_p)
    # Parameters for HHL algorithm
    num_ancillae = 3
    num_time_slices = 50
    params = dict()
    params['problem'] = {
        'name': 'linear_system'
    }
    params['backend'] = {
        'provider': 'qiskit.BasicAer',
        'name': 'statevector_simulator'
    }
    params['algorithm'] = {
        'truncate_powerdim': False,
        'truncate_hermitian': False
    }
    params['reciprocal'] = {
        'name': 'Lookup',
        'negative_evals': True
    }
    params['eigs'] = {
        'expansion_mode': 'suzuki',
        'expansion_order': 2,
        'name': 'EigsQPE',
        'negative_evals': True,
        'num_ancillae': num_ancillae,
        'num_time_slices': num_time_slices
    }
    params['initial_state'] = {
        'name': 'CUSTOM'
    }
    params['iqft'] = {
        'name': 'STANDARD'
    }
    params['qft'] = {
        'name': 'STANDARD'
    }
    algo_input = LinearSystemInput(matrix=matrix, vector=vector_u)
    hhl = HHL.init_params(params, algo_input)
    # Quantum circuit for HHL
    qc = hhl.construct_circuit()
    # HHL solution output register and success bit register
    io = hhl._io_register
    anc1 = hhl._ancilla_register
    # Quantum registers for Swap test
    psas = QuantumRegister(n_io, 'psas')
    anc2 = QuantumRegister(1, 'anc2')
    c1 = ClassicalRegister(1, 'c1')
    c2 = ClassicalRegister(1, 'c2')
    # Add registers to quantum circuit
    qc.add_register(psas, anc2, c1, c2)
    # Initialize vector_p on psas register
    qc.initialize(vector_p, psas)
    # exit(0)
    # Swap test: control swap for dot product
    qc.h(anc2)
    for i in range(n_io):
        qc.cswap(anc2, psas[i], io[i])
    qc.h(anc2)
    # Projection and meassurement
    qc.barrier(anc1)
    qc.barrier(anc2)
    qc.measure(anc1, c1)
    qc.measure(anc2, c2)
    return qc

def main(matrix, vector_u, vector_p):
    assert np.abs(np.linalg.norm(vector_u))-1<1e-10 # assert vector_u is normalized
    assert np.abs(np.linalg.norm(vector_p))-1<1e-10 # assert vector_p is normalized
    kappa = np.linalg.cond(matrix)
    shots = 8192
    qc = build_circuit(matrix, vector_u, vector_p)

    backend_qasm = BasicAer.get_backend('qasm_simulator')
    job_qasm = execute(qc, backend_qasm, shots=shots)
    result_qasm = job_qasm.result()
    counts = result_qasm.get_counts(qc)

    # backend_state=BasicAer.get_backend('statevector_simulator')
    # job_state=execute(qc,backend_state)
    # result_state=job_state.result()
    # outputstate=result_state.get_statevector(qc,decimals=3)
    # print(outputstate)
    # import pdb; pdb.set_trace()
    # return

    error = 0
    success = 0
    for key, value in counts.items():
        k = key.split()
        if int(k[1]) == 0:
            error += value
        elif int(k[0]) == 0:
            success += value
    # prob1 = 1 - error/shots # Success probability in HHL anc1
    prob2 = 1 - success/(shots-error) # Probability of failure of anc2
    if prob2 > 0.5: prob2 = .5
    dot_product = np.sqrt(1-2*prob2) #*np.linalg.norm(vector_p) # kappa*np.sqrt(prob1)
    return dot_product, counts

def classical_solver(matrix, vector_u, vector_p):
    A_inv = np.linalg.inv(np.matrix(matrix))
    # vector_u /= np.linalg.norm(vector_u)
    # vector_p /= np.linalg.norm(vector_p)
    x = A_inv*np.matrix(vector_u).T
    x /= np.linalg.norm(x)
    kappa = np.linalg.cond(matrix)
    dot_product = x.T*np.matrix(vector_p).T
    return dot_product, x, kappa

matrix = [[3, 1], [2, 1]]
vector_u = [-1, 3]
vector_p = [-1, -1]

# matrix = [[1.5, 0.5], [0.5, 1.5]]
# vector_u = [1, -1]/np.sqrt(2)
# vector_p = [1, 1]/np.sqrt(2)

vector_u/=np.linalg.norm(vector_u)
vector_p/=np.linalg.norm(vector_p)

sol = classical_solver(matrix, vector_u, vector_p)
print(sol)
sol=main(matrix, vector_u, vector_p)
print(sol)
