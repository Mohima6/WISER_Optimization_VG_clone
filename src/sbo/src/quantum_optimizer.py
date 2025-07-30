import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import Pauli
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.opflow import SummedOp

def build_quadratic_operator(Q, offset=0):
    """Convert quadratic matrix Q (and offset) to a PauliSumOp suitable for VQE.
    
    This is a toy example for demonstration only. In practice, you need to
    encode your quadratic objective into a Hamiltonian (Pauli operators).
    
    Args:
        Q (np.ndarray): symmetric matrix of quadratic terms for binary variables.
        offset (float): constant term in objective.
        
    Returns:
        PauliSumOp: operator representing the quadratic objective.
    """
    n = Q.shape[0]
    # For demo, we encode binary variables x_i ∈ {0,1} as qubit operators:
    # x_i = (1 - Z_i)/2, where Z_i is Pauli-Z on qubit i.
    # Then QUBO Q can be converted to Ising Hamiltonian with Pauli Z operators.
    pauli_ops = []
    for i in range(n):
        for j in range(n):
            coeff = Q[i, j] / 4
            if i == j:
                # Diagonal terms (x_i^2 = x_i)
                # QUBO: Q_ii x_i => coeff * (1 - Z_i)
                pauli_ops.append(PauliOp(Pauli((False,) * i + (True,) + (False,) * (n - i - 1)), -coeff))
                pauli_ops.append(PauliOp(Pauli.from_label('I' * n), coeff))
            elif i < j:
                # Off-diagonal terms: Q_ij x_i x_j
                # = Q_ij * (1 - Z_i - Z_j + Z_i Z_j)/4
                z_i = [False] * n
                z_i[i] = True
                z_j = [False] * n
                z_j[j] = True
                z_ij = [False] * n
                z_ij[i] = True
                z_ij[j] = True
                
                pauli_ops.append(PauliOp(Pauli.from_label('I' * n), coeff))
                pauli_ops.append(PauliOp(Pauli(z_i), -coeff))
                pauli_ops.append(PauliOp(Pauli(z_j), -coeff))
                pauli_ops.append(PauliOp(Pauli(z_ij), coeff))
    if offset != 0:
        pauli_ops.append(PauliOp(Pauli.from_label('I' * n), offset))
    
    return SummedOp(pauli_ops).reduce()

def run_vqe(Q, offset=0, maxiter=100):
    """Run VQE to minimize quadratic objective represented by matrix Q."""
    n = Q.shape[0]

    # Build operator for Hamiltonian
    operator = build_quadratic_operator(Q, offset)

    # Choose backend
    backend = Aer.get_backend('aer_simulator_statevector')
    quantum_instance = QuantumInstance(backend)

    # Variational form (ansatz)
    ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz', reps=3, entanglement='full', insert_barriers=True)

    # Optimizer
    optimizer = COBYLA(maxiter=maxiter)

    # Setup VQE
    vqe = VQE(ansatz, optimizer=optimizer, quantum_instance=quantum_instance)

    # Run VQE
    result = vqe.compute_minimum_eigenvalue(operator)
    print("Minimum eigenvalue (approx):", result.eigenvalue.real)
    print("Optimal parameters:", result.optimal_point)
    
    #  interpret result.optimal_point to find best binary vector (depends on ansatz)
    return result

if __name__ == "__main__":
    # Example Q matrix (QUBO) for 3 variables (simple portfolio toy problem)
    Q = np.array([[1, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 1]])
    offset = 0.0

    print("Running VQE on toy quadratic objective...")
    result = run_vqe(Q, offset)

