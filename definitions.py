import pennylane as qml
import numpy as np
from typing import Callable


N_QUBITS = 3


def observable() -> qml.Operation:
    """
    Defines the observable that are used in the variational circuits, which is the total magnetisation, i.e. the sum of Pauli Zs on each qubit.

    :return: Total magnetisation operator
    """
    # disable inspection since technically, the below sum can be empty, in which case the type is not inferred, but this
    # will never happen
    # noinspection PyTypeChecker
    return sum([qml.PauliZ(i) for i in range(N_QUBITS)])


def variational_layer(theta: np.ndarray):
    """
    Apply one variational layer, which is one instance of Circuit 19 of https://arxiv.org/pdf/1905.10876.

    :param theta: Array containing the parameters for this layer.
    Must have length 3*N_QUBITS.
    """
    assert theta.shape == (3*N_QUBITS,), f"variational layer requires shape ({3*N_QUBITS},), found shape {theta.shape}"

    # individual X and Z rotations per qubit
    # the parameters for the Z rotations come after those for the X rotations
    for idx in range(N_QUBITS):
        qml.RX(theta[idx], wires=idx)
        qml.RZ(theta[N_QUBITS+idx], wires=idx)

    # entangling controlled rotation gates, in a cyclic cascade
    qml.CRX(theta[2*N_QUBITS], wires=[N_QUBITS-1, 0])
    for idx in range(1, N_QUBITS):
        qml.CRX(theta[2*N_QUBITS+idx], wires=[N_QUBITS-idx-1, N_QUBITS-idx])


def single_pauli_z_circuit(x: float, theta: np.ndarray) -> qml.measurements.ExpectationMP:
    """
    PQC with single-qubit Z rotation as encoding gate
    """
    variational_layer(theta[:3*N_QUBITS])
    qml.RZ(x, wires=0)
    variational_layer(theta[3*N_QUBITS:])
    return qml.expval(observable())


def single_pauli_x(x: float):
    """
    Single controlled X rotation on qubit 0. Can be generalised to any N_QUBITS.
    """
    qml.RX(x, wires=0)


def triple_pauli_z_circuit(x: float, theta: np.ndarray) -> qml.measurements.ExpectationMP:
    """
    PQC with triple-qubit Z rotation as encoding gate
    """
    variational_layer(theta[:3*N_QUBITS])
    qml.RZ(x, wires=0)
    qml.RZ(x, wires=1)
    qml.RZ(x, wires=2)
    variational_layer(theta[:3*N_QUBITS])
    return qml.expval(observable())


def triple_pauli_mixed(x: float):
    """
    Controlled mixed rotations on the first 3 qubits. Cannot be generalised.
    """
    qml.RX(x, wires=0)
    qml.RY(x, wires=1)
    qml.RZ(x, wires=2)


def optimal_spectrum_diagonal_circuit(x: float, theta: np.ndarray) -> qml.measurements.ExpectationMP:
    """
    Applies a rotation on the first 3 qubits based on a diagonal Hermitian matrix, whose spectrum is engineered so that the set of differences of eigenvalues is a maximal set.
    See Golomb ruler on Wikipedia.

    :param x: the rotation angle
    """
    variational_layer(theta[:3*N_QUBITS])
    matrix = np.diag([0,1,4,9,15,22,32,34])
    # disable inspection below because "coeff" is annotated as float even though it can canonically be complex
    # noinspection PyTypeChecker
    qml.exp(qml.Hermitian(matrix, [0,1,2]), coeff=1j*x)
    variational_layer(theta[3*N_QUBITS:])
    return qml.expval(observable())


def generate_random_instance(
        circuit_function: Callable,
        device: qml.Device,
        n_params: int=6*N_QUBITS,
        seed: int=42
) -> Callable:
    rng = np.random.default_rng(seed)
    params = rng.random(n_params)
    circuit = qml.QNode(circuit_function, device)

    def circuit_function(x: float) -> float:
        return circuit(x, params)

    return circuit_function
