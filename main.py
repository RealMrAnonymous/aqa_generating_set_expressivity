import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
import matplotlib.pyplot as plt


N_QUBITS = 3
DEVICE = qml.device("default.qubit", wires=N_QUBITS)


def observable() -> qml.Operation:
    """
    Defines the observable that are used in the variational ciruits, which is the total magnetisation, i.e. the sum of Pauli Zs on each qubit.

    :return:
    Total magnetisation operator
    """
    return sum([qml.PauliZ(i) for i in range(N_QUBITS)], start=qml.I())


def variational_layer(theta: np.ndarray) -> None:
    """
    Apply one variational layer, which is one instance of Circuit 19 of https://arxiv.org/pdf/1905.10876.

    :param theta:
    Array containing the parameters for this layer.
    Must have length 3*N_QUBITS.
    """
    assert theta.shape == (3*N_QUBITS,)

    # individual X and Z rotations per qubit
    # the parameters for the Z rotations come after those for the X rotations
    for idx in range(N_QUBITS):
        qml.RX(theta[idx], wires=idx)
        qml.RZ(theta[N_QUBITS+idx], wires=idx)

    # entangling controlled rotation gates, in a cyclic cascade
    qml.CRX(theta[2*N_QUBITS], wires=[N_QUBITS-1, 0])
    for idx in range(1, N_QUBITS):
        qml.CRX(theta[2*N_QUBITS+idx], wires=[idx-1, idx])


@qml.qnode(DEVICE)
def circuit(theta: np.ndarray):
    variational_layer(theta)
    return qml.expval(observable())


def main():
    fix, ax = qml.draw_mpl(circuit)(np.zeros(3*N_QUBITS))
    plt.show()


if __name__ == "__main__":
    main()
