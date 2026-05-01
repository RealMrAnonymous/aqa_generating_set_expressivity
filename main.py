import matplotlib.pyplot as plt

from definitions import *


def main():
    device = qml.device('default.qubit', wires=N_QUBITS)
    test = generate_random_instance(optimal_spectrum_diagonal_circuit, device=device)

    xs = np.linspace(0, 2*np.pi, 100)
    plt.plot(xs, test(xs))
    plt.show()


if __name__ == "__main__":
    main()
