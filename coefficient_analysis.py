from quantum import *
import matplotlib.pyplot as plt
from typing import Callable


SEED = 42
RNG = np.random.default_rng(seed=SEED)


def get_fourier_spread(function: Callable, n_params: int, n_points: int=1000, n_iterations: int=100) -> np.ndarray:
    """
    Generates random parameters for a parametrised function, evaluates the function on the interval [0,2pi], and calculates the Fourier series coefficients.

    :param function: real-to-real-valued function with one input and `n_params` parameters (interpreted as angles in the range [0,2pi]), positional
    :param n_params: number of parameters expected by `function`
    :param n_points: number of points to evaluate the function on
    :param n_iterations: number of random parameter initialisations
    :return: array of shape (`n_iterations`, `n_points`) containing the Fourier coefficients per iteration
    """
    xs = np.linspace(0, 2*np.pi, n_points)
    coeffs = np.zeros((n_iterations, n_points), dtype=complex)

    for i in range(n_iterations):
        params = 2*np.pi*RNG.random(n_params) # multiply by 2pi to get uniformly random angles for the rotation gates
        coeffs[i] = np.fft.fft(function(xs, params), norm='forward')

    return coeffs


def show_random_plots(function: Callable, n_params: int, n_points: int=100, n_plots: int=5):
    xs = np.linspace(0, 2*np.pi, n_points)

    for i in range(n_plots):
        params = 2*np.pi*RNG.random(n_params)
        plt.plot(xs, function(xs, params))

    plt.show()


def main(model_name: str, show: bool=True):
    model_dict = {
        "Single Pauli Z": single_pauli_z_circuit,
        "Triple Pauli Z": triple_pauli_z_circuit,
        "Pauli Combination": optimal_pauli_combination_circuit,
        "Optimal Spectrum": optimal_spectrum_diagonal_circuit,
    }

    device = qml.device("default.qubit", wires=N_QUBITS)
    circuit = qml.QNode(model_dict[model_name], device)

    fourier_spread = get_fourier_spread(circuit, n_params=6*N_QUBITS)
    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(6,6.5), layout='constrained')
    axes = [ax for row in axes for ax in row]

    # change which coefficients are plotted in each subplot
    coefficient_map = [0, 1, 2, 16, 24, 33, 34, 35, 36]
    for idx, ax in enumerate(axes):
        idx = coefficient_map[idx]
        ax.scatter(fourier_spread[:,idx].real, fourier_spread[:,idx].imag, s=8)
        ax.set_title(f"Frequency ${f"\\pm{idx}" if idx > 0 else "0"}$")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    fig.supxlabel("Real part")
    fig.supylabel("Imaginary part")
    fig.suptitle(f"Fourier coefficient spread of model {model_name}")

    if show:
        plt.show()
    else:
        plt.savefig(f"plots/Coefficients {model_name}.png", dpi=300)


if __name__ == '__main__':
    model_name = "Optimal Spectrum"
    show = False

    main(model_name, show)
