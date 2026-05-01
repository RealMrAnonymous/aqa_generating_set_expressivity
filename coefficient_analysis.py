from definitions import *
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


def main():
    device = qml.device("default.qubit", wires=N_QUBITS)
    circuit = qml.QNode(optimal_spectrum_diagonal_circuit, device)

    # show_random_plots(circuit, n_params=6*N_QUBITS, n_points=1000)

    fourier_spread = get_fourier_spread(circuit, n_params=6*N_QUBITS)
    fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2, 3, sharex=True, sharey=True)
    axs = [ax0, ax1, ax2, ax3, ax4, ax5]
    for idx in range(6):
        if idx == 0: axs[idx].scatter(fourier_spread[:,16].real, fourier_spread[:,16].imag)
        if idx == 1: axs[idx].scatter(fourier_spread[:,20].real, fourier_spread[:,20].imag)
        if idx == 2: axs[idx].scatter(fourier_spread[:,24].real, fourier_spread[:,24].imag)
        if idx == 3: axs[idx].scatter(fourier_spread[:,26].real, fourier_spread[:,26].imag)
        if idx == 4: axs[idx].scatter(fourier_spread[:,27].real, fourier_spread[:,27].imag)
        if idx == 5: axs[idx].scatter(fourier_spread[:,29].real, fourier_spread[:,29].imag)
        axs[idx].set_xlim(-1, 1)
        axs[idx].set_ylim(-1, 1)

    plt.show()


if __name__ == '__main__':
    main()
