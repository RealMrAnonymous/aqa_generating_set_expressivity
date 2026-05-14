import matplotlib.pyplot as plt
from tqdm import tqdm

from quantum import *
from random_functions import *


RNG = qml.numpy.random.default_rng(42)


def mse_loss(circuit: qml.Qnode, theta: np.ndarray, collocation_points: np.ndarray, target: Callable) -> float:
    """
    Calculates the mean squared error between a parametrised circuit and a target function at a set of collocation points
    """
    return qml.numpy.mean((circuit(collocation_points, theta) - target(collocation_points))**2)


def fit_one_model(
        target: Callable,
        circuit: qml.QNode,
        n_collocation_points: int,
        n_params: int=6*N_QUBITS,
        n_steps: int=1000,
        stepsize: float=0.01,
        progress_bar: bool=True,
) -> tuple:
    """
    Fits a parametrised circuit to a target function.
    Optimisation is done with the given parameters and at *n_collocation_points* number of collocation points.

    :return: list of *n_steps* loss values and array of *n_params* final parameter values
    """
    collocation_points = np.linspace(0, 2*np.pi, n_collocation_points)

    def cost(theta: np.ndarray):
        return mse_loss(circuit, theta, collocation_points, target)

    optimiser = qml.AdamOptimizer(stepsize=stepsize)
    theta = RNG.uniform(0, 2*np.pi, size=n_params, requires_grad=True)

    loss_values = np.zeros(n_steps, dtype=float)
    iterator = tqdm(range(n_steps), total=n_steps) if progress_bar else range(n_steps)
    for step in iterator:
        theta, loss_values[step] = optimiser.step_and_cost(cost, theta)

    return loss_values, theta


def main():
    device = qml.device('default.qubit', wires=N_QUBITS)
    target = generate_random_fourier_series(3, RNG)
    circuit = qml.QNode(triple_pauli_z_circuit, device)

    loss_values, theta = fit_one_model(target, circuit, 10)
    print(f"Final loss value: {loss_values[-1]:.2e}")

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.semilogy(loss_values)
    ax1.set_title("Loss values per training step")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Loss value")
    ax1.grid(which='both')

    xs = np.linspace(0, 2*np.pi, 1000)
    print(f"Generalisation loss: {np.sqrt(mse_loss(circuit, theta, xs, target)):.2e}")
    collocation_points = np.linspace(0, 2*np.pi, 10)

    ax2.plot(xs, target(xs), color='blue', label='target')
    ax2.scatter(collocation_points, target(collocation_points), color='blue')
    ax2.plot(xs, circuit(xs, theta), color='orange', label='fit')
    ax2.scatter(collocation_points, circuit(collocation_points, theta), color='orange')
    ax2.set_title("Target and fit")
    ax2.legend()
    ax2.grid()

    fig.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
