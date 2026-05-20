import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import factorial
from typing import Callable


def generate_random_fourier_series(max_freq: int, rng: np.random.Generator) -> Callable:
    """
    Generates a real function as a truncated fourier series with coefficients in the complex unit discs.

    :param max_freq: the highest frequency in the series
    :param rng: random number generator
    :return: callable real->real function
    """
    assert max_freq >= 0, "max_freq must be a non-negative integer"

    coefficients = np.zeros(2*max_freq+1, dtype=complex) # also include negative frequencies
    coefficients[0] = rng.uniform(-1, 1)
    for idx in range(1, max_freq+1):
        # to sample uniformly within the unit disc, take the square root of the radius
        coefficients[idx] = np.sqrt(rng.uniform(0,1)) * np.exp(1j*rng.uniform(0,2*np.pi))
        coefficients[-idx] = np.conjugate(coefficients[idx])

    freqs = np.concatenate([
        np.arange(0, max_freq + 1, 1, dtype=int),
        np.arange(-max_freq, 0, 1, dtype=int)
    ])

    def truncated_series(x: float | np.ndarray) -> float | np.ndarray:
        if type(x) is float:
            return np.real(np.sum(coefficients * np.exp(1j*x*freqs)))
        elif type(x) is np.ndarray:
            terms = coefficients * np.exp(1j * np.kron(x, freqs).reshape(*x.shape, freqs.shape[0]))
            return np.real(np.sum(terms, axis=-1))
        else:
            raise TypeError(f"input must be of type float or np.ndarray, but has type {type(x)}")

    return truncated_series


# def generate_random_polynomial(degree: int, rng: np.random.Generator) -> Callable:
#     # the range is arbitrary
#     coefficients = rng.uniform(-1, 1, size=degree+1)
#
#     def polynomial(x: float | np.ndarray) -> float | np.ndarray:
#         xs = np.repeat(x, degree+1).reshape(*x.shape, degree+1)
#         powers = np.tile(np.arange(degree+1), (*x.shape, 1))
#         return np.sum(coefficients * np.power(xs, powers), axis=1)
#
#     return polynomial


def generate_random_polynomial(degree: int, rng: np.random.Generator) -> Callable:
    """
    Generate a random polynomial by generating [degree] roots on the interval [0,2pi].
    Doesn't work very well for degrees higher than 3 since the values at the edges of the interval explode.

    :param degree: number of roots, i.e. the degree of the polynomial
    :param rng: random number generator
    :return: callable real->real function
    """
    assert degree > 0, "degree must be a non-negative integer"

    # generate random roots on the interval [0, 2pi]
    roots = rng.uniform(0, 2*np.pi, size=degree)
    # generate a leading coefficient
    coeff = rng.uniform(-1, 1)
    poly = Polynomial.fromroots(roots) * coeff

    def polynomial(x: float | np.ndarray) -> float | np.ndarray:
        return poly(x)

    return polynomial


def generate_exp_cos(lamb: float, kap: float) -> Callable:
    """
    Generate an instance of the function exp(-lamb*kap*x) * cos(lamb*x).
    Values kap=0.05 and lamb=8, or kap=0.02 and lamb=20 work well.

    :return: callable real->real function
    """
    def exp_cos(x: float | np.ndarray) -> float | np.ndarray:
        return np.exp(-kap*lamb*x) * np.cos(lamb*x)
    return exp_cos


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    rng = np.random.default_rng(42)
    # series = generate_random_fourier_series(5, rng)
    exp_cos = generate_exp_cos(3, 0.05)

    xs = np.linspace(0, 2*np.pi, 1000)
    # plt.plot(xs, series(xs))
    plt.plot(xs, exp_cos(xs))

    # for i in range(5):
    #     poly = generate_random_polynomial(3, rng)
    #     plt.plot(xs, poly(xs))

    plt.grid()
    plt.show()
