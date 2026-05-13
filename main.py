import matplotlib.pyplot as plt

from definitions import *

def combinations(a: int, b: int, c: int) -> list:
    return [a*s1 + b*s2 + c*s3 for s1 in (-1,1) for s2 in (-1,1) for s3 in (-1,1)]


def omega(eigen_values: list) -> set:
    result = set()
    for a in eigen_values:
        for b in eigen_values:
            result.add(a-b)
    return result


def main():
    max = 100
    best_size = 0
    best = tuple()
    for a in range(max):
        for b in range(max):
            for c in range(max):
                om = omega(combinations(a, b, c))
                if len(om) > best_size:
                    best_size = len(om)
                    best = (a,b,c)

    print(best_size, best)
    result = [val // 2 for val in omega(combinations(best[0], best[1], best[2]))]
    print(result)



if __name__ == "__main__":
    main()
