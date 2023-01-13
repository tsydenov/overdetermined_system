import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import glob
from typing import Callable


def make_matrix(deg: int, X) -> np.ndarray:
    A = np.zeros((len(X), deg + 1))
    for i in range(len(X)):
        for j in range(deg + 1):
            A[i][j] = legendre(j, X[i])
    return A


def legendre(j: int, x: float) -> float:
    if j == 0:
        return 1
    if j == 1:
        return x
    else:
        return (2 * (j - 1) + 1) / j * x * legendre(j - 1, x) - (j - 1) / j * legendre(j - 2, x)


def householder(matrix: np.ndarray, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = matrix.copy()
    b = vector.copy()
    m, n = A.shape
    for j in range(n):
        if np.any(A[j + 1:, j]):
            e = np.zeros(m - j)
            e[0] = 1
            if A[j, j] >= 0:
                u = A[j:, j] + np.linalg.norm(A[j:, j]) * e
            else:
                u = A[j:, j] - np.linalg.norm(A[j:, j]) * e
            u /= np.linalg.norm(u)
            A[j:, j:] -= np.outer(2 * u, u @ A[j:, j:])
            b[j:] -= 2 * u * (u @ b[j:])
    return A, b


def qr_solve(matrix: np.ndarray, vector: np.ndarray, degree: int) -> tuple[np.ndarray, float]:
    start = time.time()
    A, b = householder(matrix, vector)
    # result = np.linalg.solve(A[:degree + 1, :], b[:degree + 1])
    result = np.zeros(degree+1)
    for i in range(degree, -1, -1):
        result[i] = (b[i] - sum(x * a for x, a in zip(result[i+1:], A[i, i+1:]))) / A[i, i]
    end = time.time()
    result_time = end - start
    return result, result_time


def normal_equation(matrix: np.ndarray, vector: np.ndarray) -> tuple[np.ndarray, float]:
    start = time.time()
    result = np.linalg.solve(np.transpose(matrix) @ matrix, np.transpose(matrix) @ vector)
    end = time.time() - start
    return result, end


def closure(coef: np.ndarray, deg: int) -> Callable[[float], float]:
    def f(x: float):
        s = 0
        for i in range(deg + 1):
            s += coef[i] * legendre(i, x)
        return s
    return f


def sme(x: list[float], y: list[float]) -> float:
    return np.sqrt(np.sum(np.square(np.subtract(x, y))) / len(x)) / max(y)


def obtain(data: str, degree: int) -> None:
    print(f'{data}:')

    X, Y = np.loadtxt(data, unpack=True)

    cond_matrix = []
    cond_ne_matrix = []
    sme_qr = []
    sme_ne = []
    qr_time_list = []
    ne_time_list = []
    qr_approx_list = []
    ne_approx_list = []

    for deg in range(degree):
        matrix = make_matrix(deg, X)
        qr_coefficients, qr_time = qr_solve(matrix, Y, deg)
        ne_coefficients, ne_time = normal_equation(matrix, Y)

        # приближенные значения
        qr_approx = list(map(closure(qr_coefficients, deg), X))
        ne_approx = list(map(closure(ne_coefficients, deg), X))
        qr_approx_list.append(qr_approx)
        ne_approx_list.append(ne_approx)

        sme_qr.append(sme(qr_approx, Y))
        sme_ne.append(sme(ne_approx, Y))

        qr_time_list.append(qr_time)
        ne_time_list.append(ne_time)

        cond_matrix.append(np.linalg.cond(matrix))
        cond_ne_matrix.append(np.linalg.cond(np.transpose(matrix) @ matrix))

    dt = pd.DataFrame(({'cond(AT*A)': cond_ne_matrix,
                        'SME (НУ)': sme_ne,
                        'время, с (НУ)': ne_time_list,
                        'cond(A)': cond_matrix,
                        'SME (QR)': sme_qr,
                        'время, с (QR)': qr_time_list}))

    # графики найденных функций
    # fig, axs = plt.subplots(2, 4, figsize=(10, 8))
    # fig.suptitle(f'{data[:-4]}')
    # axs[0, 0].scatter(X, Y, c='black', s=1, marker='x')
    # for i, ax in zip([0, 1, 2, 3, 5, 7, 9, 10], axs.flatten()):
    #     ax.scatter(X, qr_approx_list[i], s=0.1, c='red')
    #     ax.scatter(X, ne_approx_list[i], s=0.1, c='blue')
    #     ax.set_title(f'N={i}')

    # график SME в зависимоти от N
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.yscale('log')
    ax.plot(sme_qr, c='red', label='SME of QR')
    ax.plot(sme_ne, c='blue', label='SME of NE')
    ax.legend(fontsize=12)
    ax.set_title(f'{data}', fontsize=12)
    ax.set_xlabel('N', fontsize=12)
    ax.set_ylabel('SME', fontsize=12)

    print(f'{dt.to_string()}\n')
    writer = pd.ExcelWriter(f'results_from_{data[:-4]}.xlsx')
    dt.to_excel(writer, f'{data[:-4]}')
    writer.save()
    writer.close()


def main():
    degree = 11
    data_files = glob.glob('*.txt')
    for data in data_files:
        obtain(data, degree)
    print('Finish!')
    plt.show()


if __name__ == '__main__':
    main()
