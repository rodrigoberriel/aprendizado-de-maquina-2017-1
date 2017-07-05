import os
import numpy as np
import matplotlib.pyplot as plt

import utils
import constants


def load_runner(fname):
    data = np.loadtxt(fname, delimiter='\t', dtype=bytes).astype(str)
    years = data[:, 0].astype(np.float)
    times = data[:, 1].astype(np.float)
    return years, times


def exercicio5():
    utils.print_header(5)

    years, times = load_runner(os.path.join(constants.DATA_DIR, constants.FILENAME_RUNNER_DATABASE))
    N = years.shape[0]

    f, w0_hat, w1_hat = utils.linear_model(years, times)
    y_pred = np.array([f(year) for year in years])

    tau_b = utils.KendallTauB(years, times)
    p = utils.Pearson(years, times)

    # Slide 59, Aula 4
    def reject_kendall(tau, alpha): return abs(tau) > utils.get_z(alpha) * np.sqrt((2*(2*N+5)) / (9*N*(N-1)))

    # Slide 52, Aula 4
    def reject_pearson(p, alpha): return abs((p*np.sqrt(N-2)) / (np.sqrt(1-(p**2)))) > utils.t_student(N-2, alpha/2)

    print('a)')
    print('\tLinear equation: {:.3f} {} {:.3f}x'.format(w0_hat, '+' if w1_hat >= 0 else '-', abs(w1_hat)))
    print('\tRMSE: {:.3f}'.format(utils.RMSE(y_pred, times)))
    plt.scatter(years, times, linewidths=0)
    plt.plot(years, f(years), c='r')
    plt.axhline(y=f(2016), color='g', linestyle='--')
    plt.scatter(2016, f(2016), c='g', linewidths=0)
    plt.tight_layout()
    plot_fname = os.path.join(constants.OUTPUT_DIR, 'exercicio5-a.pdf')
    plt.savefig(plot_fname, bbox_inches='tight')
    plt.show()

    print('b)')
    print('\tPrediction for 2016: {:.3f} seconds'.format(f(2016)))

    print('c)')
    print('\tKendall\'s tau: {:.3f}'.format(tau_b))
    print('\tNull hypothesis rejected:\n\t- 95%: {}\n\t- 99%: {}'.format(
        reject_kendall(tau_b, 0.05), reject_kendall(tau_b, 0.01))
    )

    print('d)')
    print('\tPearson correlation coefficient: {:.3f}'.format(p))
    if abs(p) > 0.85:
        print('\t|p| > 0.85 and null hypothesis rejected:\n\t- 95%: {}\n\t- 99%: {}'.format(
            reject_pearson(p, 0.05), reject_pearson(p, 0.01))
        )

    exit()


if __name__ == '__main__':
    exercicio5()
