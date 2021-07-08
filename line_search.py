import numpy as np


def backtracking_armijo_ls(f, df, x, d, m1=0.9, tau=0.5):
    phi = lambda a: f(x + a * d)
    d_phi = lambda a: d @ df(x + a * d)
    alpha = 1
    phi0 = phi(0)
    d_phi0 = d_phi(0)

    # print("phi(alpha)={}\n(phi0 + m1 * alpha * d_phi0)={}".format(phi(alpha), (phi0 + m1 * alpha * d_phi0)))
    while phi(alpha) > (phi0 + m1 * alpha * d_phi0):
        alpha = tau * alpha

    return alpha


def armijo_wolfe_ls(f: callable, df: callable, x, d, a_max, m1=0.1, m2=0.9, eps=1e-16, max_iter=1000, tau=0.1):
    phi = lambda a: f(x + a * d)
    d_phi = lambda a: d @ df(x + a * d)

    phi_0 = phi_prev = phi(0)
    d_phi_0 = d_phi_prev = d_phi(0)
    a_prev = a_max*m1

    def interpolate(a_lo, a_hi, phi_lo, phi_hi, d_phi_lo, d_phi_hi):

        d1 = d_phi_lo + d_phi_hi - 3 * ((phi_lo - phi_hi) / (a_lo - a_hi))
        d2 = np.sign(a_hi - a_lo) * np.sqrt(d1 ** 2 - d_phi_lo * d_phi_hi)

        a = a_hi - (a_hi - a_lo) * (d_phi_hi + d2 - d1) / (d_phi_hi - d_phi_lo + 2 * d2)

        return a

    def zoom(a_lo, a_hi, phi_lo, phi_hi, d_phi_lo, d_phi_hi):

        while True:
            a_j = interpolate(a_lo, a_hi, phi_lo, phi_hi, d_phi_lo, d_phi_hi)

            phi_j = phi(a_j)
            d_phi_j = d_phi(a_j)

            if phi_j > phi_0 + m1 * a_j * d_phi_0 \
                    or phi_j >= phi_lo:
                a_hi = a_j
                d_phi_hi = d_phi_j
            else:
                if abs(d_phi_j) <= -m2 * d_phi_0:  # goldstein
                    return a_j
                if d_phi_j * (a_hi - a_lo) >= 0:  # it's increasing, shift the interval
                    a_hi = a_lo
                a_lo = a_j
                d_phi_lo = d_phi_j


    if a_max is None:  # in our case, None == inf
        print("no max to a")
        a = 1
    else:
        a = a_max
    i = 1

    while i <= max_iter \
            and abs(a - a - a_prev) > eps \
            and np.all(phi_prev > 1e-6):
        phi_a = phi(a)
        d_phi_a = d_phi(a)

        if phi_a > phi_0 + m1 * a * d_phi_0 \
                or (phi_a >= phi_prev and i > 1):
            return zoom(a_prev, a, phi_prev, phi_a, d_phi_a, d_phi_prev)

        if abs(d_phi_a) <= -m2 * d_phi_0:
            return a
        if d_phi_a >= 0:
            return zoom(a, a_prev, phi_a, phi_prev, d_phi_a, d_phi_prev)
        a_prev = a
        d_phi_prev = d_phi_a
        phi_prev = phi_a
        a = a * tau
        # print(a)

    return a