

def test_delta_prime_convergence_n_points(sample_stride: int):
    from tearing_mode_solver.outer_region_solver import solve_system, delta_prime
    from tearing_mode_solver.profiles import generate_j_profile, generate_q_profile
    from tearing_mode_solver.helpers import TearingModeParameters

    j_prof = generate_j_profile(1.5, 2.0)
    q_prof = generate_q_profile(1.5, 2.0)

    j_rs, js = zip(*j_prof)
    q_rs, qs = zip(*q_prof)

    j_rs = j_rs[::sample_stride]
    js = js[::sample_stride]

    q_rs = q_rs[::sample_stride]
    qs = qs[::sample_stride]

    params = TearingModeParameters(
        2, 1,
        1e8,
        1e-12,
        1.0,
        10.0,
        list(zip(q_rs, qs)),
        list(zip(j_rs, js))
    )

    sol = solve_system(params)

    dp = delta_prime(sol)

    return len(qs), dp


def compare_spline_fits():
    from tearing_mode_solver.profiles import generate_j_profile, generate_q_profile
    from tearing_mode_solver.outer_region_solver import rational_surface
    from scipy.interpolate import CubicSpline, UnivariateSpline
    from matplotlib import pyplot as plt

    j_prof = generate_j_profile(1.5, 2.0)
    q_prof = generate_q_profile(1.5, 2.0)

    j_rs, js = zip(*j_prof)
    q_rs, qs = zip(*q_prof)

    r_s = rational_surface(q_prof)

    xrange = [1, 50, 100, 200, 500, 1000]

    for i in xrange:
        # fig, ax = plt.subplots(1)
        # r_temp = q_rs[::i]
        # q_temp = qs[::i]
        # ax.scatter(r_temp, q_temp)
        # cs = CubicSpline(r_temp, q_temp)
        # ax.plot(q_rs, cs(q_rs), label="CS s=None")

        # cs_s0 = UnivariateSpline(r_temp, q_temp, s=0)
        # ax.plot(q_rs, cs_s0(q_rs), label="US s=0")


        r_temp = j_rs[::i]
        j_temp = js[::i]
        j_cs = CubicSpline(r_temp, j_temp)
        j_cs_deriv = j_cs.derivative()

        dj_dr = j_cs_deriv(r_temp)


        r_temp = q_rs[::i]
        q_temp = qs[::i]
        cs = CubicSpline(r_temp, q_temp)

        q = cs(r_temp)
        q0 = q[0]
        m=2
        n=1

        A = 2.0*(q/q0*m*dj_dr)/(n*q - m)

        fig2, ax2 = plt.subplots(1)

        ax2.plot(r_temp, A)


    plt.show()


def test_splines():
    from scipy.interpolate import UnivariateSpline, CubicSpline
    import numpy as np
    from matplotlib import pyplot as plt

    xs = np.linspace(0.0, 1.0, 20)
    ys = np.sin(10.0*np.pi*xs)

    cs = CubicSpline(xs, ys)
    us = UnivariateSpline(xs, ys)

    print(us.get_knots())

    line_xs = np.linspace(0.0, 1.0, 200)

    plt.scatter(xs, ys)
    plt.plot(line_xs, cs(line_xs))
    plt.plot(line_xs, us(line_xs))

    plt.show()


if __name__=='__main__':

    dps = []
    arr_len = []
    xrange = [1,50,100,200,500,1000]
    for i in xrange:
        len_q, dp = test_delta_prime_convergence_n_points(i)
        arr_len.append(len_q)
        dps.append(dp)

    print(dps)

    from matplotlib import pyplot as plt

    plt.plot(arr_len, dps, marker='x')
    plt.show()

    #compare_spline_fits()