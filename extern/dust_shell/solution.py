import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from math import pi, sqrt, exp, log
import mpmath as mp

from sympy import false

if __name__ == "__main__":
    # compute ODE solution for radiation-driven dusty shell
    a_rad = 7.5646e-15      # erg cm^-3 K^-4
    c = 2.99792458e10       # cm s^-1
    a0 = 2.0e5              # ('reference' sound speed) [cm s^-1]
    Msun = 2.0e33           # g
    parsec_in_cm = 3.086e18  # cm

    specific_luminosity = 2000.             # erg s^-1 g^-1
    GMC_mass = 1.0e6 * Msun                 # g
    epsilon = 0.5                           # dimensionless
    M_shell = (1 - epsilon) * GMC_mass      # g
    L_star = (epsilon * GMC_mass) * specific_luminosity  # erg s^-1
    r_0 = 5.0 * parsec_in_cm  # cm
    
    sigma_star = 0.3 * r_0  # N.B.: smaller values yield unphysical reduced flux f
    H_shell = 0.3 * r_0  # cm

    sigma_shell = H_shell / (2.0*sqrt(2.0*log(2.0)))
    rho_0 = M_shell / ((4. / 3.) * np.pi * r_0 * r_0 * r_0)  # g cm^-3
    kappa0 = 20.0           # specific opacity [cm^2 g^-1]
    tau0 = M_shell * kappa0 / (4.0 * np.pi * r_0 * r_0)

    r = r_0
    rad_force = tau0 * (r/r_0)**(-2.) * (L_star / c)
    dt = 1.0e9  # s

    print(f"fiducial optical depth tau0 = {tau0}")

    print_code_test = False
    if print_code_test:
        print(f"radiation force F = {rad_force} (cgs)")
        print(f"dMomentum = {dt*rad_force}")

        Frad_shell = L_star / (4.0 * np.pi * r_0 * r_0)
        Erad_edd = (3.0 * Frad_shell / c) * (tau0 + 2. / 3.)
        Trad_edd = (Erad_edd / a_rad)**(1. / 4.)
        print(f"Trad = {Trad_edd}")

    # solution for radiation flux F in cgs
    def Frad(r):
        # r is in cgs!
        normfac = L_star/(4.0*np.pi*r*r)
        term1 = mp.erf(r/(sqrt(2.0)*sigma_star))
        term2 = 2.0*r/(sqrt(2.0*np.pi)*sigma_star) * \
            exp(-0.5 * (r/sigma_star)**2)
        return normfac*(term1 - term2)

    def dlnF_dr(r):
        # r is in cgs!
        # compute d ln Frad / dr
        return (-2/r) + (2*(r/sigma_star)**2) / \
            (-2*r + mp.exp(0.5*(r/sigma_star)**2) * (mp.sqrt(2*pi)
             * sigma_star) * mp.erf(r/(mp.sqrt(2)*sigma_star)))

    # density profile in cgs
    def rho(r):
        # r is in cgs!
        # compute density profile
        normfac = M_shell/(4.0*np.pi*r*r*sqrt(2.0*np.pi*sigma_shell**2))
        expfac = exp(-(r-r_0)**2 / (2.0*sigma_shell**2))
        return normfac*expfac

    def df_dr_lim(f, r_in):
        # r_in is in units of r_0
        r = r_in * r_0  # cm
        tau = kappa0 * rho(r)
        G = dlnF_dr(r)
        sigma = sigma_star
        deriv = (2*r*(2*(10 - 4*sqrt(4 - 3*f**2) + 3*f**2*(-5 + 3*sqrt(4 - 3*f**2)))*r**2 +
                      2*(f**2*(6 - 9*sqrt(4 - 3*f**2)) + 4*(-1 + sqrt(4 - 3*f**2)))*sigma**2 +
                      3*f*(-8 + 9*f**2)*r*sigma**2*tau) +
                 exp(r**2/(2.*sigma**2))*sqrt(2*pi)*sigma**3 *
                 (8 - 8*sqrt(4 - 3*f**2) -
                  3*f*(-8*r*tau + f*(4 - 6*sqrt(4 - 3*f**2) + 9*f*r*tau))) *
                 mp.erf(r/(sqrt(2)*sigma))) /\
            (15.*f*r*sigma**2*(2*r - exp(r**2/(2.*sigma**2))*sqrt(2*pi)*sigma *
                               mp.erf(r/(sqrt(2)*sigma))))
        return -r_0*deriv

    def df_dr(f, r_in):
        # r_in is dimensionless
        sqrtfac = sqrt(4.0 - 3.0*f*f)
        normfac = 3.0*f*sqrtfac / (5.0*sqrtfac - 8.0)
        r = r_in * r_0
        term1 = (dlnF_dr(r) / 3.0) * (5.0 - 2.0*sqrtfac)
        term2 = (2.0/r) * (2.0 - sqrtfac)
        term3 = rho(r) * kappa0 * f
        return r_0 * normfac*(term1 + term2 + term3)

    # compute ODE solution for f == F/cE (dimensionless)
    def func_M1(t, y):
        # return dy / dt == func(t,y)
        # in this case, y == f (reduced flux)
        #               t == r (radius)
        flux = y[0]
        r = t
        if(flux <= 0.):
            print(f"r = {r}; f = {flux}")
            assert(flux > 0.)
        deriv = df_dr(flux, r)
        return [deriv]

    # solve for critical point
    def g(r):
        return float(3.0*r_0*dlnF_dr(r*r_0) + (4.0/r) + 2.0*sqrt(3.0)*r_0*rho(r*r_0)*kappa0)

    #plt.figure()
    #r_arr = np.linspace(1.0e-3, 2.0, 100)
    #g_vec = np.vectorize(g)
    #plt.plot(r_arr, g_vec(r_arr))
    #plt.hlines(y=0., xmin=0., xmax=2, color='black', linestyles='--')
    #plt.ylim(-1, 10)
    #plt.savefig("critical_point.pdf")
    #plt.clf()

    root_sol = root_scalar(g, bracket=[1.0, 1.5], method='bisect')
    print(f"critical point r_crit = {root_sol}\n")

    r_crit = root_sol.root  # critical point at f(r_crit) = f_crit
    f_crit = 2.0*sqrt(3.0) / 5.0
    df_dr_crit = df_dr_lim(f_crit, r_crit)
    eps = 1.0e-6

    def solve_branch(f0, r0, r1):
        # this equation is stiff
        sol = solve_ivp(func_M1, [r0, r1], [f0], method='BDF')
        #print(f"{sol}")
        return sol

    # solve right branch
    r_start = r_crit + eps
    r_end = 7.0  # in units of r_0
    f_start = f_crit + eps * df_dr_crit
    print(f"r_crit = {r_crit}; r_start = {r_start}")
    print(f"f_crit = {f_crit}; f_start = {f_start}")
    print(f"df_dr(f_crit) = {df_dr_crit}\n")
    right_sol = solve_branch(f_start, r_start, r_end)
    print("")

    solve_left = True
    if solve_left:
        # solve left branch
        r_start = r_crit - eps
        r_end = 1.0e-4  # in units of r_0
        f_start = f_crit - eps * df_dr_crit
        print(f"r_crit = {r_crit}; r_start = {r_start}")
        print(f"f_crit = {f_crit}; f_start = {f_start}")
        print(f"df_dr(f_crit) = {df_dr_crit}\n")
        left_sol = solve_branch(f_start, r_start, r_end)

    plt.figure()
    if solve_left:
        plt.plot(left_sol.t, left_sol.y[0], label='left branch')
    plt.plot(right_sol.t, right_sol.y[0], label='right branch')
    plt.scatter(r_crit, f_crit, color='black', s=15.0)

    plt.xlabel('radius r (dimensionless)')
    plt.ylabel('reduced flux f (dimensionless)')
    #plt.xlim(0., 3.5)
    plt.xlim(0., 7.0)
    plt.ylim(0., 1.)
    plt.savefig('solution.pdf')

    # save solution to text file
    r_over_r0 = np.concatenate((left_sol.t[::-1], right_sol.t))
    reduced_flux = np.concatenate((left_sol.y[0][::-1], right_sol.y[0]))
    Frad_vec = np.vectorize(Frad)
    Frad_cgs = Frad_vec(r_over_r0 * r_0)
    Erad_cgs = Frad_cgs / (c*reduced_flux)
    np.savetxt("initial_conditions.txt", np.c_[
               r_over_r0, reduced_flux, Erad_cgs, Frad_cgs],\
               header="r_over_r0 reduced_flux Erad_cgs Frad_cgs")
