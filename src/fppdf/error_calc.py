from pathlib import Path

import numpy as np
import scipy.linalg as la
from scipy.optimize import newton

from fppdf.chi2s import chi2min, chi2min_fun
from fppdf.global_pars import chi2_pars, dload_pars, fit_pars, inout_pars, pdf_pars
from fppdf.outputs import evgrido, parsout, covmatout_err, parsout_err

OUTDIR_EV = Path("outputs/evscans")
OUTDIR_EV.mkdir(exist_ok=True, parents=True)
VERBOSE = True


def _print_verbose(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def hesserror_dynamic_tol_new(afi, hess, jaci):
    """Find the hessian members using dynamic tolerance.
    The output PDF is written with the suffix _dyntol.
    """
    pdf_name = f"{inout_pars.label}_dyntol"
    output_log = OUTDIR_EV / f"{pdf_name}.dat"
    msht_fix = False

    if msht_fix:
        hess, afi = hessfix(hess)

    chi2_pars.chi2ind = True
    chi2_pars.chitotind = False
    chi2_pars.L0 = False

    # For closure tests use experimental cov matrix (quicker)
    if inout_pars.pdin:
        chi2_pars.t0 = False

    # chi2_pars.t0=False

    #   Calculate eigenvalues + eigenvector of Hessian and then invert/flip so correspond to C=H^-1 with appropriate ordering
    lam0, eig = la.eigh(hess)
    lam = 1.0 / lam0
    lam = np.flip(lam)
    eig = np.flip(eig, 1)

    #### Remove negative eigenvalues
    nlam_neg = np.count_nonzero(lam < 0, axis=0)
    if nlam_neg > 0:
        lam_noneg = np.delete(lam, np.s_[-nlam_neg:])
        lam = lam_noneg.copy()
        eig_noneg = eig[:, :-nlam_neg]
        eig = eig_noneg.copy()

    #   Rescale
    eig = eig * np.sqrt(lam)

    #   Central chi^2 - from full covariance matrix
    af = afi.copy()
    chi0 = chi2min(afi)

    #   Output central grid and increase replica number ready for eigenvector generation
    evgrido(pdf_name)
    fit_pars.irep = fit_pars.irep + 1

    #   Header for evscan file
    output_log.write_text(f"chi2_0 = {chi0:.5f}, neig = {len(lam)}")

    #   Loop over eigenvectors
    for j in range(0, len(lam) - chi2_pars.lam_sub):
        print('j = ', j)

        # jth eigenvector
        eig0 = eig[:, j].flatten()

        # +/- directions
        # for i in [-1,1]:
        for i in [-1, 1]:

            delt = 1.0
            idelmax = 20

            for k in range(1, idelmax):

                tchi = delt * k * i
                print('tchi = ', tchi)
                _, dchi2max, arg_dchi2max, chi2t = get_chi2_ind(af, tchi, eig0.copy())

                if chi2t > 1e40:
                    print('shift outside allowed region -> reduce t')
                    chi2 = chi2t
                    while chi2 > 1e40:
                        tchi /= 2.0
                        print('tchi = ', tchi)
                        afin = af.copy() + tchi * eig0.copy()
                        chi2 = chi2min(afin)
                    delt = np.abs(tchi) / 5.0
                    tchi = delt * k * i
                    print('tchi = ', tchi)
                    _, dchi2max, arg_dchi2max, chi2t = get_chi2_ind(af, tchi, eig0.copy())

                if dchi2max > 1e1:
                    tchi_i = tchi - delt * i
                    tchi_del = delt
                    print('limit far surpassed -> reduce t by hand')
                    dchi2 = dchi2max
                    while dchi2 > 1e1:
                        tchi_del /= 2.0
                        tchi = tchi_i + tchi_del * i
                        print('tchi = ', tchi)
                        _, dchi2, arg_dchi2max, chi2t = get_chi2_ind(af, tchi, eig0.copy())
                    print('tchi out = ', tchi)

                # test=newton_func_ind(tchi,af,eig0,arg_dchi2max)
                if dchi2max > 1.0:
                    try:
                        tchi_out = newton(
                            newton_func_ind,
                            tchi,
                            args=(af, eig0, arg_dchi2max),
                            maxiter=40,
                            tol=0.01,
                        )
                    except RuntimeError as err:
                        print('newton method failing, try brute force...')

                        dchi2 = 0.0
                        delt = i * 0.1
                        tchi = delt

                        while dchi2 < 1.0:
                            tchi += delt
                            print('tchi = ', tchi)
                            deltachi2_lim, dchi2, arg_dchi2max, chi2t = get_chi2_ind(
                                af, tchi, eig0.copy()
                            )
                        tchi_out = tchi

                    print('tchi_out = ', tchi_out)
                    # afin=af.copy()+tchi_out*eig0.copy()
                    _, dchi2, arg_dchi2max, chi2t = get_chi2_ind(af, tchi_out, eig0.copy())
                    # delchi2t=chi2min(afin)-chi0
                    delchi2t = chi2t - chi0
                    print('delchi2 out = ', delchi2t)
                    if delchi2t > 1e10:
                        print('tchi leads to unstable region of parameter space - set t=0')
                        tchi_out = 1e-10
                        delchi2t = 0.0
                    evscan_output_dyT(j, tchi_out, delchi2t, arg_dchi2max, output_log, pdf_name)
                    break


# Function defined so that is zero at deltachi^2_i=deltachi^2_lim for constraining dataset
def newton_func_ind(x, afi, eig, arg_dchi2max):

    af = afi.copy() + eig * x
    chi2t_t = chi2min(af)
    deltachi2 = np.array(chi2_pars.chi_ind_arr) - np.array(chi2_pars.chi0_ind_arr)
    deltachi2_lim = deltachi2 / np.array(chi2_pars.chilim_arr)
    lim = deltachi2_lim[arg_dchi2max]
    dchi = deltachi2[arg_dchi2max]
    out = lim - 1.0

    print(arg_dchi2max)
    print('Newton - chi2 =', dchi)
    print('Newton - chi2/lim =', lim)
    print('Newton - tchi =', x)

    dload_pars.tchi_newton = x

    return out


def bruteforce_ind(x, afi, eig, arg_dchi2max):

    af = afi.copy() + eig * x
    chi2t_t = chi2min(af)
    deltachi2 = np.array(chi2_pars.chi_ind_arr) - np.array(chi2_pars.chi0_ind_arr)
    deltachi2_lim = deltachi2 / np.array(chi2_pars.chilim_arr)
    lim = deltachi2_lim[arg_dchi2max]
    dchi = deltachi2[arg_dchi2max]
    out = lim - 1.0

    print('Brute force - chi2 =', dchi)
    print('Brute force - chi2/lim =', lim)
    print('Brute Force - tchi =', x)

    return out


def get_chi2_ind(af, t, eig):
    """Get the individual chi2s"""
    afint = af.copy() + t * eig
    chi2t_t = chi2min(afint)
    deltachi2 = np.array(chi2_pars.chi_ind_arr) - np.array(chi2_pars.chi0_ind_arr)
    deltachi2_lim = deltachi2 / np.array(chi2_pars.chilim_arr)
    dchi2max = np.max(deltachi2_lim)
    arg_dchi2max = np.argmax(deltachi2_lim)
    # print(deltachi2)
    print('deltachi^2/lim = ', dchi2max)
    # print(chi2_pars.chilim_arr)
    # print(np.max(deltachi2_lim))
    print('limiting dataset = ', np.argmax(deltachi2_lim))
    # print(deltachi2[np.argmax(deltachi2_lim)])
    # exit()

    return (deltachi2_lim, dchi2max, arg_dchi2max, chi2t_t)


def hessfix(hess):

    inputfile = 'input/fixinput.dat'
    distuv = np.loadtxt(inputfile, skiprows=1, max_rows=9, usecols=1)
    distdv = np.loadtxt(inputfile, skiprows=11, max_rows=9, usecols=1)
    distsea = np.loadtxt(inputfile, skiprows=21, max_rows=9, usecols=1)
    distsp = np.loadtxt(inputfile, skiprows=31, max_rows=9, usecols=1)
    distg = np.loadtxt(inputfile, skiprows=41, max_rows=10, usecols=1)
    distsm = np.loadtxt(inputfile, skiprows=52, max_rows=9, usecols=1)
    distdbub = np.loadtxt(inputfile, skiprows=62, max_rows=8, usecols=1)
    distcharm = np.loadtxt(inputfile, skiprows=71, max_rows=9, usecols=1)

    disttot = np.concatenate([distuv, distdv, distsea, distsp, distg, distsm, distdbub, distcharm])

    #   remove pars from array when fixed

    pdf_pars.npar_free = 0
    afin = np.zeros((1))
    dist_afin = np.zeros((1))
    pdf_pars.par_free_i = np.zeros((1), dtype=np.int8)
    for i in range(0, len(pdf_pars.parsin)):
        if pdf_pars.par_isf[i] == 1:
            dist_afin = np.append(dist_afin, disttot[i])
        if pdf_pars.par_isf[i] == 1 and disttot[i] == 0:
            afin = np.append(afin, pdf_pars.parsin[i])
            pdf_pars.npar_free += 1
            pdf_pars.par_free_i = np.append(pdf_pars.par_free_i, i)
        else:
            pdf_pars.par_isf[i] = 0

    afin = np.delete(afin, 0)
    dist_afin = np.delete(dist_afin, 0)
    pdf_pars.par_free_i = np.delete(pdf_pars.par_free_i, 0)

    hessout = hess.copy()

    mask = dist_afin > 0
    hessout = np.delete(hessout, mask, axis=0)
    hessout = np.delete(hessout, mask, axis=1)

    return (hessout, afin)


def hesserror_new_backup(afi, hess):
    """Compute the hessian eigenvectors with fixed tolerance"""
    msht_fix = False

    if msht_fix:
        hess, afi = hessfix(hess)

    delchisq_exact = True
    # For closure tests use experimental cov matrix (quicker)
    if inout_pars.pdin:
        chi2_pars.t0 = False

    tol_exact = 0.01
    tol = np.sqrt(10.0)
    tol = 1.0

    tol = np.sqrt(chi2_pars.t2_err)

    #   Calculate eigenvalues + eigenvector of Hessian and then invert/flip so correspond to C=H^-1 with appropriate ordering
    lam0, eig = la.eigh(hess)
    lam = 1.0 / lam0
    lam = np.flip(lam)
    eig = np.flip(eig, 1)

    #### Remove negative eigenvalues
    nlam_neg = np.count_nonzero(lam < 0, axis=0)

    if nlam_neg > 0:
        lam_noneg = np.delete(lam, np.s_[-nlam_neg:])
        lam = lam_noneg.copy()
        eig_noneg = eig[:, :-nlam_neg]
        eig = eig_noneg.copy()

    #   Rescale
    eig = eig * np.sqrt(lam)

    #   Central chi^2
    chi0 = chi2min(afi)
    chi0pos = chi2_pars.chi_pos1

    # The name of the log file will depend on the tolerance and whether dynamic tolerance is being used
    output_log = OUTDIR_EV / f"{inout_pars.label}_tol={chi2_pars.t2_err}.dat"

    #   Output central grid and increase replica number ready for eigenvector generation
    evgrido()
    fit_pars.irep = fit_pars.irep + 1

    #   Header for evscan file
    output_log.write_text(f"chi2_0 = {chi0:.5f}, neig = {len(lam)}")

    #   Loop over eigenvectors
    for j in range(0, len(lam)):
        # jth eigenvector
        eig0 = eig[:, j].flatten()
        _print_verbose(f"Eigenvector j={j}: {eig0}")

        # +/- directions
        for i in [-1, 1]:
            # new parameter
            tchi = i * tol
            afin = afi + tchi * eig0

            _print_verbose(f"Tolerance: {tchi}")
            _print_verbose(f"delpar = {afin - afi}")
            # by calling chi2min(afin) we are also updating the free parameter pdf_pars.pdfparsi
            chi2t = chi2min(afin)
            delchi2t = chi2t - chi0
            _print_verbose(f"chidel = chi2_tol - chi2_0 = {chi2t} - {chi0} = {delchi2t}")

            if delchi2t > 1e40:
                print('shift outside allowed region -> reduce t')
                delchi2 = delchi2t
                while delchi2 > 1e40:
                    tchi /= 2.0
                    print('tchi = ', tchi)
                    afin = afi + tchi * eig0
                    chi2t = chi2min(afin)
                    delchi2 = chi2t - chi0
                delchi2t = delchi2

            if not delchisq_exact:
                print('fixed t - output')
                delchi2pos = chi2_pars.chi_pos1 - chi0pos
                evscan_output(j, tchi, delchi2t, delchi2pos, output_log=output_log)
            elif delchi2t < 0.0:
                print('delchi^2 less than 0 - minimum not reached!')
                tchi, delchi2 = tchi_findmin(chi0, tchi, afi, eig0, tol, output_log=output_log)
                if delchi2 > np.power(tol, 2) * 5.0:
                    delchi2, tchi = dchi2_toolarge(tol, tchi, afi, eig0, chi0, delchi2)
                delchi2pos = chi2_pars.chi_pos1 - chi0pos
                evscan_output(j, tchi, delchi2, delchi2pos, output_log=output_log)
            elif np.abs(delchi2t - np.power(tol, 2)) < tol_exact:
                print('|delchi2 - T^2| < ', tol_exact, ' - done')
                delchi2pos = chi2_pars.chi_pos1 - chi0pos
                evscan_output(j, tchi, delchi2t, delchi2pos, output_log=output_log)
            else:
                # If delchi^2 really big just keep dividing t/2 before applying Newton's method (quicker)
                if delchi2t > np.power(tol, 2) * 5.0:
                    _print_verbose(f"Deltachi2 very big, {delchi2t}")
                    delchi2t, tchi = dchi2_toolarge(tol, tchi, afi, eig0, chi0, delchi2t)
                    if delchi2t < 0.0:
                        tchi, delchi2t, af_out, dchi_out = tchi_findmin(
                            chi0, tchi, afi, eig0, tol, output_log=output_log
                        )

                # tchi_out=tchi_bruteforce(delchi2t,chi0,tchi,af,eig0,tol)
                print('|delchi2 - T^2| > ', tol_exact, " -> Use Newton's method")
                try:
                    tchi_out = newton(
                        newton_func, tchi, args=(afi, eig0, tol, chi0), maxiter=40, tol=0.01
                    )
                    # If min has taken t to the other eigenvector direction use brute force instead
                    if tchi_out / (i * tol) < 0:
                        print('tchi has wrong sign!')
                        tchi_out = tchi_bruteforce(delchi2t, chi0, tchi, afi, eig0, tol)
                    print('tchi out = ', tchi_out)
                    afin = afi + tchi_out * eig0
                    delchi2t = chi2min(afin) - chi0
                    print('delchi2 out = ', delchi2t)
                    if delchi2t > np.power(tol, 2) * 5.0:
                        delchi2t, tchi = dchi2_toolarge(tol, tchi, afi, eig0, chi0, delchi2t)
                    # if np.abs(delchi2t-np.power(tol,2)) < tol_exact:
                    #     print('Now |delchi2 - T^2| < ',tol_exact,' - done')
                    delchi2pos = chi2_pars.chi_pos1 - chi0pos
                    evscan_output(j, tchi_out, delchi2t, delchi2pos, output_log=output_log)
                except RuntimeError as err:
                    print(err)
                    tchi_out = dload_pars.tchi_newton
                    print('tchi = ', tchi_out)
                    afin = afi + tchi_out * eig0
                    delchi2t = chi2min(afin) - chi0
                    delchi2pos = chi2_pars.chi_pos1 - chi0pos
                    evscan_output(j, tchi_out, delchi2t, delchi2pos, output_log=output_log)


def hesserror_new(afi_i, hessin_i):

    end_run = True
    hessin = hessin_i.copy()
    afi = afi_i.copy()
    irun = 0

    while end_run == True:

        af_out, end_run = hesserror_new_call(afi, hessin, irun)
        irun += 1
        if end_run == True:
            out = chi2min_fun(af_out, True, True)
            hessout = out[2] / 2.0
            covmatout_err(hessout, -out[1] / 2.0)
            parsout_err()
            # print('hessout - ',hessout)
            # print(end_run)
            hessin = hessout
            afi = af_out.copy()


def hesserror_new_call(afi, hessin, irun):
    """Compute the hessian eigenvectors with fixed tolerance"""
    msht_fix = False

    hess = hessin.copy()
    af_out = afi.copy()

    end_run = False

    if msht_fix:
        hess, afi = hessfix(hess)

    delchisq_exact = True
    # For closure tests use experimental cov matrix (quicker)
    if inout_pars.pdin:
        chi2_pars.t0 = False

    tol_exact = 0.01
    tol = np.sqrt(10.0)
    tol = 1.0

    tol = np.sqrt(chi2_pars.t2_err)

    #   Calculate eigenvalues + eigenvector of Hessian and then invert/flip so correspond to C=H^-1 with appropriate ordering
    lam0, eig = la.eigh(hess)
    lam = 1.0 / lam0
    lam = np.flip(lam)
    eig = np.flip(eig, 1)

    #### Remove negative eigenvalues
    nlam_neg = np.count_nonzero(lam < 0, axis=0)

    if nlam_neg > 0:
        lam_noneg = np.delete(lam, np.s_[-nlam_neg:])
        lam = lam_noneg.copy()
        eig_noneg = eig[:, :-nlam_neg]
        eig = eig_noneg.copy()

    #   Rescale
    eig = eig * np.sqrt(lam)

    #   Central chi^2
    chi0 = chi2min(afi)
    chi0pos = chi2_pars.chi_pos1

    # The name of the log file will depend on the tolerance and whether dynamic tolerance is being used
    output_log = OUTDIR_EV / f"{inout_pars.label}_tol={chi2_pars.t2_err}.dat"

    evgrido()
    fit_pars.irep = 0

    #   Header for evscan file
    if irun == 0:
        output_log.write_text(f"chi2_0 = {chi0:.5f}, neig = {len(lam)}")
        with open(output_log, 'a') as outputfile:
            outputfile.write('\n')
    else:
        with open(output_log, 'a') as outputfile:
            outputfile.write('chi2 new = ')
            outputfile.write(str(chi0))
            outputfile.write('\n')

    #   Loop over eigenvectors
    for j in range(0, len(lam) - chi2_pars.lam_sub):
        # for j in range(49, 50):
        # jth eigenvector
        eig0 = eig[:, j].flatten()

        if end_run:
            break

        _print_verbose(f"Eigenvector j={j}: {eig0}")

        # +/- directions
        for i in [-1, 1]:
            # new parameter
            tchi = i * tol
            afin = afi + tchi * eig0

            if end_run:
                break

            # chi2tt=chi2min_fun(afin, True, True)

            # print(chi2tt[0])
            # print('')
            # print(chi2tt[2])
            # print('')
            # print(chi2tt[2]/hess)
            # exit()

            _print_verbose(f"Tolerance: {tchi}")
            _print_verbose(f"delpar = {afin - afi}")
            # by calling chi2min(afin) we are also updating the free parameter pdf_pars.pdfparsi
            chi2t = chi2min(afin)

            delchi2t = chi2t - chi0
            _print_verbose(f"chidel = chi2_tol - chi2_0 = {chi2t} - {chi0} = {delchi2t}")

            if delchi2t > 1e40:
                print('shift outside allowed region -> reduce t')
                delchi2 = delchi2t
                while delchi2 > 1e40:
                    tchi /= 2.0
                    print('tchi = ', tchi)
                    afin = afi + tchi * eig0
                    chi2t = chi2min(afin)
                    delchi2 = chi2t - chi0
                delchi2t = delchi2

            if not delchisq_exact:
                print('fixed t - output')
                delchi2pos = chi2_pars.chi_pos1 - chi0pos
                evscan_output(j, tchi, delchi2t, delchi2pos, output_log=output_log)
            elif delchi2t < 0.0:
                print('delchi^2 less than 0 - minimum not reached!')
                tchi, delchi2, af_out, dchi_out = tchi_findmin(
                    chi0, tchi, afi, eig0, tol, output_log=output_log
                )
                if delchi2 > np.power(tol, 2) * 5.0:
                    delchi2, tchi = dchi2_toolarge(tol, tchi, afi, eig0, chi0, delchi2)
                delchi2pos = chi2_pars.chi_pos1 - chi0pos
                evscan_output(j, tchi, delchi2, delchi2pos, output_log=output_log)
                if dchi_out < -0.2:
                    end_run = True
                    with open(output_log, 'a') as outputfile:
                        outputfile.write('New min below cutoff : shift mininum and re-evaluate')
                        outputfile.write('\n')

            elif np.abs(delchi2t - np.power(tol, 2)) < tol_exact:
                print('|delchi2 - T^2| < ', tol_exact, ' - done')
                delchi2pos = chi2_pars.chi_pos1 - chi0pos
                evscan_output(j, tchi, delchi2t, delchi2pos, output_log=output_log)
            else:
                # If delchi^2 really big just keep dividing t/2 before applying Newton's method (quicker)
                if delchi2t > np.power(tol, 2) * 5.0:
                    _print_verbose(f"Deltachi2 very big, {delchi2t}")
                    delchi2t, tchi = dchi2_toolarge(tol, tchi, afi, eig0, chi0, delchi2t)
                    if delchi2t < 0.0:
                        tchi, delchi2t, af_out, dchi_out = tchi_findmin(
                            chi0, tchi, afi, eig0, tol, output_log=output_log
                        )

                        if dchi_out < -0.2:
                            end_run = True
                            with open(output_log, 'a') as outputfile:
                                outputfile.write(
                                    'New min below cutoff : shift mininum and re-evaluate'
                                )
                                outputfile.write('\n')

                # tchi_out=tchi_bruteforce(delchi2t,chi0,tchi,af,eig0,tol)
                print('|delchi2 - T^2| > ', tol_exact, " -> Use Newton's method")
                try:
                    tchi_out = newton(
                        newton_func, tchi, args=(afi, eig0, tol, chi0), maxiter=40, tol=0.01
                    )
                    # If min has taken t to the other eigenvector direction use brute force instead
                    if tchi_out / (i * tol) < 0:
                        print('tchi has wrong sign!')
                        tchi_out = tchi_bruteforce(delchi2t, chi0, tchi, afi, eig0, tol)
                    print('tchi out = ', tchi_out)
                    afin = afi + tchi_out * eig0
                    delchi2t = chi2min(afin) - chi0
                    print('delchi2 out = ', delchi2t)
                    if delchi2t > np.power(tol, 2) * 5.0:
                        delchi2t, tchi = dchi2_toolarge(tol, tchi, afi, eig0, chi0, delchi2t)
                    # if np.abs(delchi2t-np.power(tol,2)) < tol_exact:
                    #     print('Now |delchi2 - T^2| < ',tol_exact,' - done')
                    delchi2pos = chi2_pars.chi_pos1 - chi0pos
                    evscan_output(j, tchi_out, delchi2t, delchi2pos, output_log=output_log)
                except RuntimeError as err:
                    print(err)
                    tchi_out = dload_pars.tchi_newton
                    print('tchi = ', tchi_out)
                    afin = afi + tchi_out * eig0
                    delchi2t = chi2min(afin) - chi0
                    delchi2pos = chi2_pars.chi_pos1 - chi0pos
                    evscan_output(j, tchi_out, delchi2t, delchi2pos, output_log=output_log)

    return (af_out, end_run)


def dchi2_toolarge(tol, tchi, af, eig0, chi0, delchi2in):

    delchi2 = delchi2in
    print("delchi2 very big - divide t by 2 before applying Newton's method")
    while delchi2 > np.power(tol, 2) * 5.0:
        tchi /= 2.0
        print('tchi = ', tchi)
        afin = af.copy() + tchi * eig0.copy()
        chi2t = chi2min(afin)
        delchi2 = chi2t - chi0
        print('delchi2 = ', delchi2)
    print('delchi2 = ', delchi2)

    return (delchi2, tchi)


def tchi_findmin(chi0, tchi, afi, eig, tol, output_log=None):
    print('Starting findmin...')

    delt = 0.1 * tchi
    imax = 40

    dchi_min = 1e40
    t = 0.0

    printmin = True
    if output_log is None:
        raise ValueError("output_log is needed for tchi_findmin")
    outputfile = output_log.open("a")

    try:
        for _ in range(1, imax):
            t += delt
            print('tchi = ', t)
            af = afi.copy() + t * eig
            dchi_f = chi2min(af) - chi0
            print('delchi2 = ', dchi_f)
            print('dchi_min = ', dchi_min)
            if dchi_min > dchi_f:
                dchi_min = dchi_f
                if printmin:
                    if chi2_pars.chidel_min > dchi_f:
                        parsout(output_log.name)
            elif printmin:
                print('min reached!')
                t -= delt
                af = afi.copy() + t * eig
                af_out = af
                dchi_f = chi2min(af) - chi0
                dchi_out = dchi_f
                if chi2_pars.chidel_min > dchi_f:
                    chi2_pars.chidel_min = dchi_f
                print('tchi = ', t)
                print('delchi2 = ', dchi_f)
                printmin = False
                outputfile.write(f"New min = {dchi_f}\n")

            if dchi_f > np.power(tol, 2) + dchi_min:
                print('tol reached')
                print('tchi = ', t)
                print('delchi2 = ', dchi_f)
                break
    finally:
        outputfile.close()

    return (t, dchi_f, af_out, dchi_out)


def tchi_bruteforce(delchi, chi0, tchi, afi, eig, tol):

    print('Use brute force...')

    if delchi > np.power(tol, 2):
        tmin = 0.0
    else:
        tmin = tchi

    imax = 40
    delt = 0.01 * tchi

    t = tmin
    t_i = t
    dchi_i = delchi

    tout = t

    tarr = np.array([t])
    deltarr = np.array([delchi])

    for i in range(1, imax):
        t += delt
        af = afi.copy() + t * eig
        dchi_f = chi2min(af) - chi0
        print('t = ', t)
        print('delchi2 = ', dchi_f)
        tarr = np.append(tarr, t)
        deltarr = np.append(deltarr, dchi_f)
        if dchi_f > np.power(tol, 2):
            # delt2=np.power(t,2)-np.power(t_i,2)
            # mchi=(dchi_f-dchi_i)/delt2
            # tchi_down=quad_int_tchi(tol,t_i,dchi_i,mchi)
            # tchi_up=quad_int_tchi(tol,t,dchi_f,mchi)
            # af_up=afi.copy()+tchi_up*eig
            # dchi_up=chi2min(af_up)-chi0
            # af_down=afi.copy()+tchi_down*eig
            # dchi_down=chi2min(af_down)-chi0
            # tarr=np.append(tarr,tchi_up)
            # deltarr=np.append(deltarr,dchi_up)
            # tarr=np.append(tarr,tchi_down)
            # deltarr=np.append(deltarr,dchi_down)
            break
        else:
            dchi_i = dchi_f
            t_i = t

    tolarr = np.ones((len(tarr))) * np.power(tol, 2)
    deltarr = np.abs(deltarr - tolarr)
    it = np.argmin(deltarr)
    tchi_out = tarr[it]

    return tchi_out


def quad_int_tchi(y, a, b, m):

    x = (np.power(y, 2) - b) / m + a
    x = np.sqrt(x)

    return x


def newton_func(x, afi, eig, tol, chi0):
    """Function to be use with scipy's Newton's method
    which is zero at correct t=T
    """
    print("Calling newton func")

    af = afi + eig * x
    chi2t = chi2min(af)
    out = chi2t - chi0 - np.power(tol, 2)

    #     print('Newton - chidel =', chi2t - chi0)
    #     print('Newton - tchi =', x)

    dload_pars.tchi_newton = x

    return out


def evscan_output_dyT(j, tchi, delchi2t, arg_dchi2max, output_log, pdf_name):
    """Writte down the result of the dynamic tolerance scan."""

    if delchi2t < 0:
        delchi2t_sign = -np.sqrt(-delchi2t) * np.sign(tchi)
    else:
        delchi2t_sign = np.sqrt(delchi2t) * np.sign(tchi)

    dlab = chi2_pars.cldataset_arr[arg_dchi2max]

    with open(output_log, 'a') as outputfile:

        neig = np.rint((j + 1) * np.sign(tchi))

        if np.abs(tchi) < 1e-9:
            tchi = 0.0

        if np.sign(tchi) > 0:
            L = [
                'eig[',
                str(neig),
                ']  : ',
                ' t = ',
                str(f'{tchi:.5f}'),
                '   T = ',
                str(f'{delchi2t_sign:.5f}'),
                ' Limiting dataset = ',
                dlab,
            ]
        else:
            L = [
                'eig[',
                str(neig),
                '] : ',
                ' t = ',
                str(f'{tchi:.5f}'),
                '  T = ',
                str(f'{delchi2t_sign:.5f}'),
                ' Limiting dataset = ',
                dlab,
            ]
        outputfile.writelines(L)
        outputfile.write('\n')

    evgrido(pdf_name)
    fit_pars.irep = fit_pars.irep + 1


def evscan_output(j, tchi, delchi2t, delchi2pos, output_log=None):
    """Write down the final results of the scan"""

    if output_log is None:
        raise FileNotFoundError(
            "A output_log file is needed in order to write down the results of the scan"
        )

    if delchi2t < 0:
        delchi2t_sign = -np.sqrt(-delchi2t) * np.sign(tchi)
    else:
        delchi2t_sign = np.sqrt(delchi2t) * np.sign(tchi)

    with open(output_log, 'a') as outputfile:

        neig = np.rint((j + 1) * np.sign(tchi))

        if np.sign(tchi) > 0:
            if fit_pars.nnpdf_pos:
                L = [
                    'eig[',
                    str(neig),
                    ']  : ',
                    ' t = ',
                    str(f'{tchi:.5f}'),
                    '   T = ',
                    str(f'{delchi2t_sign:.5f}'),
                    ' T^2(pos) = ',
                    str(f'{delchi2pos:.5f}'),
                ]
            else:
                L = [
                    'eig[',
                    str(neig),
                    ']  : ',
                    ' t = ',
                    str(f'{tchi:.5f}'),
                    '   T = ',
                    str(f'{delchi2t_sign:.5f}'),
                ]
        else:
            if fit_pars.nnpdf_pos:
                L = [
                    'eig[',
                    str(neig),
                    '] : ',
                    ' t = ',
                    str(f'{tchi:.5f}'),
                    '  T = ',
                    str(f'{delchi2t_sign:.5f}'),
                    ' T^2(pos) = ',
                    str(f'{delchi2pos:.5f}'),
                ]
            else:
                L = [
                    'eig[',
                    str(neig),
                    '] : ',
                    ' t = ',
                    str(f'{tchi:.5f}'),
                    '  T = ',
                    str(f'{delchi2t_sign:.5f}'),
                ]
        outputfile.writelines(L)
        outputfile.write('\n')

    evgrido()
    fit_pars.irep = fit_pars.irep + 1
