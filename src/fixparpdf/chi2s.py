import functools
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.stats as st
from validphys.api import API
from validphys.calcutils import calc_chi2

from fixparpdf.data_theory import compute_theory, dat_calc_rep, pos_calc
from fixparpdf.global_pars import (
    chi2_pars,
    dload_pars,
    fit_pars,
    inout_pars,
    pdf_closure,
    pdf_pars,
    shared_global_data,
)
from fixparpdf.pdfs import MSHTPDF, initpars, parcheck, parset, sumrules


class ParameterError(Exception):
    pass


@functools.cache
def xgrid_calc() -> np.ndarray:

    xgridtot = []

    for inputfile in Path("input/xarr/").glob("grid*.dat"):
        xcheck = np.loadtxt(inputfile)
        xgridtot = np.append(xgridtot, xcheck)

    xgridtot = np.unique(np.sort(xgridtot))

    return xgridtot


def _prepare_parameters(free_parameters: np.ndarray, check=False) -> np.ndarray:
    """
    This wrapper reads the whole set of initial parameters into an array
    and fills in with free parameters, then updates the parameters
    applying the sum rules.
    """
    # Read up the initial set of parameters
    initial_parameters = initpars()
    # parset looks at ``pdf_pars.par_isf`` and fills in the free parameters
    parameters_raw = parset(free_parameters, initial_parameters)
    if check:
        err = parcheck(parameters_raw)
        if err:
            raise ParameterError("Error checking parameters")
    return sumrules(parameters_raw)


def chi2min_fun(afree, jac_calc=False, hess_calc=False, vp_pdf=None):
    """
    Compute the chi2 for a MSHTPDF for the parameters arfree.

    If jac_calc is True, computes also the jacobian d chi / d parfree
    if hess_calc is True, computes also the hessian matrix
    """
    err = False

    jac = np.zeros((pdf_pars.npar_free))
    hess = np.zeros((pdf_pars.npar_free, pdf_pars.npar_free))
    hessp = np.zeros((pdf_pars.npar_free, pdf_pars.npar_free))

    try:
        pdf_parameters = _prepare_parameters(afree, check=True)
    except ParameterError:
        out = 1e50
        return (out, jac, hess, err, hessp)

    # TODO: do we need to set up these variables?
    # I think this is required in error_calc, to update the free parameters
    # corresponding to a specific eigenvalue variation before printing the corresponding grid.

    pdf_pars.pdfparsi = pdf_parameters

    if vp_pdf is None:
        vp_pdf = MSHTPDF(name="pdf", pdf_parameters=pdf_parameters, pdf_function="msht")
    else:
        # TODO when a vp_pdf is given we probably don't need to re-fill in parameters
        # but for now just check they are the same and fail otherwise
        np.testing.assert_allclose(
            pdf_parameters,
            vp_pdf._pdf_parameters,
            err_msg="The given PDF has different parameters from the input",
        )

    if fit_pars.deld_const:
        # TODO when is deld_ used?
        pdf_pars.deld_arr = np.zeros((4 * pdf_pars.npar_free + 1))
        pdf_pars.delu_arr = np.zeros((4 * pdf_pars.npar_free + 1))
        pdf_pars.deld_arr[0] = pdf_pars.pdfparsi[10]
        pdf_pars.delu_arr[0] = pdf_pars.pdfparsi[1]

    pdflabel_arr = np.empty(pdf_pars.npar_free + 1, dtype='U256')

    if jac_calc:
        eps_arr = np.zeros((pdf_pars.npar_free + 1))
        chi2_pars.eps_arr_newmin = eps_arr

        jac, hess, out0, out1, hessp = jaccalc_newmin(hess_calc, vp_pdf)
        out = out0 + out1

    else:

        print("> Calculating chi2 <")
        exp_chi2, pos = chi2totcalc(vp_pdf=vp_pdf)
        out = exp_chi2 + pos
        out0 = exp_chi2
        pdf_pars.idir += 1  # iterate up so new folder

        ndat = chi2_pars.ndat
        print(f"chi2/N_dat={out/ndat:.5}")
        print(f"chi2tot (no pos)={exp_chi2:.5}")
        print(f"pos pen ={pos:.5}")
        print(f"chi2tot (no pos)/N_dat={out0/ndat:.5}")

    return (out, jac, hess, err, hessp)


def chi2corr(datasets, vp_pdf=None, theta_idx=None):
    # TODO: add docstr
    """ """
    if pdf_closure.pdpdf:
        raise Exception
        (out, theorytot, cov, covin) = chi2corr_pdf()
        diffs_out = 0.0
    else:
        (out, theorytot, cov, covin, diffs_out) = chi2corr_global(
            datasets, vp_pdf, theta_idx=theta_idx
        )

    return (out, theorytot, cov, covin, diffs_out)


def chi2corr_pdf():
    raise Exception("This is used in the closure test?")


def chi2corr_ind_plot(imin, imax):

    for i in range(imin, imax + 1):
        dataset_testii = fit_pars.dataset_40[i]
        # print(dataset_testii)

    if chi2_pars.t0:
        cov_gl = dload_pars.covt0
    else:
        cov_gl = dload_pars.covexp

    cov_ind = cov_gl[
        chi2_pars.idat_low_arr[imin] : chi2_pars.idat_up_arr[imax],
        chi2_pars.idat_low_arr[imin] : chi2_pars.idat_up_arr[imax],
    ]
    cov = cov_ind

    dattot = dload_pars.darr_gl[chi2_pars.idat_low_arr[imin] : chi2_pars.idat_up_arr[imax]]

    output = 'BCDMSP_dwsh_dat.dat'
    output = 'outputs/chi2ind_group/' + output
    with open(output, 'w') as outputfile:

        for i in range(0, len(dattot)):
            print(dattot[i], np.sqrt(cov_ind[i, i]))
            L = [str(f'{dattot[i]:10}'), ' ', str(f'{np.sqrt(cov_ind[i,i]):6}')]
            outputfile.writelines(L)
            outputfile.write('\n')

    exit()


def chi2corr_ind_group(imin, imax):

    if chi2_pars.t0:
        cov_gl = dload_pars.covt0
    else:
        cov_gl = dload_pars.covexp

    cov_ind = cov_gl[
        chi2_pars.idat_low_arr[imin] : chi2_pars.idat_up_arr[imax],
        chi2_pars.idat_low_arr[imin] : chi2_pars.idat_up_arr[imax],
    ]
    cov = cov_ind
    cov_inv = la.inv(cov)

    theory = dload_pars.tharr_gl[chi2_pars.idat_low_arr[imin] : chi2_pars.idat_up_arr[imax]]
    dattot = dload_pars.darr_gl[chi2_pars.idat_low_arr[imin] : chi2_pars.idat_up_arr[imax]]

    chi2_pars.ndat = len(dattot)

    diffs = np.array(dattot - theory)
    out = diffs @ cov_inv @ diffs

    return (out, chi2_pars.ndat)


def chi2corr_lab(i):
    """Return dataset index i from the share data"""
    return shared_global_data["data"].datasets[i]


def chi2corr_global(datasets, vp_pdf=None, theta_idx=None):
    """Compute chi2 for the input datasets.
    Datasets must be an iterable of DataSetSpec objects.
    """
    # Make datasets into a tuple if it is a list so that it is cacheable
    datasets = tuple(datasets)

    if fit_pars.nlo_cuts:
        # intersection=[{"dataset_inputs": dload_pars.dscomb, "theoryid": 212}]
        intersection = [{"dataset_inputs": dload_pars.dscomb, "theoryid": 200}]

    if vp_pdf is None:
        raise Exception("A PDF needs to be given")

    #     din=[fit_pars.dataset_40[imin]]
    #     if dload_pars.dflag==1:
    #         # print('DLOAD')
    #         for i in range(imin+1,imax+1):
    #             din.append(fit_pars.dataset_40[i])
    #             # print(i,fit_pars.dataset_40[i])
    #         dload_pars.dscomb=din

    dload_pars.ifk = 0
    if dload_pars.dflag == 1:
        chi2_pars.idat = 0

    # Compute a list of theory predictions for all datasets between imin and imax, for the given PDF
    theorytot = compute_theory(datasets, vp_pdf, theta_idx=theta_idx)

    # Get the covmat for all the dataset we have calculated chi2 for. The order is the same as the vector of theories
    cov = shared_global_data["data"].produce_covmat(
        pdf=vp_pdf, datasets=datasets, use_t0=chi2_pars.t0
    )
    covin = la.inv(cov)

    # TODO: remvoe the dload global state

    if chi2_pars.t0:
        dload_pars.covt0 = cov
        dload_pars.covt0_inv = covin
        dload_pars.dcov = 0
        # LHL ADDED NEW - so that t0 def is used in minimisation
    elif chi2_pars.t0_noderiv:
        if dload_pars.dcov == 1:
            print("Computing t0 cov...", end="")
            try:
                cov = shared_global_data["data"].produce_covmat(
                    pdf=vp_pdf, datasets=datasets, use_t0=True
                )
                covin = la.inv(cov)
            except (la.LinAlgError, ValueError) as err:
                print('t0 cov may be ill behaved, trying exp cov instead...')
                cov = shared_global_data["data"].produce_covmat(datasets=datasets, use_t0=False)
                covin = la.inv(cov)

            # TODO: these two variables can be removed?
            dload_pars.covt0 = cov
            dload_pars.covt0_inv = covin
            print('...finished')
            dload_pars.dcov = 0
    else:
        dload_pars.covexp = cov
        dload_pars.covexp_inv = covin

    if dload_pars.dflag == 1:
        print('DLOAD')
        if fit_pars.pseud:
            fit_pars.pseud = False
            dattot0 = dat_calc_rep(datasets, genrep=False)
            fit_pars.pseud = True
            dattot = dat_calc_rep(datasets, genrep=True)
        else:
            dattot = dat_calc_rep(datasets, genrep=False)
        chi2_pars.ndat = len(dattot)
        dload_pars.darr_gl = dattot

    else:
        print('NO DLOAD')
        dattot = dload_pars.darr_gl

    if inout_pars.pdout:
        outputfile = open('outputs/pseudodata/' + pdf_closure.pdlabel + '.dat', 'w')

        #       pseudodata calculate using internal NNPDF routing - shift defined wrt real data so redefine as wrt pseudodata
        if fit_pars.pseud:
            p_data = theorytot + dattot - dattot0
            for i in range(0, len(dattot)):
                outputfile.write(str(p_data[i]))
                outputfile.write('\n')
        # calculated using own routine
        else:
            for i in range(0, len(theorytot)):
                outputfile.write(str(theorytot[i]))
                outputfile.write('\n')

    if dload_pars.dflag == 1:
        if inout_pars.pdin:
            inputfile = 'outputs/pseudodata/' + pdf_closure.pdlabel + '.dat'
            print('testdir =', pdf_closure.pdlabel)
            distin = np.loadtxt(inputfile)
            dattot = distin
            dload_pars.darr_gl = dattot

            if pdf_closure.pdfscat:

                covin = la.inv(cov)
                _, eig = la.eigh(covin)
                cov_d = la.inv(eig) @ covin @ eig

                dattot_d = la.inv(eig) @ dattot

                for i in range(0, len(cov)):
                    dattot_d[i] = dattot_d[i] + np.random.normal() * np.sqrt(1.0 / cov_d[i, i])

                dattot = eig @ dattot_d

                dload_pars.darr_gl = dattot

        dload_pars.dflag = 0
    else:
        dattot = dload_pars.darr_gl

    chi2_pars.ndat = len(dattot)

    diffs = dattot - theorytot

    dload_pars.tharr_gl = theorytot

    # Compute chi2
    try:
        out = diffs @ covin @ diffs
    except la.LinAlgError as err:
        print(err)
        print('t0 cov may be ill behaved, trying exp cov instead...')
        inp = dict(
            dataset_inputs=dload_pars.dscomb, theoryid=fit_pars.theoryidi, use_cuts="internal"
        )
        cov = API.dataset_inputs_covmat_from_systematics(**inp)
        try:
            out = calc_chi2(la.cholesky(cov, lower=True), diffs)
            print('out=', out)
            print('cov=', cov)
            print('diffs=', diffs)
            print('dattot =', dattot)
            print('theorytot =', theorytot)
        except la.LinAlgError as erra:
            print(erra)
            print('No, theory ill behaved - set chi^2=1e50')
            out = 1e50
            print('out=', out)
    except ValueError as err1:
        print(err1)
        print('t0 cov ill behaved, trying exp cov instead...')
        inp = dict(
            dataset_inputs=dload_pars.dscomb, theoryid=fit_pars.theoryidi, use_cuts="internal"
        )
        cov = API.dataset_inputs_covmat_from_systematics(**inp)
        try:
            out = calc_chi2(la.cholesky(cov, lower=True), diffs)
            print('out=', out)
            print('cov=', cov)
            print('diffs=', diffs)
            print('dattot =', dattot)
            print('theorytot =', theorytot)
        except ValueError as err1a:
            print(err1a)
            print('No, theory ill behaved - set chi^2=1e50')
            out = 1e50
            print('out=', out)

    return (out, theorytot, cov, covin, diffs)


def hess_fun(afree):

    hess_calc = True
    jac_calc = True
    out = chi2min_fun(afree, jac_calc, hess_calc)[2]
    print('Hessian = ', out)
    return out


def jac_fun(afree):

    hess_calc = False
    jac_calc = True
    out = chi2min_fun(afree, jac_calc, hess_calc)[1]
    print('Jacobian = ', out)
    return out


def chi2min(afree=None, vp_pdf=None):
    """Compute the chi2 that will be used during the minimization
    But disable the computation of the hessian and jacobian.
    Takes as input the free parameters of the problem.
    """
    hess_calc = False
    jac_calc = False
    outarr = chi2min_fun(afree, jac_calc, hess_calc, vp_pdf=vp_pdf)
    out = outarr[0]
    return out


def chilim_calc(nd):

    n = np.rint(nd)

    frac = 1e-3

    chisq = 0.0
    sum = 0.0
    i = 0
    cl50 = 0.0
    cl68 = 0.0

    while cl68 == 0.0:
        chisq += frac * n
        sum += st.chi2.pdf(chisq, nd) * frac * n
        # print(sum,chisq)
        if sum > 0.5 and i == 0:
            cl50 = chisq
            i = 1
        if sum > 0.68:
            cl68 = chisq

    return (cl50, cl68)


def chilim_sort():

    output = 'test.dat'
    output = 'outputs/chi2ind_group/' + inout_pars.label + '.dat'
    with open(output, 'w') as outputfile:

        for i in range(0, len(chi2_pars.chi_ind_arr)):
            print(i, chi2_pars.clnd_arr[i], chi2_pars.cldataset_arr[i], chi2_pars.chi_ind_arr[i])
            L = [
                str(f'{i:2}'),
                ' ',
                str(f'{chi2_pars.clnd_arr[i]:4}'),
                ' ',
                str(f'{chi2_pars.cldataset_arr[i]:45}'),
                ' ',
                str(f'{chi2_pars.chi_ind_arr[i]:5}'),
            ]
            outputfile.writelines(L)
            outputfile.write('\n')
            # print(chi2_pars.chi0_ind_arr[i])

    print(sum(chi2_pars.chi_ind_arr))

    # exit()


def chilim_fill(nd, chi, dlab):

    (cl50, cl68) = chilim_calc(nd)
    # print(nd,cl50,cl68,cl68/cl50-1.)

    # (cl50,cl68)=chilim_calc(88)
    # print(88,cl50,cl68,cl68/cl50-1.)
    # (cl50,cl68)=chilim_calc(59)
    # print(59,cl50,cl68,cl68/cl50-1.)
    # (cl50,cl68)=chilim_calc(82)
    # print(82,cl50,cl68,cl68/cl50-1.)
    # (cl50,cl68)=chilim_calc(84)
    # print(84,cl50,cl68,cl68/cl50-1.)
    # (cl50,cl68)=chilim_calc(115)
    # print(115,cl50,cl68,cl68/cl50-1.)
    # (cl50,cl68)=chilim_calc(123)
    # print(123,cl50,cl68,cl68/cl50-1.)
    # (cl50,cl68)=chilim_calc(463)
    # print(463,cl50,cl68,cl68/cl50-1.)

    # exit()

    if chi2_pars.L0:
        chilim = cl68 - cl50
    else:
        chilim = (cl68 / cl50 - 1.0) * chi
    # print(chi,chilim)

    chi2_pars.chilim_arr.append(chilim)
    chi2_pars.cldataset_arr.append(dlab.name)
    chi2_pars.clnd_arr.append(nd)
    chi2_pars.chi0_ind_arr.append(chi)
    # print(i,chilim/chi,cl68,cl50,cl68/cl50,nd)


def chi2totcalc(vp_pdf=None):
    """If a VP pdf is used, use VP to compute the chi2"""

    # Prepare an array to save the chi2 for each dataset
    chiarr = np.zeros(fit_pars.imaxdat - fit_pars.imindat)
    datasets = shared_global_data["data"].datasets[fit_pars.imindat : fit_pars.imaxdat]
    chiarr[0] = chi2corr(datasets, vp_pdf=vp_pdf)[0]

    ndtot = 0
    chi2totind = 0.0
    if chi2_pars.chi2ind:
        chi2_pars.chi_ind_arr = []
        chi2_pars.idat = 0.0

        # Get the full covmat and make it into a dataframe indexed by label
        # The compute the chi2 for each dataset
        if chi2_pars.t0:
            pdf = vp_pdf
        else:
            pdf = None

        covmat = shared_global_data["data"].produce_covmat(datasets=datasets, pdf=pdf, use_t0=chi2_pars.t0)
        labels = [i.name for i in datasets for _ in range(i.load_commondata().ndata)]
        dfcov = pd.DataFrame(covmat, columns=labels, index=labels)

        idat = 0
        for dataset in datasets:
            # NB: using double squares here to preserve the matrix-structure even in ndata=1 datasets
            partial_cov = dfcov.loc[[dataset.name], [dataset.name]]
            ndat = dataset.load_commondata().ndata

            fdat = idat + ndat
            # Compute the chi2 of the individual dataset:
            theory = dload_pars.tharr_gl[idat:fdat]
            # This should be equal to the commondata central value unless we are loading closure data
            dattot = dload_pars.darr_gl[idat:fdat]

            diffs = np.array(dattot - theory)
            cov_inv = la.inv(partial_cov)

            chi2 = diffs @ cov_inv @ diffs

            chi2_pars.chi_ind_arr.append(chi2)
            chi2totind += chi2
            ndtot += ndat

            if chi2_pars.calc_cl:
                chilim_fill(ndat, chi2, dataset)


    out0 = np.sum(chiarr)
    out0 = np.abs(out0)

    out1 = 0.0
    if fit_pars.pos_const:
        pos_data = shared_global_data["posdata"].datasets
        out1 = np.sum(pos_calc(pos_data, vp_pdf))

    chi2_pars.chi_pos1 = out1

    if chi2_pars.calc_cl:
        chi2_pars.chi_pos0 = out1
    chi2_pars.calc_cl = False

#     if fit_pars.deld_const:
#         # dv
#         out0 = out0 + del_pen(pdf_pars.deld_arr[0])[0]
#         # uv
#         out0 = out0 + del_pen(pdf_pars.delu_arr[0])[0]

    return (out0, out1)


def hess_ij_calc_d2(diffi, diffj, cov, covin):

    dattot = dload_pars.darr_gl

    diffp = diffi + diffj
    diffm = diffi - diffj

    outp = diffp @ covin @ diffp
    outm = diffm @ covin @ diffm

    out = outp - outm
    out = out / 4.0
    out = out * 2.0

    return out


def hess_ij_calc_not0_new(theoryi, theoryj, outi, outj, cov, covin):

    diffij = theoryi - theoryj
    outij = diffij @ covin @ diffij
    outa = outi / 2.0 + outj / 2.0 - outij

    out = outa

    return out


@nb.njit
def hess_ij_calc_newmin(diffi, diffj, covin):
    """Compute off-diagonal entries of the hessian"""
    out = diffi @ covin @ diffj * 2.0
    return out


def hess_ij_calc_new(theory0, theoryi, theoryj, cov0, covi, covj, out1j, out1i, outi, outj):

    dattot = dload_pars.darr_gl

    diffij = theoryi - theoryj
    outij = calc_chi2(la.cholesky(cov0, lower=True), diffij)

    outa = outi + outj - outij

    diff1 = theory0 - dattot
    diff2i = theoryi - theory0
    out2i = calc_chi2(la.cholesky(covj, lower=True), diff2i)
    diff3i = theoryi - dattot
    out3i = calc_chi2(la.cholesky(covj, lower=True), diff3i)

    out10 = calc_chi2(la.cholesky(cov0, lower=True), diff1)
    out20 = outi
    out30 = calc_chi2(la.cholesky(cov0, lower=True), diff3i)

    out1ii = out1i - out10
    out2i = out2i - out20
    out3i = out3i - out30

    outb = out3i - out2i - out1ii

    diff2j = theoryj - theory0
    out2j = calc_chi2(la.cholesky(covi, lower=True), diff2j)
    diff3j = theoryj - dattot
    out3j = calc_chi2(la.cholesky(covi, lower=True), diff3j)

    out20 = out2j
    out30 = calc_chi2(la.cholesky(cov0, lower=True), diff3j)

    out1jj = out1j - out10
    out2j = out2j - out20
    out3j = out3j - out30

    outc = out3j - out2j - out1jj

    out = outa + outb + outc

    return out


def hess_ii_calc_d2(diff2, cov, covin):

    dattot = dload_pars.darr_gl
    #    diff2=tp-tm
    # out2=calc_chi2(la.cholesky(cov, lower=True), diff2)
    out2 = diff2 @ covin @ diff2
    out = 2.0 * out2

    return out


@nb.njit
def hess_ii_calc_newmin(diff2, covin):
    """Compute diagonal entries of the hessian
    as dT/dpar . cov0^-1 . dT/dpar
    """
    out2 = diff2 @ covin @ diff2
    # print(np.sum(diff2),out2)
    # print(covin)
    out = 2.0 * out2

    return out


def hess_ii_calc_not0(theory0, theoryi, cov, covin):

    dattot = dload_pars.darr_gl

    diff2 = theoryi - theory0

    # out2=calc_chi2(la.cholesky(cov, lower=True), diff2)
    out2 = diff2 @ covin @ diff2

    # print(diff2)
    # outputfile=open('test_oldmin.dat','w')
    # for i in range(0,len(diff2)):
    #     outputfile.write(str(diff2[i]))
    #     outputfile.write('\n')
    # print(np.sum(diff2),out2)
    # print(covin)

    out = 2.0 * out2

    return out


def hess_ii_calc_t0(theory0, theoryi, cov0, covi, out10):

    dattot = dload_pars.darr_gl

    diff1 = theory0 - dattot
    out1i = calc_chi2(la.cholesky(covi, lower=True), diff1)
    diff2 = theoryi - theory0
    out2i = calc_chi2(la.cholesky(covi, lower=True), diff2)
    diff3 = theoryi - dattot
    out3i = calc_chi2(la.cholesky(covi, lower=True), diff3)

    out20 = calc_chi2(la.cholesky(cov0, lower=True), diff2)
    out30 = calc_chi2(la.cholesky(cov0, lower=True), diff3)

    out1 = out1i - out10
    out2 = out2i - out20 * 2.0
    out3 = out3i - out30

    out = 2.0 * (out3 - out2 - out1)

    return (out, out1i, out20)


def betacalc_not0(theory0, theoryi, cov0):

    print('test')

    dattot = dload_pars.darr_gl

    diff1 = theory0 - dattot
    out1 = calc_chi2(la.cholesky(cov0, lower=True), diff1)
    diff2 = theoryi - theory0
    out2 = calc_chi2(la.cholesky(cov0, lower=True), diff2)
    diff3 = theoryi - dattot
    out3 = calc_chi2(la.cholesky(cov0, lower=True), diff3)

    out = out3 - out2 - out1

    #    test above is correct (below is slower so don't use)
    #    difft1=theoryi-theory0
    #    outt1=2.*difft1@la.inv(cov0)@diff1

    #    print('test',out,outt1)

    return out


def betacalc(theory0, theoryi, cov0, covi):

    dattot = dload_pars.darr_gl

    diff1 = theory0 - dattot
    out1 = calc_chi2(la.cholesky(cov0, lower=True), diff1)
    diff2 = theoryi - theory0
    out2 = calc_chi2(la.cholesky(cov0, lower=True), diff2)
    diff3 = theoryi - dattot
    out3 = calc_chi2(la.cholesky(cov0, lower=True), diff3)

    diffc = theory0 - dattot
    outc = calc_chi2(la.cholesky(covi, lower=True), diffc)

    out = out3 - out2 - out1
    out = out3 - out2 - 2.0 * out1 + outc  # outc-out1: dcov/dpar

    #    test above is correct (below is slower so don't use)

    #    difft1=theoryi-theory0
    #    outt1=2.*difft1@la.inv(cov0)@diff1
    #    test=outc-out1+outt1
    #    print('test',out,test)

    return out


def jaccalc_newmin(hess_calc: bool, vp_pdf: MSHTPDF):
    """Compute the derivative of the chi2 wrt the free parameters
    (given in Eq.6 https://people.duke.edu/~hpgavin/lm.pdf) and the Hessian
    (given after Eq.9 https://people.duke.edu/~hpgavin/lm.pdf)

    Returns:
        jaccarr: array of dchi2/dparameter
        hessarr: hessian matrix
        chi0: chi2 without positivity
        central_pos_penalty: positivity penalty for the central value
        hessparr: nparxnpar empty matrix
    """
    print('JACCALC NEWMIN')

    imin = fit_pars.imindat
    imax = fit_pars.imaxdat
    datasets = shared_global_data["data"].datasets[imin:imax]

    # compute chi2, theory predictions, cov, cov_inv, and (theory - data)
    (chi0, _, _, cov0in, diffs0) = chi2corr(datasets, vp_pdf)

    print(f"Free parameters: {pdf_pars.npar_free}")

    # Allocate the array for the jacobian
    jacarr = np.zeros((pdf_pars.npar_free))

    # Allocate the arrays to compute the hessian
    tarr = []
    hessarr = np.zeros((pdf_pars.npar_free, pdf_pars.npar_free))
    jacarr = np.zeros((pdf_pars.npar_free))

    # Prepare positivity output
    if fit_pars.pos_const:
        positivity_data = shared_global_data["posdata"].datasets
        positivity_points = pos_calc(positivity_data, vp_pdf)
        # Positive points
        positivity_indexes = positivity_points > 0.0
        positivity_loss = np.sum(positivity_points)
    else:
        positivity_loss = 0.0

    for ip in range(pdf_pars.npar_free):
        parameter_index = pdf_pars.par_free_i[ip]

        # compute J_i, i.e. the derivative of the theory predictions wrt the free parameter i=parameter_index (theory)
        # NB: at the moment the derivative is computed at the level of the observable for the theta_idx=parameter_index
        # but in principle we could do it at the level of the PDF if central_predictions could take a second PDF
        theory = compute_theory(datasets, vp_pdf=vp_pdf, theta_idx=parameter_index)
        tarr.append(theory)

        jac_result = -2.0 * theory @ cov0in @ diffs0

        if hess_calc:
            # compute the diagonal entries of the hessian, defined as J cov0in J^T
            tdiag = hess_ii_calc_newmin(theory, cov0in)
            hessarr[ip, ip] = tdiag

            # And the non-diagonal terms
            for jp in range(ip):
                hij = hess_ij_calc_newmin(theory, tarr[jp], cov0in)
                hessarr[ip, jp] = hij
                hessarr[jp, ip] = hij

        # Finally compute the penalties for the points with negative contributions:
        if fit_pars.pos_const:
            tmp = pos_calc(positivity_data, vp_pdf, theta_idx=parameter_index)
            # Take only the indexes which contribute
            jac_result += np.sum(tmp[positivity_indexes])

        jacarr[ip] = jac_result

    hessparr = np.zeros((pdf_pars.npar_free, pdf_pars.npar_free))

    return (jacarr, hessarr, chi0, positivity_loss, hessparr)


def jaccalc_d2(label_arr, label_arrm, eps_arr, hess_calc, il, ih):

    difft = np.zeros((pdf_pars.npar_free + 1))
    chiarr = np.zeros((pdf_pars.npar_free + 1))
    chiarrm = np.zeros((pdf_pars.npar_free + 1))
    hessarr = np.zeros((pdf_pars.npar_free + 1, pdf_pars.npar_free + 1))

    pdf_pars.PDFlabel = label_arr[0].strip()

    pdf_pars.iPDF = 0
    (chiarr[0], theory0, cov0, cov0in, diffs_out) = chi2corr(il, ih - 1)
    difft = [0.0]

    out0 = chiarr[0]

    for ip in range(1, pdf_pars.npar_free + 1):
        pdf_pars.PDFlabel = label_arr[ip].strip()
        pdf_pars.iPDF = ip
        (chiarr[ip], theoryp, cov, covin, diffs_out) = chi2corr(il, ih - 1)
        pdf_pars.PDFlabel = label_arrm[ip].strip()
        pdf_pars.iPDF = ip + pdf_pars.npar_free
        (chiarrm[ip], theorym, cov, covin, diffs_out) = chi2corr(il, ih - 1)

        theoryd = theoryp - theorym
        difft.append(theoryd)

    for ip in range(1, pdf_pars.npar_free + 1):

        chiarr[ip] = chiarr[ip] - chiarrm[ip]
        chiarr[ip] = chiarr[ip] / eps_arr[ip] / 2.0

    chiarr = np.delete(chiarr, 0)
    jacarr = chiarr

    if hess_calc:

        for ip in range(1, pdf_pars.npar_free + 1):

            if ip == 1:
                tii0 = hess_ii_calc_d2(difft[ip], cov0, cov0in)
                tii = [tii0]
            else:
                tii0 = hess_ii_calc_d2(difft[ip], cov0, cov0in)
                tii.append(tii0)

            for jp in range(1, ip + 1):

                if ip == jp:
                    hii = tii[ip - 1]
                    hii = hii / np.power(eps_arr[ip], 2) / 4.0
                    hessarr[ip, jp] = hii
                else:
                    #                    hij=hess_ij_calc_d2(tarrp[ip],tarrm[ip],tarrp[jp],tarrm[jp],cov0)
                    hij = hess_ij_calc_d2(difft[ip], difft[jp], cov0, cov0in)
                    hij = hij / eps_arr[ip] / eps_arr[jp] / 4.0
                    hessarr[ip, jp] = hij

        hessarr = np.delete(hessarr, 0, 0)
        hessarr = np.delete(hessarr, 0, 1)
        hessarr = hessarr + hessarr.T - np.diag(hessarr.diagonal())

    out1 = 0.0
    if fit_pars.pos_const:
        raise Exception("Not reached")
        chiarr = np.zeros((pdf_pars.npar_free + 1))
        chiarrm = np.zeros((pdf_pars.npar_free + 1))
        hessparr = np.zeros((pdf_pars.npar_free + 1, pdf_pars.npar_free + 1))

        for ip in range(1, pdf_pars.npar_free + 1):

            pdf_pars.PDFlabel = label_arr[ip].strip()
            out31 = pos_calc(fit_pars.pos_data31)
            if fit_pars.pos_40:
                out40 = pos_calc(fit_pars.pos_data40)
                chi2pos = out31 + out40
                out1 = chi2pos
            else:
                out1 = out31
                pdf_pars.PDFlabel = label_arrm[ip].strip()

            chiarr[ip] = out1

            pdf_pars.PDFlabel = label_arrm[ip].strip()
            out31 = pos_calc(fit_pars.pos_data31)
            if fit_pars.pos_40:
                out40 = pos_calc(fit_pars.pos_data40)
                chi2pos = out31 + out40
                out1 = chi2pos
            else:
                out1 = out31
                pdf_pars.PDFlabel = label_arrm[ip].strip()

            chiarrm[ip] = out1

        for ip in range(1, pdf_pars.npar_free + 1):

            hessparr[ip, ip] = (chiarr[ip] - 2.0 * chiarr[0] + chiarrm[ip]) / np.power(
                eps_arr[ip], 2
            )
            chiarr[ip] = chiarr[ip] - chiarrm[ip]
            chiarr[ip] = chiarr[ip] / eps_arr[ip] / 2.0

        if chi2_pars.add_hessp:

            for ip in range(1, pdf_pars.npar_free + 1):
                for jp in range(1, pdf_pars.npar_free + 1):
                    if ip == jp:
                        hessarr[ip - 1, ip - 1] = hessarr[ip - 1, ip - 1] + hessparr[ip, ip]

        chiarr = np.delete(chiarr, 0)
        jacarr = jacarr + chiarr

    if fit_pars.deld_const:
        (chi0d, chi0u, diffd, diffu, hessd, hessu, idv, iuv) = del_pen_calc()
        out1 = out1 + chi0d
        out1 = out1 + chi0u
        print(chi0d, chi0u)
        if idv > 0:
            jacarr[idv - 1] = jacarr[idv - 1] + diffd
            # print(id,diffd)
            hessarr[idv - 1, idv - 1] = hessarr[idv - 1, idv - 1] + hessd
        if iuv > 0:
            jacarr[iuv - 1] = jacarr[iuv - 1] + diffu
            # print(iu,diffu)
            hessarr[iuv - 1, iuv - 1] = hessarr[iuv - 1, iuv - 1] + hessu

    # print(jacarr)
    # print(hessarr)
    # os.quit()

    return (out0, out1, jacarr, hessarr)


def hess_zeros(hess):

    hessi = hess.copy()
    dimh = len(hess)

    result = np.all((hess == 0), axis=1)

    for i in range(len(result)):
        if result[i]:
            print('Row: ', i)
            print(hess[i, :])
    #            hessi[i,:]=1e-10
    #            hessi[:,i]=1e-10

    idx = np.argwhere(np.all(hessi[..., :] == 0, axis=0))
    #    print('idx',idx)
    hessd = np.delete(hessi, idx, axis=1)
    hessd1 = np.delete(hessd, idx, axis=0)
    #    print('hessi',hessi)
    #    print('hessd',hessd)
    #    print('hessd1',hessd1)

    hessin = la.inv(hessd1)

    result = hessin.copy()

    output_size = (dimh, dimh)
    indices = (idx, idx)

    result = np.zeros(output_size)
    existing_indices = [
        np.setdiff1d(np.arange(axis_size), axis_indices, assume_unique=True)
        for axis_size, axis_indices in zip(output_size, indices)
    ]
    result[np.ix_(*existing_indices)] = hessin

    hessd2 = result

    #    print('hessd2',hessd2)

    return hessd2


def calc_covmat_t0(inpt0):
    """Compute covmat given an object inpt0 containing, among other things, the dataset_inputs info.

    Intersection cuts might be included in inpt0.
    """
    # TODO but are they being used?
    if inpt0["use_cuts"] != "internal":
        raise ValueError("intersection cuts not implemented at this point")

    # TODO originally this code used dload_pars.dscomb for some stuff, inpt0 for other and the predictions from what was stored
    # in the global state. Now it will use _only_ the inpt0, it should've always been compatible anyway.
    dnames = [i["dataset"] for i in inpt0["dataset_inputs"]]
    pdfset = inpt0["t0pdfset"]
    use_t0 = inpt0["use_t0"]
    return shared_global_data["data"].produce_covmat(pdf=pdfset, names=tuple(dnames), use_t0=use_t0)
