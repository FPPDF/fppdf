#!/usr/bin/env python3

from argparse import ArgumentParser, ArgumentTypeError
import subprocess as sp
import time

from reportengine.utils import yaml_safe
from fixparpdf.utils import _existing_path, init_global_pars



def _no_free_parameters(vp_pdf):
    """Just output the various chi2 and stat information."""
    # TODO: to be removed, this can be done through VP directly I think
    from fixparpdf.chi2s import chi2min
    from fixparpdf.global_pars import chi2_pars, fit_pars, inout_pars
    from fixparpdf.outputs import evgrido, gridout, parsout, plotout, resout_nofit

    if inout_pars.pdout:
        print('Output PD to file...')
        chi2_pars.t0 = False
        fit_pars.pos_const = False
        _ = chi2min([], vp_pdf=vp_pdf)
        return

    elif chi2_pars.chi2ind:
        chi2_pars.t0 = True
        fit_pars.pos_const = False
        chi2t0i = chi2min([], vp_pdf=vp_pdf)

        print('chi2(t0) in (no pos pen):', chi2t0i, chi2t0i / chi2_pars.ndat)
        return

    chi2_pars.t0 = False
    fit_pars.pos_const = False
    chi2expi = chi2min([], vp_pdf=vp_pdf)
    chi2_pars.t0 = True
    chi2t0i = chi2min([], vp_pdf=vp_pdf)
    fit_pars.pos_const = True

    if not inout_pars.pdin:
        fit_pars.pos_const = True

    chi2posi = chi2min([], vp_pdf=vp_pdf)
    pospeni = chi2posi - chi2t0i

    print('chi2(exp) in:', chi2expi + pospeni, (chi2expi + pospeni) / chi2_pars.ndat)
    print('chi2(t0) in:', (chi2t0i + pospeni), (chi2t0i + pospeni) / chi2_pars.ndat)
    print('pos penalty in = ', pospeni, pospeni / chi2_pars.ndat)
    print('chi2(exp) in (no pos pen):', chi2expi, chi2expi / chi2_pars.ndat)
    print('chi2(t0) in (no pos pen):', chi2t0i, chi2t0i / chi2_pars.ndat)

    gridout()
    parsout()
    plotout()
    evgrido()
    resout_nofit(pospeni, chi2t0i, chi2expi, chi2_pars.ndat)


def main():
    parser = ArgumentParser()

    parser.add_argument("runcard", type=_existing_path, help="Runcard defining the run")
    parser.add_argument(
        "--config", type=_existing_path, help="Configuration file to override default parameters"
    )
    args = parser.parse_args()

    # It is possible to call the previous script as well
    #     fixpar_script = Path(__file__).parent / "fixpar_nnpdf.py"
    #     sp.run(["python3", fixpar_script.as_posix(), "--config", args.runcard.as_posix()])

    ##################

    config = yaml_safe.load(args.runcard.open("r"))
    # init global variables
    init_global_pars(config)
    # import global variables to be used in this module
    from fixparpdf.global_pars import inout_pars, pdf_pars, chi2_pars, fit_pars, pdf_closure, dload_pars

    tzero = time.process_time()
    print(tzero)

    # this can be removed
    if inout_pars.readcov:
        return _hessian_error_calculation()

    # TODO: the input files right now are read through the global_pars
    # they can be given to the right functions directly here
    from fixparpdf.inputs import readin

    afi = readin()
    outputfile = open('outputs/buffer/' + inout_pars.label + '.dat', 'a')
    outputfile.write(f"inputnam = {inout_pars.inputnam}\n")

    from validphys.api import API

    from fixparpdf.pdfs import MSHTPDF, initpars, parset, sumrules

    # Prepare the initial PDF that might be used for the computation
    # of the chi2, the minimization, etc.
    if pdf_pars.lhin:
        # If using directly an LHAPDF input, load said PDF
        vp_pdf = API.pdf(pdf=pdf_pars.PDFlabel_lhin)
    else:
        # NB the name is irrelevant since the PDF object doesn't leave the program
        pdfname = f"{inout_pars.label}_run+{pdf_pars.idir}"

        # Take the free parameters and create the parameter set
        parin = initpars()
        pdf_parameters_raw = parset(afi, parin, are_free=pdf_pars.par_isf)
        # Modify it so the sumrules work (is this independent from the parametrization or need the function below?)
        pdf_parameters = sumrules(pdf_parameters_raw)

        vp_pdf = MSHTPDF(name=pdfname, pdf_parameters=pdf_parameters, pdf_function="msht")

    if len(afi) == 0:
        return _no_free_parameters(vp_pdf)

    print('Using LM algorithm...')
    print(f"afi = {afi}")
    from fixparpdf.chi2s import chi2min
    from fixparpdf.levmar import levmar

    if pdf_closure.pdpdf:

        chi2_pars.t0 = False
        fit_pars.pos_const = False
        chi2 = chi2min(afi)
        print('chi2 in:', chi2, (chi2) / chi2_pars.ndat)

        afo = levmar(afi)
        print('afo = ', afo)
    else:
        # Compute the experimental chi2
        chi2_pars.t0 = False
        fit_pars.pos_const = False
        chi2expi = chi2min(afi, vp_pdf=vp_pdf)

        # Now the t0 chi2
        chi2_pars.t0 = True
        chi2t0i = chi2min(afi, vp_pdf=vp_pdf)

        # And now the positivity
        fit_pars.pos_const = True
        chi2posi = chi2min(afi, vp_pdf=vp_pdf)

        # TODO: these three calls should have t0/pos as input parameter to the chi2 function
        # and possibily also the PDF *object* should be an input
        pospeni = chi2posi - chi2t0i

        # Now compute the loss functions (ie., chi2+positivity)
        loss_exp = chi2expi + pospeni
        loss_t0 = chi2posi
        ndat = chi2_pars.ndat
        print(f"chi2(exp) in:{loss_exp:.8} {loss_exp/ndat:.5}")
        print(f"chi2(t0) in:{loss_t0:.8} {loss_t0/ndat:.5}")
        print(f"chi2(exp) in:{pospeni:.5}")

        chi2_pars.t0 = True
        if fit_pars.nnpdf_pos:
            fit_pars.pos_const = True
        else:
            fit_pars.pos_const = False

        chi2_pars.t0 = False
        dload_pars.dflag = 1

        if chi2_pars.t0_noderivin:
            chi2_pars.t0_noderiv = True
            chi2_pars.t0 = False
            dload_pars.dcov = 1

        if inout_pars.pdin:
            chi2_pars.t0 = False
            chi2_pars.t0_noderiv = False
            # chi2_pars.t0=True
            # chi2_pars.t0_noderiv=True
        if inout_pars.pd_output:
            chi2_pars.t0 = False
            chi2_pars.t0_noderiv = False

        chi2_pars.t0 = False
        chi2_pars.t0_noderiv = True

        # TODO this should open a context manager and pass down the file already opened
        outputfile = open('outputs/buffer/' + inout_pars.label + '.dat', 'a')
        outputfile.write("diff2=False")
        print("diff2=False")
        outputfile.write(inout_pars.inputnam)
        outputfile.write("\n")
        afo = levmar(afi)

        if chi2_pars.t0_noderivin and not inout_pars.pdin:
            chi2_pars.t0_noderiv = False
            chi2_pars.t0 = True
        if inout_pars.pd_output:
            chi2_pars.t0 = False
            chi2_pars.t0_noderiv = False

        print('afo = ', afo)

    from fixparpdf.outputs import evgrido, gridout, parsout, plotout, resout

    fit_pars.pos_const = False
    chi2t0f = chi2min(afo)
    chi2_pars.t0 = False
    chi2expf = chi2min(afo)
    fit_pars.pos_const = True
    chi2posf = chi2min(afo)
    pospenf = chi2posf - chi2expf

    gridout()
    parsout()
    plotout()
    evgrido()
    resout(pospeni, pospenf, chi2t0i, chi2expi, chi2t0f, chi2expf, chi2_pars.ndat)
    t1 = time.process_time()
    print(f"time= {t1 - tzero}")


if __name__ == "__main__":
    main()
