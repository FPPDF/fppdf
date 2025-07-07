from validphys.api import API
from scipy.optimize import minimize
from reportengine.utils import yaml_safe
import time
import argparse
from validphys.loader import _get_nnpdf_profile

from fixparpdf import global_pars


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="Path to the config file")
args = parser.parse_args()

##################
# Loading yaml config
##################
with open(args.config, 'r') as file:
    config = yaml_safe.load(file)

# Read the necessary inputs
_input_config = config.get("inout_parameters", {})
_basis_config = config.get("basis pars", {})
_pdf_pars = config.get("pdf pars", {})
_chi2_pars = config.get("chi2_parameters", {})
_fit_pars = config.get("fit_parameters", {})

# Remove deprecated parameters just in case
_pdf_pars.pop("uselha", None)

# Add extra flags that go into the input
_pseudodata_config = config.get("pseudodata flags", None)
if _pseudodata_config:
    _input_config["pdout"] = _pseudodata_config.get("pdfout")
    _input_config["pdin"] = _pseudodata_config.get("pdin")
    _input_config["pd_output"] = _pseudodata_config.get("pd_output", False)

_pdf_closure_config = config.get("pdf closure", {})
if _pdf_closure_config:
    _pdf_pars["pdfscat"] = _pdf_closure_config.get("pdfscat", False)
    _pdf_pars["pdflabel"] = _pdf_closure_config.get("pdflabel")
    _pdf_pars["pdpdf"] = _pdf_closure_config.get("pdpdf", False)

_full_dataset = config["dataset_inputs"]
_pos_dataset = config["posdatasets"]
_fit_pars["dataset_40"] = _full_dataset
_fit_pars["pos_data40"] = _pos_dataset

# Instantiate the global configuration
global_pars.inout_pars = inout_pars = global_pars.InoutPars(**_input_config)
global_pars.basis_pars = global_pars.BasisPars(**_basis_config)
global_pars.pdf_pars = pdf_pars = global_pars.PDFPars(**_pdf_pars)
global_pars.chi2_pars = chi2_pars = global_pars.Chi2Pars(**_chi2_pars)

fit_pars = global_pars.fit_pars = global_pars.FitPars(**_fit_pars)

# Some quick checks
if global_pars.pdf_pars.lhin != global_pars.fit_pars.fixpar:
    raise ValueError

pdf_closure = global_pars.pdf_closure
min_pars = global_pars.min_pars
dload_pars = global_pars.dload_pars

# WARNING
# Since global_pars is imported by various of the modules below, they cannot be imported
# until the global pars are set
# TODO: lift the entire config out 
from fixparpdf.outputs import gridout, parsout, plotout, evgrido, resout, resout_nofit
from fixparpdf.inputs import readincov, readin
from fixparpdf.levmar import levmar
from fixparpdf.pdfs import initpars, parset, sumrules, MSHTPDF
from fixparpdf.chi2s import chi2min
from fixparpdf.error_calc import hesserror_new, hesserror_dynamic_tol_new

profile = _get_nnpdf_profile(None)

tzero = time.process_time()

print('inputnam = ', inout_pars.inputnam)


##### TODO: start populating the shared data here
global_pars.shared_populate_data()

if inout_pars.readcov:
    chi2_pars.t0 = True
    # fit_pars.pos_const=False
    if fit_pars.nnpdf_pos:
        fit_pars.pos_const = True
    else:
        fit_pars.pos_const = False
    (afi, hess, jac) = readincov()
    if chi2_pars.dynamic_tol:
        # hesserror_dynamic_tol(afi,hess,jac)
        hesserror_dynamic_tol_new(afi, hess, jac)
    else:
        hesserror_new(afi, hess, jac)
        # hesserror(afi,hess,jac)

else:
    afi = readin()

outputfile = open('outputs/buffer/' + inout_pars.label + '.dat', 'a')
outputfile.write("inputnam = ")
outputfile.write(inout_pars.inputnam)
outputfile.write("\n")

# msht_kfs()

print(tzero)

use_levmar = True

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


if inout_pars.readcov:

    print('finish!')

elif len(afi) == 0:  # no free pars

    if inout_pars.pdout:

        print('Output PD to file...')
        chi2_pars.t0 = False
        fit_pars.pos_const = False
        chi2expi = chi2min(afi, vp_pdf=vp_pdf)

    elif chi2_pars.chi2ind:
        chi2_pars.t0 = True
        fit_pars.pos_const = False
        chi2t0i = chi2min(afi, vp_pdf=vp_pdf)

        print('chi2(t0) in (no pos pen):', chi2t0i, chi2t0i / chi2_pars.ndat)
        t1 = time.process_time()
        print('time= ', t1 - tzero)
    else:

        chi2_pars.t0 = False
        fit_pars.pos_const = False
        chi2expi = chi2min(afi, vp_pdf=vp_pdf)
        chi2_pars.t0 = True
        chi2t0i = chi2min(afi, vp_pdf=vp_pdf)
        fit_pars.pos_const = True

        if inout_pars.pdin:

            chi2posi = chi2min(afi, vp_pdf=vp_pdf)
            pospeni = chi2posi - chi2t0i

            # print('chi2(exp) in:',chi2expi,(chi2expi)/chi2_pars.ndat)
            # t1=time.process_time()
            # print('time= ',t1-tzero)

            print('chi2(exp) in:', chi2expi + pospeni, (chi2expi + pospeni) / chi2_pars.ndat)
            print('chi2(t0) in:', (chi2t0i + pospeni), (chi2t0i + pospeni) / chi2_pars.ndat)
            print('pos penalty in = ', pospeni, pospeni / chi2_pars.ndat)
            print('chi2(exp) in (no pos pen):', chi2expi, chi2expi / chi2_pars.ndat)
            print('chi2(t0) in (no pos pen):', chi2t0i, chi2t0i / chi2_pars.ndat)
            t1 = time.process_time()
            print('time= ', t1 - tzero)

            gridout()
            parsout()
            plotout()
            evgrido()
            resout_nofit(pospeni, chi2t0i, chi2expi, chi2_pars.ndat)

        else:
            fit_pars.pos_const = True

            chi2posi = chi2min(afi, vp_pdf=vp_pdf)
            pospeni = chi2posi - chi2t0i

            print('chi2(exp) in:', chi2expi + pospeni, (chi2expi + pospeni) / chi2_pars.ndat)
            print('chi2(t0) in:', (chi2t0i + pospeni), (chi2t0i + pospeni) / chi2_pars.ndat)
            print('pos penalty in = ', pospeni, pospeni / chi2_pars.ndat)
            print('chi2(exp) in (no pos pen):', chi2expi, chi2expi / chi2_pars.ndat)
            print('chi2(t0) in (no pos pen):', chi2t0i, chi2t0i / chi2_pars.ndat)
            t1 = time.process_time()
            print('time= ', t1 - tzero)

            gridout()
            parsout()
            plotout()
            evgrido()
            resout_nofit(pospeni, chi2t0i, chi2expi, chi2_pars.ndat)

elif use_levmar:

    print('Using LM algorithm...')
    print('afi = ', afi)

    if pdf_closure.pdpdf:

        chi2_pars.t0 = False
        fit_pars.pos_const = False
        chi2 = chi2min(afi)
        print('chi2 in:', chi2, (chi2) / chi2_pars.ndat)

        # afo=levmar(afi)
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

        # chi2_pars.t0=False
        # afo=levmar(afi)
        # print('afo = ', afo)

        chi2_pars.t0 = False
        dload_pars.dflag = 1

        if chi2_pars.t0_noderivin:
            chi2_pars.t0_noderiv = True
            chi2_pars.t0 = False
            dload_pars.dcov = 1

        if fit_pars.nnpdf_pos and False:
            # TODO: there are a few calls to levmar here that need to be treated
            if inout_pars.pdin:
                chi2_pars.t0 = False
                chi2_pars.t0_noderiv = False
            fit_pars.lampos = 1e1
            print('lampos = 1e1')
            outputfile = open('outputs/buffer/' + inout_pars.label + '.dat', 'a')
            outputfile.write("lampos = 1e1")
            outputfile.write(inout_pars.inputnam)
            outputfile.write("\n")
            afo = levmar(afi)
            fit_pars.lampos = 1e2
            print('lampos = 1e2')
            outputfile = open('outputs/buffer/' + inout_pars.label + '.dat', 'a')
            outputfile.write("lampos = 1e2")
            outputfile.write(inout_pars.inputnam)
            outputfile.write("\n")
            afo = levmar(afo)
            fit_pars.lampos = 1e3
            print('lampos = 1e3')
            outputfile = open('outputs/buffer/' + inout_pars.label + '.dat', 'a')
            outputfile.write("lampos = 1e3")
            outputfile.write(inout_pars.inputnam)
            outputfile.write("\n")
            afo = levmar(afo)
            if chi2_pars.t0_noderivin:
                print('lampos = 1e3, t0 full')
                fit_pars.lampos = 1e3
                chi2_pars.t0_noderiv = False
                chi2_pars.t0 = True
                outputfile = open('outputs/buffer/' + inout_pars.label + '.dat', 'a')
                outputfile.write("lampos = 1e3")
                outputfile.write(inout_pars.inputnam)
                outputfile.write("\n")
                afo = levmar(afo)
        else:

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

    if pdf_closure.pdpdf:

        gridout()
        parsout()
        plotout()
        evgrido()

        t1 = time.process_time()
        print('time= ', t1 - tzero)

    else:

        gridout()
        parsout()
        plotout()
        evgrido()

        fit_pars.pos_const = False
        chi2t0f = chi2min(afo)
        chi2_pars.t0 = False
        chi2expf = chi2min(afo)
        fit_pars.pos_const = True
        chi2posf = chi2min(afo)
        pospenf = chi2posf - chi2expf

        print('chi2(exp) in:', chi2expi + pospeni, (chi2expi + pospeni) / chi2_pars.ndat)
        print('chi2(t0) in:', (chi2t0i + pospeni), (chi2t0i + pospeni) / chi2_pars.ndat)
        print('pos penalty in = ', pospeni, pospeni / chi2_pars.ndat)
        print('chi2(exp) out:', chi2expf + pospenf, (chi2expf + pospenf) / chi2_pars.ndat)
        print('chi2(exp) out (no pos pen):', chi2expf, (chi2expf) / chi2_pars.ndat)
        print('chi2(t0) out:', (chi2t0f + pospenf), (chi2t0f + pospenf) / chi2_pars.ndat)
        print('chi2(t0) out (no pos pen):', chi2t0f, (chi2t0f) / chi2_pars.ndat)
        print('pos penalty out = ', pospenf, pospenf / chi2_pars.ndat)

        resout(pospeni, pospenf, chi2t0i, chi2expi, chi2t0f, chi2expf, chi2_pars.ndat)
        t1 = time.process_time()
        print('time= ', t1 - tzero)

else:
    chi2_pars.t0 = False
    fit_pars.pos_const = False
    chi2expi = chi2min(afi)
    chi2_pars.t0 = True
    chi2t0i = chi2min(afi)
    ##    fit_pars.pos_const=True
    fit_pars.pos_const = True
    chi2posi = chi2min(afi)
    pospeni = chi2posi - chi2t0i
    print('chi2(exp) in:', chi2expi, chi2expi / chi2_pars.ndat)
    print('chi2(t0) in:', chi2t0i, chi2t0i / chi2_pars.ndat)
    print('pos penalty = ', pospeni)
    res = minimize(
        chi2min, afi, method='Nelder-Mead', tol=1e-4, options={'maxiter': 10000, 'maxfev': 10000}
    )
    # res = minimize(chi2min, afi, method='Newton-CG', jac=jac_fun, hess=hess_fun, options = {'disp': True})
    #    res = minimize(chi2min, afi, method='CG', jac=jac_fun, options = {'maxiter': 10000})
    print('pars out =', res.x)
    print(chi2min(res.x), chi2min(res.x) / chi2_pars.ndat)
    gridout()
    parsout()
    plotout()
    evgrido()
    fit_pars.pos_const = False
    chi2t0f = chi2min(res.x)
    chi2_pars.t0 = False
    chi2expf = chi2min(res.x)
    fit_pars.pos_const = True
    chi2posf = chi2min(res.x)
    pospenf = chi2posf - chi2expf
    print('chi2(exp) in:', chi2expi, chi2expi / chi2_pars.ndat)
    print('chi2(t0) in:', chi2t0i, chi2t0i / chi2_pars.ndat)
    print('pos penalty in = ', pospeni)
    print('chi2(exp)= ', chi2expf, chi2expf / chi2_pars.ndat)
    print('chi2(t0)= ', chi2t0f, chi2t0f / chi2_pars.ndat)
    print('pos penalty= ', pospenf)
    resout(pospeni, pospenf, chi2t0i, chi2expi, chi2t0f, chi2expf, chi2_pars.ndat)
    t1 = time.process_time()
    print('time= ', t1 - tzero)
