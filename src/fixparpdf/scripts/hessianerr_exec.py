#!/usr/bin/env python3

from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path
import subprocess as sp
import time

from reportengine.utils import yaml_safe

from fixparpdf import global_pars


def _existing_path(value):
    value_path = Path(value)
    if not value_path.exists():
        return ArgumentTypeError(f"Could not find {value_path}")
    return value_path

# TODO: lot s of repeated code from fixparpdf_exec. Move this function there or do smt else
def parse_config(config):
    # Read the necessary inputs from runcard
    _input_config = config.get("inout_parameters", {})
    _basis_config = config.get("basis pars", {})
    _pdf_pars = config.get("pdf pars", {})
    _chi2_pars = config.get("chi2_parameters", {})
    _fit_pars = config.get("fit_parameters", {})

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

    return _input_config, _basis_config, _pdf_pars, _chi2_pars, _fit_pars


def _hessian_error_calculation():
    """Compute the hessian error with or without dynamic tolerance"""
    # TODO: to be tested
    from fixparpdf.error_calc import hesserror_dynamic_tol_new, hesserror_new
    from fixparpdf.global_pars import chi2_pars, fit_pars
    from fixparpdf.inputs import readincov

    chi2_pars.t0 = True
    fit_pars.pos_const = fit_pars.nnpdf_pos
    (afi, hess, jac) = readincov()
    if chi2_pars.dynamic_tol:
        hesserror_dynamic_tol_new(afi, hess, jac)
    else:
        hesserror_new(afi, hess, jac)
    print("finished!")


def main():

    parser = ArgumentParser()

    parser.add_argument("runcard", type=_existing_path, help="Runcard defining the run")
    parser.add_argument("covfile", type=str, help="cov file")
    parser.add_argument("dynamic_tol", type=bool, help="dynamic tolerance")
    parser.add_argument("t2_err", type=float, help="tolerance")

    parser.add_argument(
        "--config", type=_existing_path, help="Configuration file to override default parameters"
    )
    args = parser.parse_args()

    config = yaml_safe.load(args.runcard.open("r"))

    _input_config, _basis_config, _pdf_pars, _chi2_pars, _fit_pars = parse_config(config)

    # Add evscan input
    _input_config['covinput'] = args.covfile
    _input_config['readcov'] = True  # useless?
    _chi2_pars['dynamic_tol'] = args.dynamic_tol
    _chi2_pars['t2_err'] = args.t2_err

    # Instantiate the global configuration
    global_pars.basis_pars = global_pars.BasisPars(**_basis_config)
    global_pars.inout_pars = inout_pars = global_pars.InoutPars(**_input_config)
    global_pars.pdf_pars = pdf_pars = global_pars.PDFPars(**_pdf_pars)
    global_pars.chi2_pars = chi2_pars = global_pars.Chi2Pars(**_chi2_pars)
    global_pars.fit_pars = fit_pars = global_pars.FitPars(**_fit_pars)
    # Take a reference to the ones with default values
    pdf_closure = global_pars.pdf_closure
    dload_pars = global_pars.dload_pars

    if global_pars.pdf_pars.lhin != global_pars.fit_pars.fixpar:
        raise ValueError("Both pdf_pars::lhin and fit_pars::fixpar must have the same value")

    # Populate the share data
    # TODO: this can take _full_dataset so that it doesn't go into _fit_pars
    global_pars.shared_populate_data()

    tzero = time.process_time()
    print(tzero)

    return _hessian_error_calculation()


if __name__ == "__main__":
    main()
