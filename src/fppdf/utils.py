"""
Common functions to fixparpdf and hessianerr scripts
"""

from argparse import ArgumentTypeError
from pathlib import Path

from fppdf import global_pars


def existing_path(value):
    value_path = Path(value)
    if not value_path.exists():
        return ArgumentTypeError(f"Could not find {value_path}")
    return value_path


def init_global_pars(config):
    """Initialize the global variables staring from the input runcard"""

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
    _fit_pars["dataset_40"] = _full_dataset
    _fit_pars["pos_data40"] = config.get("posdatasets", [])
    _fit_pars["added_filter_rules"] = config.get("added_filter_rules", [])

    # Fill default labels:
    label = _input_config.setdefault("label", "init")
    _input_config.setdefault("covinput", f"{label}.dat")
    if "readcov" in _input_config:
        if _input_config.pop("readcov"):
            print("inout_parameters::readcov is no longer needed")

    # Instantiate the global configuration
    global_pars.basis_pars = global_pars.BasisPars(**_basis_config)
    global_pars.inout_pars = global_pars.InoutPars(**_input_config)
    global_pars.pdf_pars = global_pars.PDFPars(**_pdf_pars)
    global_pars.chi2_pars = global_pars.Chi2Pars(**_chi2_pars)
    global_pars.fit_pars = global_pars.FitPars(**_fit_pars)

    # if global_pars.pdf_pars.lhin != global_pars.fit_pars.fixpar:
    #     raise ValueError("Both pdf_pars::lhin and fit_pars::fixpar must have the same value")

    # Populate the share data
    # TODO: this can take _full_dataset so that it doesn't go into _fit_pars
    thid = config["theoryid"]
    use_thcovmat = "theorycovmatconfig" in config

    global_pars.shared_populate_data(theoryid=thid, use_theory_covmat=use_thcovmat)
