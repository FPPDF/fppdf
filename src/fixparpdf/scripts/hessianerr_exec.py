#!/usr/bin/env python3

from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path
import subprocess as sp
import time

from reportengine.utils import yaml_safe

from fixparpdf import global_pars  
from fixparpdf.utils import _existing_path, init_global_pars



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
   
    parser.add_argument(
        "--config", type=_existing_path, help="Configuration file to override default parameters"
    )
    args = parser.parse_args()

    config = yaml_safe.load(args.runcard.open("r"))
    # init global variables
    init_global_pars(config)

    tzero = time.process_time()
    print(tzero)

    return _hessian_error_calculation()


if __name__ == "__main__":
    main()
