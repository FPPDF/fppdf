#!/usr/bin/env python3

from argparse import ArgumentParser
import time

from reportengine.utils import yaml_safe

from fixparpdf.utils import existing_path, init_global_pars


def main():

    parser = ArgumentParser()
    parser.add_argument("runcard", type=existing_path, help="Runcard defining the run")
    parser.add_argument(
        "--config", type=existing_path, help="Configuration file to override default parameters"
    )
    args = parser.parse_args()

    config = yaml_safe.load(args.runcard.open("r"))
    # init global variables
    init_global_pars(config)

    from fixparpdf.error_calc import hesserror_dynamic_tol_new, hesserror_new
    from fixparpdf.global_pars import chi2_pars, fit_pars
    from fixparpdf.inputs import readincov

    # Override a bunch of parameters
    chi2_pars.t0 = True
    fit_pars.pos_const = fit_pars.nnpdf_pos

    afi, hess, jac = readincov()

    tzero = time.time()
    if chi2_pars.dynamic_tol:
        print("Dynamic tolerance active")
        hesserror_dynamic_tol_new(afi, hess, jac)
    else:
        hesserror_new(afi, hess)
    tfinal = time.time()
    print(f"Finished in {tfinal-tzero:.3}s")


if __name__ == "__main__":
    main()
