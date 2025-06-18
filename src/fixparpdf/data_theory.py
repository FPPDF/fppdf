from functools import cache

import numpy as np

from fixparpdf.global_pars import fit_pars, pdf_pars, shared_global_data


@cache
def dat_calc_rep(datasets, genrep=None):
    """Given a list of datanames, produce the replica pseudodata."""
    if genrep is None:
        genrep = fit_pars.pseud
    return shared_global_data["data"].produce_replica(
        datasets=datasets, irep=fit_pars.irep, genrep=genrep
    )


def del_pen(delta):
    # TODO TBR
    sig = 0.05
    d0 = 0.3
    pen = np.exp(-(delta - d0) / sig)
    deriv = -pen / sig
    deriv2 = pen / sig / sig
    return (pen, deriv, deriv2)


def del_pen_calc():
    # TODO TBR
    chi0d = del_pen(pdf_pars.deld_arr[0])[0]
    chi0u = del_pen(pdf_pars.delu_arr[0])[0]
    idv = 0
    iuv = 0
    diffd = 0.0
    diffu = 0.0
    hessd = 0.0
    hessu = 0.0
    for ip in range(1, pdf_pars.npar_free + 1):
        deltad = pdf_pars.deld_arr[ip] - pdf_pars.deld_arr[0]
        deltau = pdf_pars.delu_arr[ip] - pdf_pars.delu_arr[0]
        if np.abs(deltad) > 1e-30:
            idv = ip
            diffd = del_pen(pdf_pars.deld_arr[0])[1]
            hessd = del_pen(pdf_pars.deld_arr[0])[2]
        if np.abs(deltau) > 1e-30:
            iuv = ip
            diffu = del_pen(pdf_pars.delu_arr[0])[1]
            hessu = del_pen(pdf_pars.delu_arr[0])[2]
    return (chi0d, chi0u, diffd, diffu, hessd, hessu, idv, iuv)


def compute_theory(datasets, vp_pdf, theta_idx=None) -> np.ndarray:
    """Compute theory predictions for all given datasets for the given PDF.

        If a theta_idx is given, the derivative with respect to the parameter will be taken.
        The result is a concatenation of all theory predictions
    """
    ret = []
    for dataset in datasets:
        ret.append(vp_pdf.central_predictions(dataset, derivative_theta_idx=theta_idx))
    return np.concatenate(ret)


def pos_calc(pdata, vp_pdf, theta_idx=None) -> np.ndarray:
    """Compute positivity penalty term
    Takes as input a list of dicts containing positivity_datasets and a pdf object,
    If theta_idx is not None, it returns the derivative wrt the given parameter.

    If theta_idx is None, the "good" (positive) points are set to 0

    Returns the contribution to be included in a loss function:
        lambda*PosPenalty
    """
    lam = fit_pars.lampos
    ret = compute_theory(pdata, vp_pdf, theta_idx=theta_idx)

    if theta_idx is None:
        ret = np.minimum(ret, 0.0)

    # Return a positivity contribution to be summed to the loss
    return -lam * ret       
