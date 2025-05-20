import numpy as np
from validphys.convolution import central_predictions

from global_pars import fit_pars, pdf_pars, shared_global_data


def dat_calc_rep(dscomb, genrep=None):
    """Given a list of datanames, produce the replica pseudodata."""
    dnames = tuple([i["dataset"] for i in dscomb])
    if genrep is None:
        genrep = fit_pars.pseud
    return shared_global_data["data"].produce_replica(
        names=dnames, irep=fit_pars.irep, genrep=genrep
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


def _elu(x, alpha=1e-7):
    """Exponential Linear Unit"""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def pos_calc(pdata, vp_pdf) -> float:
    """Compute positivity penalty term
    Takes as input a list of dicts containing a positivity dataset {dataset} and a pdf
    """
    if vp_pdf is None:
        raise Exception("Wrong call of pos_calc")

    tot = 0.0
    lam = fit_pars.lampos

    for pos_dict in pdata:
        posds = shared_global_data["posdata"].select_dataset(pos_dict["dataset"])
        res = _elu(-lam * central_predictions(posds, vp_pdf).values)
        tot += np.sum(res)
    return tot
