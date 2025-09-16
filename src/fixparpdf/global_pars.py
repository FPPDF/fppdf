"""
These classes hold the global configuration for the code.

They are all frozen and instantiated as the runcards are read.
"""

# TODO 1: make the second sentence above to be true, at the moment they are instantiated at the bottom
# TODO 2: change the comments to doscstrings (probably can be automated as well)

import dataclasses
from functools import cache, cached_property, lru_cache
import os
from pathlib import Path

from nnpdf_data.validphys_compatibility import legacy_to_new_map
import numpy as np
import pandas as pd
from validphys.api import API
from validphys.convolution import central_predictions
from validphys.covmats import dataset_inputs_covmat_from_systematics
from validphys.lhaindex import get_lha_datapath
from validphys.pseudodata import make_replica


@dataclasses.dataclass
class Chi2Pars:
    # eps arr for newmin
    eps_arr_newmin = []
    # use t0 prescription for covariance matrix in chi^2 calculation
    t0 = False
    # if true then calculatess individual dataset chi^2 values
    chi2ind = False
    # flag for iterating through datasets when doing individually
    idat = 0.0
    # arrays with lower/upper indices of given dataset
    idat_up_arr = np.zeros((100), dtype=int)
    idat_low_arr = np.zeros((100), dtype=int)
    # if true then total chi^2 given bt individual dataset chi^2 values
    chitotind = False
    # number of data points - set in code
    ndat = 0
    # Only use t0 for chi2i and chi2o in levmar (i.e. no derivatives at all)
    t0_noderivin = True
    # global flag (initiator flag above)
    t0_noderiv = False
    # if true add diagonal elements of hessian from positivity (if used) - set to false for final output
    add_hessp = False
    # use dynamic tolerance
    dynamic_tol: bool = False
    # T^2 value for error calculation
    t2_err: float = 1.0
    # pos chi_0
    chi_pos0 = 0.0
    # pos chi_1
    chi_pos1 = 0.0
    # min chidel
    chidel_min = 0.0
    # chi_0^2 values
    chi0_ind_arr = []
    # chi_2 values
    chi_ind_arr = []
    # chi^2 limits
    chilim_arr = []
    # dataset name array
    cldataset_arr = []
    # dataset name array
    clnd_arr = []
    # flag to only calculate CLs once
    calc_cl = True
    # if true than LO closure (for dynamic tol)
    L0 = False


@dataclasses.dataclass
class BasisPars:
    # If true then two term gluon parameterisation used
    g_second_term: bool = True
    # if true and g_second_term=False then includes 7th Chebyshev for gluon
    g_cheb7: bool = False
    # If true then delta_S+ set to that of sea
    asp_fix: bool = True
    # If true the delta_d=delta_u for d_V and u_V (need to set delta_d fixed or will crash!)
    dvd_eq_uvd: bool = False
    # if true then fix A_s+ to force xT_8 -> 0 as x -> 0
    t8_int: bool = False
    # location of different PDF pars in parameter array
    i_uv_min: int = 0
    i_uv_max: int = 9
    i_dv_max: int = 18
    i_sea_max: int = 27
    i_sp_max: int = 36
    i_g_max: int = 46
    i_sm_max: int = 56
    i_dbub_max: int = 64
    i_ch_max: int = 73
    i_dv_min: int = i_uv_max
    i_sea_min: int = i_dv_max
    i_sp_min: int = i_sea_max
    i_g_min: int = i_sp_max
    i_sm_min: int = i_g_max
    i_dbub_min: int = i_sm_max
    i_ch_min: int = i_dbub_max
    # If true then have (up to) 8 rather than 6 Chebyshevs in input
    Cheb_8: bool = False
    # Number pars
    n_pars: int = 0

    # TODO: should this class be just fixed constants?
    # why can it change?

    def __post_init__(self):
        if self.Cheb_8:
            self.i_uv_min = 0
            self.i_uv_max += 2
            self.i_dv_max += 4
            self.i_sea_max += 6
            self.i_sp_max += 8
            self.i_g_max += 10
            self.i_sm_max += 10
            self.i_dbub_max += 12
            self.i_ch_max += 14
            self.i_dv_min = self.i_uv_max
            self.i_sea_min = self.i_dv_max
            self.i_sp_min = self.i_sea_max
            self.i_g_min = self.i_sp_max
            self.i_sm_min = self.i_g_max
            self.i_dbub_min = self.i_sm_max
            self.i_ch_min = self.i_dbub_max


@dataclasses.dataclass
class MinPars:
    # if true use sgd
    sgd = False
    # tolerance for lev mar
    tollm = 0.1


@dataclasses.dataclass
class PDFPars:
    #
    # use external LHAPDF grid as input
    lhin: bool = False
    # if true then scatters when doing direct PDF pd fit
    pdfscat: bool = False
    # labels PDF set for theory evaluation - used internally, value here arbitrary
    pdflabel: str = None
    # labels central PDF (with no parameter variations in fit for each iteration)
    PDFlabel_cent: str = 'init'
    #  if true then do direct fit to PDF pseudodata
    pdpdf: bool = False
    # counter to ensure new lhapdf grid used for every new theory evaluation
    idir: int = 0
    # When reading directly from LHAPDF (to be removed)
    PDFlabel_lhin: str = None
    # array containing input PDF parameters - set in code
    parsin = []
    # array containing PDF parameters as they get updated
    pdfparsi = []
    # array containing info on whether PDF parameters are free or not in fit
    par_isf = []
    # array containing PDF parameter indices that are free
    par_free_i = []
    # number of free parameters
    npar_free = 0
    # array containing free parameters and their + epsilon values for derivatives - internal
    parinarr = []
    # array containing free parameters for new minimisation - internal
    parinarr_newmin = []
    # internal counters for parinarr
    parin_newmin_counter = 0
    parin_newmin_reset = False
    # integer label of PDF set used for chi2 minimisation - internal
    iPDF = 0
    # array of delta_d (internal use)
    deld_arr = []
    # array of delta_u (internal use)
    delu_arr = []

    @cached_property
    def lhapdfdir(self) -> str:
        if os.environ.get("LHAPDF_DATA_PATH") is not None:
            return os.environ["LHAPDF_DATA_PATH"]
        return get_lha_datapath() + "/"


@dataclasses.dataclass
class PDFClosure:
    # if true then scatters when doing direct PDF pd fit
    pdfscat = False
    # label of pseudodata used
    pdlabel = 'NNPDF40pch_gl_l0'
    #  if true then do direct fit to PDF pseudodata
    pdpdf = False


@dataclasses.dataclass
class DloadPars:
    # flag to avoid reloading datasets after storing to memory
    dflag = 1
    # array of global dataset stored to memory
    darr_gl = []
    # array of global dataset theory values stored to memory
    tharr_gl = []
    # errors for pdf closure
    err_gl = []
    # true values for pdf closure
    true_gl = []
    # x values for pdf closure
    x_gl = []
    # dictionary of datasets in fit
    dscomb = []
    # dictionary of nlo datasets in fit
    dscomb_nlo = []
    # dictionary of nnlo datasets in fit
    dscomb_nnlo = []
    # impose NLO (p charm) cuts irrespective of theory id
    nlo_cuts = False
    # stored covariance matrix and inverse for no t0 min
    covexp = []
    covexp_inv = []
    # flag used when t0_noderiv=True (avoid recalculating cov matrix)
    dcov = 1
    # stored covariance matrix and inverse for no t0_noderiv=True
    covt0 = []
    covt0_inv = []
    # fk table storing array - when to reload
    fk_loadarr = np.ones((100), dtype=bool)
    # fk table storing array
    fk_arr = []
    # fk table index
    fk_ind = np.ones((100), dtype=int)
    ifk = 0
    # array for if hadronic
    t_had = np.zeros((100), dtype=bool)
    # xarr
    xarr_tot = []
    # tchi_newton
    tchi_newton = 0.0


@dataclasses.dataclass(frozen=True)
class InoutPars:
    """

    Parameters
    ----------
        inputnam: str
            Input file with the PDF parameters
        label: str
            Label for the output files
        covinput: str
            File from where to read the covariance matrix, defaults to <label>.dat
    """

    # Name of pdf parameter inputnam, value here arbitrary
    inputnam: str
    # labels ofr output files
    label: str = 'init'
    # name of covariance matrix inputnam, if used, value here arbitrary
    covinput: str = None

    # if true then write pseudodata out to file
    pdout: bool = False
    # if true then write pseudodata out to file
    pdin: bool = False
    # if true then append irep to label, apart from in evgrid (for pd fits)
    pd_output: bool = False
    replica: int = None

    def __post_init__(self):
        if self.pdin:
            self.pdout = False

        # If the covinput is not filled, automatically fill it from the label
        if self.covinput is None:
            self.covinput = f"{self.label}.dat"

    @property
    def pdf_output_lab(self):
        if self.replica is None:
            return self.label
        return f"{self.label}_irep{self.replica:04}"

    @property
    def theory_covmat_path(self):
        return Path("outputs") / "thcovmat" / f"{self.label}.csv"


@dataclasses.dataclass
class FitPars:
    """
    Fit Level parameters
    """

    # run with fixed input parameters (override input card flags)
    fixpar: bool = False
    # impose NNPDF positivity in fit
    nnpdf_pos: bool = False
    # if true then generates a replica from baseline dataset with seed irep
    # WIP Equivalent to genrep
    pseud: bool = False
    # irep number - also used when generating error grids
    irep: int = 0
    # if true impose weight to prefer delta_d > 0.25
    deld_const: bool = False
    # lhrep - rep number for lhin=True set
    lhrep: int = 0

    # Dataset
    dataset_40: list[dict] = dataclasses.field(default_factory=list)
    pos_data40: list[dict] = dataclasses.field(default_factory=list)
    added_filter_rules: list[dict] = dataclasses.field(default_factory=list)

    # min number of datasets
    imindat: int = 0
    # global positivity lambda
    lampos: bool = 1e3
    # If true then group datasets together for dynamic tolerance
    dynT_group: bool = False
    # If true then remove datasets with n < 5 from dynamic tolerance
    dynT_ngt5: bool = False
    # positivity constraint flag (set in code)
    pos_const: bool = False  # TODO: never set by input
    # deprecated
    nlo_cuts: bool = False

    @property
    def imaxdat(self):
        return len(self.dataset_40)


def _sanitize(name):
    """Sanitize old-version names"""
    newname, _ = legacy_to_new_map(name)
    return newname


@cache
def cached_central_predictions(ds, pdf):
    # TODO: this caching can (and should) be lifted to validphys
    return central_predictions(ds, pdf)


# TODO: this is a temporary function to remove states and caches from the functions above
@dataclasses.dataclass(frozen=True)
class DataHolder:
    """
    Class holding all data fixed information.
    At the moment it holds the data for the full possible set
    and the separate functions will query what they need.

    dataset:
        list of NNPDF dataset DataSetSpec objects
    theoryid:
        target theory ID
    theory_covmat:
        Whether the theory covmat should be considered or not
    """

    datasets: tuple
    theoryid: int = 40_000_000
    # NOTE: intersection cuts are being ignored, but should've been added upon construction
    use_theory_covmat: bool = False  # Whether to use the theory covmat or not

    @cached_property
    def _data_dict(self):
        return {str(ds): ds for ds in self.datasets}

    @cache
    def select_dataset(self, name):
        """Select dataset by name"""
        return self._data_dict[_sanitize(name)]

    @cache
    def select_datasets(self, names):
        """Select datasets given a list of names"""
        return [self.select_dataset(n) for n in names]

    @cache
    def _read_thcovmat(self):
        """Read the theory covariance matrix (if needed) and save it to memory."""

        thcovmat_path = inout_pars.theory_covmat_path
        if not thcovmat_path.exists():
            raise FileNotFoundError(
                f"Theory covmat not found at {thcovmat_path}. Please make sure to run `fixpar_setupfit` before runing the fit!"
            )

        filecovmat = pd.read_csv(
            thcovmat_path, index_col=[0, 1, 2], header=[0, 1, 2], sep="\t|,", engine="python"
        )
        # Remove string in column id
        filecovmat.columns = filecovmat.index
        return filecovmat

    @lru_cache
    def produce_covmat(
        self, pdf=None, imin=None, imax=None, names=None, datasets=None, use_t0=True
    ):
        """Produce the t0 covmat for the given PDF for the datasets.
        Either by using imin-imax or by using a tuple of names.
        """
        central_values = None
        if datasets is None:
            if names is None and (imin is not None and imax is not None):
                datasets = self.datasets[imin:imax]
            else:
                datasets = self.select_datasets(names=names)

        if use_t0:
            if pdf is None:
                raise ValueError("PDF missing for t0")
            if isinstance(pdf, str):
                pdf = API.pdf(pdf=pdf)
            central_values = [
                cached_central_predictions(ds, pdf).to_numpy().reshape(-1) for ds in datasets
            ]

        # TODO: if the datasets are not ordered by experiment in the runcard the might come incorrectly here!!
        cds = [ds.load_commondata() for ds in datasets]
        covmat = dataset_inputs_covmat_from_systematics(
            cds, _list_of_central_values=central_values, use_weights_in_covmat=False
        )

        thcovmat = 0.0
        if self.use_theory_covmat:
            df = self._read_thcovmat()
            dnames = [i.setname for i in cds]
            thcovmat = df.loc[pd.IndexSlice[:, dnames], pd.IndexSlice[:, dnames]].values
            # TODO:
            #   1. Check that the order of the data in thcovmat and covmat is truly the same
            #   2. Error out cleanly if _any_ of the datasets is not found in the thcovmat

        return covmat + thcovmat

    @cache
    def produce_replica(self, names=None, datasets=None, irep=0, genrep=False):
        """Produce the replica data given a tuple of datasets"""
        if datasets is None:
            datasets = self.select_datasets(names=names)

        # Load the data
        lcd = [i.load_commondata() for i in datasets]
        if genrep:
            covmat = self.produce_covmat(use_t0=False, names=names)
        else:
            covmat = None

        return make_replica(lcd, irep, covmat, genrep=genrep)

    @cache
    def get_theory(self):
        """Get the theory spec."""
        return API.theoryid(theoryid=self.theoryid)

    @cached_property
    def q20(self):
        """Get the fit initial q2."""
        tspec = self.get_theory()
        return tspec.get_description()["Q0"]


# Limite the shared global data to what's available in this dictionary
shared_global_data = {"data": None, "posdata": None}

def shared_populate_data(theoryid=40001000, use_theory_covmat=False):

    config = {"theoryid": theoryid, "use_cuts": "internal", "added_filter_rules":fit_pars.added_filter_rules}

    datasets = []
    for dinput in fit_pars.dataset_40:
        ds = API.dataset(dataset_input=dinput, **config)
        datasets.append(ds)

    positivity_datasets = API.posdatasets(
        posdatasets=fit_pars.pos_data40, **config
    )

    shared_global_data["data"] = DataHolder(
        tuple(datasets), theoryid=theoryid, use_theory_covmat=use_theory_covmat
    )
    shared_global_data["posdata"] = DataHolder(tuple(positivity_datasets), theoryid=theoryid)


# Instantiate all global configurations
# Change this perhaps to a namedtuple or something better
global_configuration = {"min": MinPars(), "closure": PDFClosure(), "dload": DloadPars()}
basis_pars = None
pdf_pars = None
inout_pars = None
fit_pars = None
chi2_pars = None
min_pars = global_configuration["min"]
pdf_closure = global_configuration["closure"]
dload_pars = global_configuration["dload"]
fit_pars = FitPars()
