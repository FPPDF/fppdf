"""
These classes hold the global configuration for the code.

They are all frozen and instantiated as the runcards are read.
"""

# TODO 1: make the second sentence above to be true, at the moment they are instantiated at the bottom
# TODO 2: change the comments to doscstrings (probably can be automated as well)

import dataclasses
import os
from functools import cache, cached_property

from nnpdf_data.validphys_compatibility import legacy_to_new_map
import numpy as np
from validphys.api import API
from validphys.convolution import central_predictions
from validphys.covmats import dataset_inputs_covmat_from_systematics
from validphys.pseudodata import make_replica
from validphys.lhaindex import get_lha_datapath


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
    t0_noderivin = False
    # global flag (initiator flag above)
    t0_noderiv = False
    # if true add diagonal elements of hessian from positivity (if used) - set to false for final output
    add_hessp = False
    # use dynamic tolerance
    dynamic_tol = False
    # T^2 value for error calculation
    t2_err = 1.0
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
    g_second_term:bool = False
    # if true and g_second_term=False then includes 7th Chebyshev for gluon
    g_cheb7:bool = False
    # If true then delta_S+ set to that of sea
    asp_fix:bool = True
    # If true the delta_d=delta_u for d_V and u_V (need to set delta_d fixed or will crash!)
    dvd_eq_uvd:bool = False
    # if true then fix A_s+ to force xT_8 -> 0 as x -> 0
    t8_int:bool = False
    # location of different PDF pars in parameter array
    i_uv_min:int = 0
    i_uv_max:int = 9
    i_dv_max:int = 18
    i_sea_max:int = 27
    i_sp_max:int = 36
    i_g_max:int = 46
    i_sm_max:int = 56
    i_dbub_max:int = 64
    i_ch_max:int = 73
    i_dv_min:int = i_uv_max
    i_sea_min:int = i_dv_max
    i_sp_min:int = i_sea_max
    i_g_min:int = i_sp_max
    i_sm_min:int = i_g_max
    i_dbub_min:int = i_sm_max
    i_ch_min:int = i_dbub_max
    # If true then have (up to) 8 rather than 6 Chebyshevs in input
    Cheb_8:bool = False
    # Number pars
    n_pars:int = 0


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
    lhin:bool = False
    uselha: bool = False
    pdfscat: bool = False
    # labels PDF set for theory evaluation - used internally, value here arbitrary
    pdflabel:str = 'init'
    # labels central PDF (with no parameter variations in fit for each iteration)
    PDFlabel_cent:str = 'init'
    #  if true then do direct fit to PDF pseudodata
    pdpdf:bool = False
    # counter to ensure new lhapdf grid used for every new theory evaluation
    idir:int = 0
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
            return  os.environ["LHAPDF_DATA_PATH"]
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


@dataclasses.dataclass
class InoutPars:
    # labels output files, value here abitrary
    label: str = 'init'
    # name of pdf parameter inputnam, value here arbitrary
    inputnam:str = 'init'
    # if true then read in covariance matrix and evaluate PDF errors
    readcov:bool = False
    # name of covariance matrix inputnam, if used, value here arbitrary
    covinput:str = 'init'
    # if true then write pseudodata out to file
    pdout:bool = False
    # if true then write pseudodata out to file
    pdin:bool = False
    # if true then append irep to label, apart from in evgrid (for pd fits)
    pd_output:bool = False
    replica: int = None

    @property
    def pdf_output_lab(self):
        if self.replica is None:
            return self.label
        return f"{self.label}_irep{self.replica:04}"


class fit_pars:
    # run with fixed input parameters (override input card flags)
    fixpar = True
    # NNPDF theory id (211 = NNLO pch)
    theoryidi = 211
    # impose NNPDF positivity in fit
    nnpdf_pos = False
    # positivity constraint flag (set in code)
    pos_const = False
    # if true then impose 4.0 positivity constraint, if pos_const=True
    pos_40 = True
    # if true impose weight to prefer delta_d > 0.25
    deld_const = False
    # if true then generates a replica from baseline dataset with seed irep
    pseud = False
    # irep number - also used when generating error grids
    irep = 0
    # lhrep - rep number for lhin=True set
    lhrep = 0
    # NMC PD data - set covariance matrix to be diagonal
    nmcpd_diag = False
    # dataset
    dataset_40 = [
        {'dataset': 'NMCPD_dw_ite'},
        {'dataset': 'NMC'},
        {'dataset': 'SLACP_dwsh'},
        {'dataset': 'SLACD_dw_ite'},
        {'dataset': 'BCDMSP_dwsh'},
        {'dataset': 'BCDMSD_dw_ite'},
        {'dataset': 'CHORUSNUPb_dw_ite'},
        {'dataset': 'CHORUSNBPb_dw_ite'},
        {'dataset': 'NTVNUDMNFe_dw_ite', 'cfac': ['MAS']},
        {'dataset': 'NTVNBDMNFe_dw_ite', 'cfac': ['MAS']},
        {'dataset': 'HERACOMBNCEM'},
        {'dataset': 'HERACOMBNCEP460'},
        {'dataset': 'HERACOMBNCEP575'},
        {'dataset': 'HERACOMBNCEP820'},
        {'dataset': 'HERACOMBNCEP920'},
        {'dataset': 'HERACOMBCCEM'},
        {'dataset': 'HERACOMBCCEP'},
        {'dataset': 'HERACOMB_SIGMARED_C'},
        {'dataset': 'HERACOMB_SIGMARED_B'},
        {'dataset': 'DYE886R_dw_ite', 'cfac': ['QCD']},
        {'dataset': 'DYE886P', 'cfac': ['QCD']},
        {'dataset': 'DYE605_dw_ite', 'cfac': ['QCD']},
        {'dataset': 'DYE906R_dw_ite', 'cfac': ['ACC', 'QCD']},
        {'dataset': 'CDFZRAP_NEW', 'cfac': ['QCD']},
        {'dataset': 'D0ZRAP_40', 'cfac': ['QCD']},
        {'dataset': 'D0WMASY', 'cfac': ['QCD']},
        {'dataset': 'ATLASWZRAP36PB', 'cfac': ['QCD']},
        {'dataset': 'ATLASZHIGHMASS49FB', 'cfac': ['QCD']},
        {'dataset': 'ATLASLOMASSDY11EXT', 'cfac': ['QCD']},
        {'dataset': 'ATLASWZRAP11CC', 'cfac': ['QCD']},
        {'dataset': 'ATLASWZRAP11CF', 'cfac': ['QCD']},
        {'dataset': 'ATLASDY2D8TEV', 'cfac': ['QCDEWK']},
        {'dataset': 'ATLAS_DY_2D_8TEV_LOWMASS', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_WZ_TOT_13TEV', 'cfac': ['NRM', 'QCD']},
        {'dataset': 'ATLAS_WP_JET_8TEV_PT', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_WM_JET_8TEV_PT', 'cfac': ['QCD']},
        {'dataset': 'ATLASZPT8TEVMDIST', 'cfac': ['QCD'], 'sys': 10},
        {'dataset': 'ATLASZPT8TEVYDIST', 'cfac': ['QCD'], 'sys': 10},
        {'dataset': 'ATLASTTBARTOT7TEV', 'cfac': ['QCD']},
        {'dataset': 'ATLASTTBARTOT8TEV', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_TTBARTOT_13TEV_FULLLUMI', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_TTB_DIFF_8TEV_LJ_TRAPNORM', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_TTB_DIFF_8TEV_LJ_TTRAPNORM', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_TOPDIFF_DILEPT_8TEV_TTRAPNORM', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_1JET_8TEV_R06_DEC', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_2JET_7TEV_R06', 'cfac': ['QCD']},
        {'dataset': 'ATLASPHT15_SF', 'cfac': ['QCD', 'EWK']},
        {'dataset': 'ATLAS_SINGLETOP_TCH_R_7TEV', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_SINGLETOP_TCH_R_13TEV', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_SINGLETOP_TCH_DIFF_7TEV_T_RAP_NORM', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_SINGLETOP_TCH_DIFF_7TEV_TBAR_RAP_NORM', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_SINGLETOP_TCH_DIFF_8TEV_T_RAP_NORM', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_SINGLETOP_TCH_DIFF_8TEV_TBAR_RAP_NORM', 'cfac': ['QCD']},
        {'dataset': 'CMSWEASY840PB', 'cfac': ['QCD']},
        {'dataset': 'CMSWMASY47FB', 'cfac': ['QCD']},
        {'dataset': 'CMSDY2D11', 'cfac': ['QCD']},
        {'dataset': 'CMSWMU8TEV', 'cfac': ['QCD']},
        {'dataset': 'CMSZDIFF12', 'cfac': ['QCD', 'NRM'], 'sys': 10},
        {'dataset': 'CMS_2JET_7TEV', 'cfac': ['QCD']},
        {'dataset': 'CMS_1JET_8TEV', 'cfac': ['QCD']},
        {'dataset': 'CMSTTBARTOT7TEV', 'cfac': ['QCD']},
        {'dataset': 'CMSTTBARTOT8TEV', 'cfac': ['QCD']},
        {'dataset': 'CMSTTBARTOT13TEV', 'cfac': ['QCD']},
        {'dataset': 'CMSTOPDIFF8TEVTTRAPNORM', 'cfac': ['QCD']},
        {'dataset': 'CMSTTBARTOT5TEV', 'cfac': ['QCD']},
        {'dataset': 'CMS_TTBAR_2D_DIFF_MTT_TRAP_NORM', 'cfac': ['QCD']},
        {'dataset': 'CMS_TTB_DIFF_13TEV_2016_2L_TRAP', 'cfac': ['QCD']},
        {'dataset': 'CMS_TTB_DIFF_13TEV_2016_LJ_TRAP', 'cfac': ['QCD']},
        {'dataset': 'CMS_SINGLETOP_TCH_TOT_7TEV', 'cfac': ['QCD']},
        {'dataset': 'CMS_SINGLETOP_TCH_R_8TEV', 'cfac': ['QCD']},
        {'dataset': 'CMS_SINGLETOP_TCH_R_13TEV', 'cfac': ['QCD']},
        {'dataset': 'LHCBZ940PB', 'cfac': ['QCD']},
        {'dataset': 'LHCBZEE2FB_40', 'cfac': ['QCD']},
        {'dataset': 'LHCBWZMU7TEV', 'cfac': ['NRM', 'QCD']},
        {'dataset': 'LHCBWZMU8TEV', 'cfac': ['NRM', 'QCD']},
        {'dataset': 'LHCB_Z_13TEV_DIMUON', 'cfac': ['QCD']},
        {'dataset': 'LHCB_Z_13TEV_DIELECTRON', 'cfac': ['QCD']},
    ]
    #  4.0 positivity dataset
    pos_data40 = [
        {'dataset': 'POSXUQ', 'maxlambda': 1e6},
        {'dataset': 'POSXUB', 'maxlambda': 1e6},
        {'dataset': 'POSXDQ', 'maxlambda': 1e6},
        {'dataset': 'POSXDB', 'maxlambda': 1e6},
        {'dataset': 'POSXSQ', 'maxlambda': 1e6},
        {'dataset': 'POSXSB', 'maxlambda': 1e6},
        {'dataset': 'POSXGL', 'maxlambda': 1e6},
    ]
    #  4.0 positivity dataset gluon only
    pos_data40_gluononly = [{'dataset': 'POSXGL', 'maxlambda': 1e6}]
    #  4.0 positivity dataset without gluon
    pos_data40_nogluon = [
        {'dataset': 'POSXUQ', 'maxlambda': 1e6},
        {'dataset': 'POSXUB', 'maxlambda': 1e6},
        {'dataset': 'POSXDQ', 'maxlambda': 1e6},
        {'dataset': 'POSXDB', 'maxlambda': 1e6},
        {'dataset': 'POSXSQ', 'maxlambda': 1e6},
        {'dataset': 'POSXSB', 'maxlambda': 1e6},
    ]
    # pos_data40=[{'dataset': 'POSXGL', 'maxlambda': 1e6}]
    #  3.1 positivity dataset
    pos_data31 = [
        {'dataset': 'POSF2U', 'maxlambda': 1e6},
        {'dataset': 'POSF2DW', 'maxlambda': 1e6},
        {'dataset': 'POSF2S', 'maxlambda': 1e6},
        {'dataset': 'POSFLL', 'maxlambda': 1e6},
        {'dataset': 'POSDYU', 'maxlambda': 1e10},
        {'dataset': 'POSDYD', 'maxlambda': 1e10},
        {'dataset': 'POSDYS', 'maxlambda': 1e10},
        {'dataset': 'POSF2C', 'maxlambda': 1e6},
    ]
    #  3.1 positivity dataset without gluon
    pos_data31_nogluon = [
        {'dataset': 'POSF2U', 'maxlambda': 1e6},
        {'dataset': 'POSF2DW', 'maxlambda': 1e6},
        {'dataset': 'POSF2S', 'maxlambda': 1e6},
        {'dataset': 'POSDYU', 'maxlambda': 1e10},
        {'dataset': 'POSDYD', 'maxlambda': 1e10},
        {'dataset': 'POSDYS', 'maxlambda': 1e10},
        {'dataset': 'POSF2C', 'maxlambda': 1e6},
    ]
    #  3.1 positivity dataset gluon only
    pos_data31_gluononly = [{'dataset': 'POSFLL', 'maxlambda': 1e6}]
    #  3.1 positivity dataset DY_c only
    pos_data31_dyc = [{'dataset': 'POSF2C', 'maxlambda': 1e6}]
    # If true then remove gluon positivity
    pos_nogluon = False
    # If true then only F2C positivity
    pos_dyc = False
    # If true then only gluon positivity
    pos_gluononly = False

    systrue = np.zeros((77), dtype=bool)
    systrue[36] = True
    systrue[37] = True
    systrue[57] = True

    cftrue = np.ones((77), dtype=bool)
    for i in range(0, 8):
        cftrue[i] = False
    for i in range(10, 19):
        cftrue[i] = False

    # total number of datasets
    imaxdat = len(dataset_40)
    # min number of datasets
    imindat = 0
    # positivity lambda
    lampos = 1e3

    # dataset flag - global, HERAonly, noHERA, noLHC, hhcollideronly, lowenergyDIS, lowenergyDY
    dset_type = 'global'

    dataset_noHERA = [
        {'dataset': 'NMCPD_dw_ite'},
        {'dataset': 'NMC'},
        {'dataset': 'SLACP_dwsh'},
        {'dataset': 'SLACD_dw_ite'},
        {'dataset': 'BCDMSP_dwsh'},
        {'dataset': 'BCDMSD_dw_ite'},
        {'dataset': 'CHORUSNUPb_dw_ite'},
        {'dataset': 'CHORUSNBPb_dw_ite'},
        {'dataset': 'NTVNUDMNFe_dw_ite', 'cfac': ['MAS']},
        {'dataset': 'NTVNBDMNFe_dw_ite', 'cfac': ['MAS']},
        {'dataset': 'DYE886R_dw_ite', 'cfac': ['QCD']},
        {'dataset': 'DYE886P', 'cfac': ['QCD']},
        {'dataset': 'DYE605_dw_ite', 'cfac': ['QCD']},
        {'dataset': 'DYE906R_dw_ite', 'cfac': ['ACC', 'QCD']},
        {'dataset': 'CDFZRAP_NEW', 'cfac': ['QCD']},
        {'dataset': 'D0ZRAP_40', 'cfac': ['QCD']},
        {'dataset': 'D0WMASY', 'cfac': ['QCD']},
        {'dataset': 'ATLASWZRAP36PB', 'cfac': ['QCD']},
        {'dataset': 'ATLASZHIGHMASS49FB', 'cfac': ['QCD']},
        {'dataset': 'ATLASLOMASSDY11EXT', 'cfac': ['QCD']},
        {'dataset': 'ATLASWZRAP11CC', 'cfac': ['QCD']},
        {'dataset': 'ATLASWZRAP11CF', 'cfac': ['QCD']},
        {'dataset': 'ATLASDY2D8TEV', 'cfac': ['QCDEWK']},
        {'dataset': 'ATLAS_DY_2D_8TEV_LOWMASS', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_WZ_TOT_13TEV', 'cfac': ['NRM', 'QCD']},
        {'dataset': 'ATLAS_WP_JET_8TEV_PT', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_WM_JET_8TEV_PT', 'cfac': ['QCD']},
        {'dataset': 'ATLASZPT8TEVMDIST', 'cfac': ['QCD'], 'sys': 10},
        {'dataset': 'ATLASZPT8TEVYDIST', 'cfac': ['QCD'], 'sys': 10},
        {'dataset': 'ATLASTTBARTOT7TEV', 'cfac': ['QCD']},
        {'dataset': 'ATLASTTBARTOT8TEV', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_TTBARTOT_13TEV_FULLLUMI', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_TTB_DIFF_8TEV_LJ_TRAPNORM', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_TTB_DIFF_8TEV_LJ_TTRAPNORM', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_TOPDIFF_DILEPT_8TEV_TTRAPNORM', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_1JET_8TEV_R06_DEC', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_2JET_7TEV_R06', 'cfac': ['QCD']},
        {'dataset': 'ATLASPHT15_SF', 'cfac': ['QCD', 'EWK']},
        {'dataset': 'ATLAS_SINGLETOP_TCH_R_7TEV', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_SINGLETOP_TCH_R_13TEV', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_SINGLETOP_TCH_DIFF_7TEV_T_RAP_NORM', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_SINGLETOP_TCH_DIFF_7TEV_TBAR_RAP_NORM', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_SINGLETOP_TCH_DIFF_8TEV_T_RAP_NORM', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_SINGLETOP_TCH_DIFF_8TEV_TBAR_RAP_NORM', 'cfac': ['QCD']},
        {'dataset': 'CMSWEASY840PB', 'cfac': ['QCD']},
        {'dataset': 'CMSWMASY47FB', 'cfac': ['QCD']},
        {'dataset': 'CMSDY2D11', 'cfac': ['QCD']},
        {'dataset': 'CMSWMU8TEV', 'cfac': ['QCD']},
        {'dataset': 'CMSZDIFF12', 'cfac': ['QCD', 'NRM'], 'sys': 10},
        {'dataset': 'CMS_2JET_7TEV', 'cfac': ['QCD']},
        {'dataset': 'CMS_1JET_8TEV', 'cfac': ['QCD']},
        {'dataset': 'CMSTTBARTOT7TEV', 'cfac': ['QCD']},
        {'dataset': 'CMSTTBARTOT8TEV', 'cfac': ['QCD']},
        {'dataset': 'CMSTTBARTOT13TEV', 'cfac': ['QCD']},
        {'dataset': 'CMSTOPDIFF8TEVTTRAPNORM', 'cfac': ['QCD']},
        {'dataset': 'CMSTTBARTOT5TEV', 'cfac': ['QCD']},
        {'dataset': 'CMS_TTBAR_2D_DIFF_MTT_TRAP_NORM', 'cfac': ['QCD']},
        {'dataset': 'CMS_TTB_DIFF_13TEV_2016_2L_TRAP', 'cfac': ['QCD']},
        {'dataset': 'CMS_TTB_DIFF_13TEV_2016_LJ_TRAP', 'cfac': ['QCD']},
        {'dataset': 'CMS_SINGLETOP_TCH_TOT_7TEV', 'cfac': ['QCD']},
        {'dataset': 'CMS_SINGLETOP_TCH_R_8TEV', 'cfac': ['QCD']},
        {'dataset': 'CMS_SINGLETOP_TCH_R_13TEV', 'cfac': ['QCD']},
        {'dataset': 'LHCBZ940PB', 'cfac': ['QCD']},
        {'dataset': 'LHCBZEE2FB_40', 'cfac': ['QCD']},
        {'dataset': 'LHCBWZMU7TEV', 'cfac': ['NRM', 'QCD']},
        {'dataset': 'LHCBWZMU8TEV', 'cfac': ['NRM', 'QCD']},
        {'dataset': 'LHCB_Z_13TEV_DIMUON', 'cfac': ['QCD']},
        {'dataset': 'LHCB_Z_13TEV_DIELECTRON', 'cfac': ['QCD']},
    ]

    dataset_HERAonly = [
        {'dataset': 'HERACOMBNCEM'},
        #     {'dataset': 'HERACOMBNCEP460'},
        #     {'dataset': 'HERACOMBNCEP575'},
        #     {'dataset': 'HERACOMBNCEP820'},
        #     {'dataset': 'HERACOMBNCEP920'},
        #     {'dataset': 'HERACOMBCCEM'},
        #     {'dataset': 'HERACOMBCCEP'},
        #     {'dataset': 'HERACOMB_SIGMARED_C'},
        {'dataset': 'HERACOMB_SIGMARED_B'},
    ]

    dataset_noLHC = [
        {'dataset': 'NMCPD_dw_ite'},
        {'dataset': 'NMC'},
        {'dataset': 'SLACP_dwsh'},
        {'dataset': 'SLACD_dw_ite'},
        {'dataset': 'BCDMSP_dwsh'},
        {'dataset': 'BCDMSD_dw_ite'},
        {'dataset': 'CHORUSNUPb_dw_ite'},
        {'dataset': 'CHORUSNBPb_dw_ite'},
        {'dataset': 'NTVNUDMNFe_dw_ite', 'cfac': ['MAS']},
        {'dataset': 'NTVNBDMNFe_dw_ite', 'cfac': ['MAS']},
        {'dataset': 'HERACOMBNCEM'},
        {'dataset': 'HERACOMBNCEP460'},
        {'dataset': 'HERACOMBNCEP575'},
        {'dataset': 'HERACOMBNCEP820'},
        {'dataset': 'HERACOMBNCEP920'},
        {'dataset': 'HERACOMBCCEM'},
        {'dataset': 'HERACOMBCCEP'},
        {'dataset': 'HERACOMB_SIGMARED_C'},
        {'dataset': 'HERACOMB_SIGMARED_B'},
        {'dataset': 'DYE886R_dw_ite', 'cfac': ['QCD']},
        {'dataset': 'DYE886P', 'cfac': ['QCD']},
        {'dataset': 'DYE605_dw_ite', 'cfac': ['QCD']},
        {'dataset': 'DYE906R_dw_ite', 'cfac': ['ACC', 'QCD']},
        {'dataset': 'CDFZRAP_NEW', 'cfac': ['QCD']},
        {'dataset': 'D0ZRAP_40', 'cfac': ['QCD']},
        {'dataset': 'D0WMASY', 'cfac': ['QCD']},
    ]

    dataset_hhcollideronly = [
        {'dataset': 'CDFZRAP_NEW', 'cfac': ['QCD']},
        {'dataset': 'D0ZRAP_40', 'cfac': ['QCD']},
        {'dataset': 'D0WMASY', 'cfac': ['QCD']},
        {'dataset': 'ATLASWZRAP36PB', 'cfac': ['QCD']},
        {'dataset': 'ATLASZHIGHMASS49FB', 'cfac': ['QCD']},
        {'dataset': 'ATLASLOMASSDY11EXT', 'cfac': ['QCD']},
        {'dataset': 'ATLASWZRAP11CC', 'cfac': ['QCD']},
        {'dataset': 'ATLASWZRAP11CF', 'cfac': ['QCD']},
        {'dataset': 'ATLASDY2D8TEV', 'cfac': ['QCDEWK']},
        {'dataset': 'ATLAS_DY_2D_8TEV_LOWMASS', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_WZ_TOT_13TEV', 'cfac': ['NRM', 'QCD']},
        {'dataset': 'ATLAS_WP_JET_8TEV_PT', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_WM_JET_8TEV_PT', 'cfac': ['QCD']},
        {'dataset': 'ATLASZPT8TEVMDIST', 'cfac': ['QCD'], 'sys': 10},
        {'dataset': 'ATLASZPT8TEVYDIST', 'cfac': ['QCD'], 'sys': 10},
        {'dataset': 'ATLASTTBARTOT7TEV', 'cfac': ['QCD']},
        {'dataset': 'ATLASTTBARTOT8TEV', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_TTBARTOT_13TEV_FULLLUMI', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_TTB_DIFF_8TEV_LJ_TRAPNORM', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_TTB_DIFF_8TEV_LJ_TTRAPNORM', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_TOPDIFF_DILEPT_8TEV_TTRAPNORM', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_1JET_8TEV_R06_DEC', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_2JET_7TEV_R06', 'cfac': ['QCD']},
        {'dataset': 'ATLASPHT15_SF', 'cfac': ['QCD', 'EWK']},
        {'dataset': 'ATLAS_SINGLETOP_TCH_R_7TEV', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_SINGLETOP_TCH_R_13TEV', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_SINGLETOP_TCH_DIFF_7TEV_T_RAP_NORM', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_SINGLETOP_TCH_DIFF_7TEV_TBAR_RAP_NORM', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_SINGLETOP_TCH_DIFF_8TEV_T_RAP_NORM', 'cfac': ['QCD']},
        {'dataset': 'ATLAS_SINGLETOP_TCH_DIFF_8TEV_TBAR_RAP_NORM', 'cfac': ['QCD']},
        {'dataset': 'CMSWEASY840PB', 'cfac': ['QCD']},
        {'dataset': 'CMSWMASY47FB', 'cfac': ['QCD']},
        {'dataset': 'CMSDY2D11', 'cfac': ['QCD']},
        {'dataset': 'CMSWMU8TEV', 'cfac': ['QCD']},
        {'dataset': 'CMSZDIFF12', 'cfac': ['QCD', 'NRM'], 'sys': 10},
        {'dataset': 'CMS_2JET_7TEV', 'cfac': ['QCD']},
        {'dataset': 'CMS_1JET_8TEV', 'cfac': ['QCD']},
        {'dataset': 'CMSTTBARTOT7TEV', 'cfac': ['QCD']},
        {'dataset': 'CMSTTBARTOT8TEV', 'cfac': ['QCD']},
        {'dataset': 'CMSTTBARTOT13TEV', 'cfac': ['QCD']},
        {'dataset': 'CMSTOPDIFF8TEVTTRAPNORM', 'cfac': ['QCD']},
        {'dataset': 'CMSTTBARTOT5TEV', 'cfac': ['QCD']},
        {'dataset': 'CMS_TTBAR_2D_DIFF_MTT_TRAP_NORM', 'cfac': ['QCD']},
        {'dataset': 'CMS_TTB_DIFF_13TEV_2016_2L_TRAP', 'cfac': ['QCD']},
        {'dataset': 'CMS_TTB_DIFF_13TEV_2016_LJ_TRAP', 'cfac': ['QCD']},
        {'dataset': 'CMS_SINGLETOP_TCH_TOT_7TEV', 'cfac': ['QCD']},
        {'dataset': 'CMS_SINGLETOP_TCH_R_8TEV', 'cfac': ['QCD']},
        {'dataset': 'CMS_SINGLETOP_TCH_R_13TEV', 'cfac': ['QCD']},
        {'dataset': 'LHCBZ940PB', 'cfac': ['QCD']},
        {'dataset': 'LHCBZEE2FB_40', 'cfac': ['QCD']},
        {'dataset': 'LHCBWZMU7TEV', 'cfac': ['NRM', 'QCD']},
        {'dataset': 'LHCBWZMU8TEV', 'cfac': ['NRM', 'QCD']},
        {'dataset': 'LHCB_Z_13TEV_DIMUON', 'cfac': ['QCD']},
        {'dataset': 'LHCB_Z_13TEV_DIELECTRON', 'cfac': ['QCD']},
    ]

    # dataset_hhcollideronly=[{'dataset': 'D0WMASY', 'cfac': ['QCD']}]
    # dataset_hhcollideronly=[{'dataset': 'ATLASWZRAP36PB', 'cfac': ['QCD']}]

    dataset_lowenergyDIS = [
        {'dataset': 'NMCPD_dw_ite'},
        {'dataset': 'NMC'},
        {'dataset': 'SLACP_dwsh'},
        {'dataset': 'SLACD_dw_ite'},
        {'dataset': 'BCDMSP_dwsh'},
        {'dataset': 'BCDMSD_dw_ite'},
        {'dataset': 'CHORUSNUPb_dw_ite'},
        {'dataset': 'CHORUSNBPb_dw_ite'},
        {'dataset': 'NTVNUDMNFe_dw_ite', 'cfac': ['MAS']},
        {'dataset': 'NTVNBDMNFe_dw_ite', 'cfac': ['MAS']},
    ]

    dataset_lowenergyDY = [
        {'dataset': 'DYE886R_dw_ite', 'cfac': ['QCD']},
        {'dataset': 'DYE886P', 'cfac': ['QCD']},
        {'dataset': 'DYE605_dw_ite', 'cfac': ['QCD']},
        {'dataset': 'DYE906R_dw_ite', 'cfac': ['ACC', 'QCD']},
    ]

    dataset_lowenergyDISDY = [
        {'dataset': 'NMCPD_dw_ite'},
        {'dataset': 'NMC'},
        {'dataset': 'SLACP_dwsh'},
        {'dataset': 'SLACD_dw_ite'},
        {'dataset': 'BCDMSP_dwsh'},
        {'dataset': 'BCDMSD_dw_ite'},
        {'dataset': 'CHORUSNUPb_dw_ite'},
        {'dataset': 'CHORUSNBPb_dw_ite'},
        {'dataset': 'NTVNUDMNFe_dw_ite', 'cfac': ['MAS']},
        {'dataset': 'NTVNBDMNFe_dw_ite', 'cfac': ['MAS']},
        {'dataset': 'DYE886R_dw_ite', 'cfac': ['QCD']},
        {'dataset': 'DYE886P', 'cfac': ['QCD']},
        {'dataset': 'DYE605_dw_ite', 'cfac': ['QCD']},
        {'dataset': 'DYE906R_dw_ite', 'cfac': ['ACC', 'QCD']},
    ]

    # dataset_lowenergyDISDY=[{'dataset': 'HERACOMBNCEP460'}]

    # dataset
    dataset_lowenergyDISDY_HERAonly = [
        {'dataset': 'NMCPD_dw_ite'},
        {'dataset': 'NMC'},
        {'dataset': 'SLACP_dwsh'},
        {'dataset': 'SLACD_dw_ite'},
        {'dataset': 'BCDMSP_dwsh'},
        {'dataset': 'BCDMSD_dw_ite'},
        {'dataset': 'CHORUSNUPb_dw_ite'},
        {'dataset': 'CHORUSNBPb_dw_ite'},
        {'dataset': 'NTVNUDMNFe_dw_ite', 'cfac': ['MAS']},
        {'dataset': 'NTVNBDMNFe_dw_ite', 'cfac': ['MAS']},
        {'dataset': 'HERACOMBNCEM'},
        {'dataset': 'HERACOMBNCEP460'},
        {'dataset': 'HERACOMBNCEP575'},
        {'dataset': 'HERACOMBNCEP820'},
        {'dataset': 'HERACOMBNCEP920'},
        {'dataset': 'HERACOMBCCEM'},
        {'dataset': 'HERACOMBCCEP'},
        {'dataset': 'HERACOMB_SIGMARED_C'},
        {'dataset': 'HERACOMB_SIGMARED_B'},
        {'dataset': 'DYE886R_dw_ite', 'cfac': ['QCD']},
        {'dataset': 'DYE886P', 'cfac': ['QCD']},
        {'dataset': 'DYE605_dw_ite', 'cfac': ['QCD']},
        {'dataset': 'DYE906R_dw_ite', 'cfac': ['ACC', 'QCD']},
    ]

    # NLO dataset
    dataset_40_nlo = [
        {'dataset': 'NMCPD_dw_ite'},
        {'dataset': 'NMC'},
        {'dataset': 'SLACP_dwsh'},
        {'dataset': 'SLACD_dw_ite'},
        {'dataset': 'BCDMSP_dwsh'},
        {'dataset': 'BCDMSD_dw_ite'},
        {'dataset': 'CHORUSNUPb_dw_ite'},
        {'dataset': 'CHORUSNBPb_dw_ite'},
        {'dataset': 'NTVNUDMNFe_dw_ite', 'cfac': []},
        {'dataset': 'NTVNBDMNFe_dw_ite', 'cfac': []},
        {'dataset': 'HERACOMBNCEM'},
        {'dataset': 'HERACOMBNCEP460'},
        {'dataset': 'HERACOMBNCEP575'},
        {'dataset': 'HERACOMBNCEP820'},
        {'dataset': 'HERACOMBNCEP920'},
        {'dataset': 'HERACOMBCCEM'},
        {'dataset': 'HERACOMBCCEP'},
        {'dataset': 'HERACOMB_SIGMARED_C'},
        {'dataset': 'HERACOMB_SIGMARED_B'},
        {'dataset': 'DYE886R_dw_ite', 'cfac': []},
        {'dataset': 'DYE886P', 'cfac': []},
        {'dataset': 'DYE605_dw_ite', 'cfac': []},
        {'dataset': 'DYE906R_dw_ite', 'cfac': ['ACC']},
        {'dataset': 'CDFZRAP_NEW', 'cfac': []},
        {'dataset': 'D0ZRAP_40', 'cfac': []},
        {'dataset': 'D0WMASY', 'cfac': []},
        {'dataset': 'ATLASWZRAP36PB', 'cfac': []},
        {'dataset': 'ATLASZHIGHMASS49FB', 'cfac': []},
        {'dataset': 'ATLASLOMASSDY11EXT', 'cfac': []},
        {'dataset': 'ATLASWZRAP11CC', 'cfac': []},
        {'dataset': 'ATLASWZRAP11CF', 'cfac': []},
        {'dataset': 'ATLASDY2D8TEV', 'cfac': []},
        {'dataset': 'ATLAS_DY_2D_8TEV_LOWMASS', 'cfac': []},
        {'dataset': 'ATLAS_WZ_TOT_13TEV', 'cfac': ['NRM']},
        {'dataset': 'ATLAS_WP_JET_8TEV_PT', 'cfac': []},
        {'dataset': 'ATLAS_WM_JET_8TEV_PT', 'cfac': []},
        {'dataset': 'ATLASZPT8TEVMDIST', 'cfac': [], 'sys': 10},
        {'dataset': 'ATLASZPT8TEVYDIST', 'cfac': [], 'sys': 10},
        {'dataset': 'ATLASTTBARTOT7TEV', 'cfac': []},
        {'dataset': 'ATLASTTBARTOT8TEV', 'cfac': []},
        {'dataset': 'ATLAS_TTBARTOT_13TEV_FULLLUMI', 'cfac': []},
        {'dataset': 'ATLAS_TTB_DIFF_8TEV_LJ_TRAPNORM', 'cfac': []},
        {'dataset': 'ATLAS_TTB_DIFF_8TEV_LJ_TTRAPNORM', 'cfac': []},
        {'dataset': 'ATLAS_TOPDIFF_DILEPT_8TEV_TTRAPNORM', 'cfac': []},
        {'dataset': 'ATLAS_1JET_8TEV_R06_DEC', 'cfac': []},
        {'dataset': 'ATLAS_2JET_7TEV_R06', 'cfac': []},
        {'dataset': 'ATLASPHT15_SF', 'cfac': ['EWK']},
        {'dataset': 'ATLAS_SINGLETOP_TCH_R_7TEV', 'cfac': []},
        {'dataset': 'ATLAS_SINGLETOP_TCH_R_13TEV', 'cfac': []},
        {'dataset': 'ATLAS_SINGLETOP_TCH_DIFF_7TEV_T_RAP_NORM', 'cfac': []},
        {'dataset': 'ATLAS_SINGLETOP_TCH_DIFF_7TEV_TBAR_RAP_NORM', 'cfac': []},
        {'dataset': 'ATLAS_SINGLETOP_TCH_DIFF_8TEV_T_RAP_NORM', 'cfac': []},
        {'dataset': 'ATLAS_SINGLETOP_TCH_DIFF_8TEV_TBAR_RAP_NORM', 'cfac': []},
        {'dataset': 'CMSWEASY840PB', 'cfac': []},
        {'dataset': 'CMSWMASY47FB', 'cfac': []},
        {'dataset': 'CMSDY2D11', 'cfac': []},
        {'dataset': 'CMSWMU8TEV', 'cfac': []},
        {'dataset': 'CMSZDIFF12', 'cfac': ['NRM'], 'sys': 10},
        {'dataset': 'CMS_2JET_7TEV', 'cfac': []},
        {'dataset': 'CMS_1JET_8TEV', 'cfac': []},
        {'dataset': 'CMSTTBARTOT7TEV', 'cfac': []},
        {'dataset': 'CMSTTBARTOT8TEV', 'cfac': []},
        {'dataset': 'CMSTTBARTOT13TEV', 'cfac': []},
        {'dataset': 'CMSTOPDIFF8TEVTTRAPNORM', 'cfac': []},
        {'dataset': 'CMSTTBARTOT5TEV', 'cfac': []},
        {'dataset': 'CMS_TTBAR_2D_DIFF_MTT_TRAP_NORM', 'cfac': []},
        {'dataset': 'CMS_TTB_DIFF_13TEV_2016_2L_TRAP', 'cfac': []},
        {'dataset': 'CMS_TTB_DIFF_13TEV_2016_LJ_TRAP', 'cfac': []},
        {'dataset': 'CMS_SINGLETOP_TCH_TOT_7TEV', 'cfac': []},
        {'dataset': 'CMS_SINGLETOP_TCH_R_8TEV', 'cfac': []},
        {'dataset': 'CMS_SINGLETOP_TCH_R_13TEV', 'cfac': []},
        {'dataset': 'LHCBZ940PB', 'cfac': []},
        {'dataset': 'LHCBZEE2FB_40', 'cfac': []},
        {'dataset': 'LHCBWZMU7TEV', 'cfac': ['NRM']},
        {'dataset': 'LHCBWZMU8TEV', 'cfac': ['NRM']},
        {'dataset': 'LHCB_Z_13TEV_DIMUON', 'cfac': []},
        {'dataset': 'LHCB_Z_13TEV_DIELECTRON', 'cfac': []},
    ]

    # dataset_test=[{'dataset': 'HERACOMBCCEM_test'}]
    dataset_test = [{'dataset': 'LHCB_Z0_13TEV_DIELECTRON-Y'}]

    dataset_40_new = [
        {'dataset': 'NMC_NC_NOTFIXED_DW_EM-F2', 'variant': 'legacy'},
        {'dataset': 'NMC_NC_NOTFIXED_P_EM-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'SLAC_NC_NOTFIXED_P_DW_EM-F2', 'variant': 'legacy'},
        {'dataset': 'SLAC_NC_NOTFIXED_D_DW_EM-F2', 'variant': 'legacy'},
        {'dataset': 'BCDMS_NC_NOTFIXED_P_DW_EM-F2', 'variant': 'legacy'},
        {'dataset': 'BCDMS_NC_NOTFIXED_D_DW_EM-F2', 'variant': 'legacy'},
        {'dataset': 'CHORUS_CC_NOTFIXED_PB_DW_NU-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'CHORUS_CC_NOTFIXED_PB_DW_NB-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'NUTEV_CC_NOTFIXED_FE_DW_NU-SIGMARED', 'cfac': ['MAS'], 'variant': 'legacy'},
        {'dataset': 'NUTEV_CC_NOTFIXED_FE_DW_NB-SIGMARED', 'cfac': ['MAS'], 'variant': 'legacy'},
        {'dataset': 'HERA_NC_318GEV_EM-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'HERA_NC_225GEV_EP-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'HERA_NC_251GEV_EP-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'HERA_NC_300GEV_EP-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'HERA_NC_318GEV_EP-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'HERA_CC_318GEV_EM-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'HERA_CC_318GEV_EP-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'HERA_NC_318GEV_EAVG_CHARM-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'HERA_NC_318GEV_EAVG_BOTTOM-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'DYE866_Z0_800GEV_DW_RATIO_PDXSECRATIO', 'variant': 'legacy'},
        {'dataset': 'DYE866_Z0_800GEV_PXSEC', 'variant': 'legacy'},
        {'dataset': 'DYE605_Z0_38P8GEV_DW_PXSEC', 'variant': 'legacy'},
        {'dataset': 'DYE906_Z0_120GEV_DW_PDXSECRATIO', 'cfac': ['ACC'], 'variant': 'legacy'},
        {'dataset': 'CDF_Z0_1P96TEV_ZRAP', 'variant': 'legacy'},
        {'dataset': 'D0_Z0_1P96TEV_ZRAP', 'variant': 'legacy'},
        {'dataset': 'D0_WPWM_1P96TEV_ASY', 'variant': 'legacy'},
        {'dataset': 'ATLAS_WPWM_7TEV_36PB_ETA', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_7TEV_36PB_ETA', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_7TEV_49FB_HIMASS', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_7TEV_LOMASS_M', 'variant': 'legacy'},
        {'dataset': 'ATLAS_WPWM_7TEV_46FB_CC-ETA', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_7TEV_46FB_CC-Y', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_7TEV_46FB_CF-Y', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_8TEV_HIMASS_M-Y', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_8TEV_LOWMASS_M-Y', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_13TEV_TOT', 'cfac': ['NRM'], 'variant': 'legacy'},
        {'dataset': 'ATLAS_WPWM_13TEV_TOT', 'cfac': ['NRM'], 'variant': 'legacy'},
        {'dataset': 'ATLAS_WJ_8TEV_WP-PT', 'variant': 'legacy'},
        {'dataset': 'ATLAS_WJ_8TEV_WM-PT', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0J_8TEV_PT-M', 'variant': 'legacy_10'},
        {'dataset': 'ATLAS_Z0J_8TEV_PT-Y', 'variant': 'legacy_10'},
        {'dataset': 'ATLAS_TTBAR_7TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'ATLAS_TTBAR_8TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'ATLAS_TTBAR_13TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'ATLAS_TTBAR_8TEV_LJ_DIF_YT-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_TTBAR_8TEV_LJ_DIF_YTTBAR-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_TTBAR_8TEV_2L_DIF_YTTBAR-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_1JET_8TEV_R06_PTY', 'variant': 'legacy_decorrelated'},
        {'dataset': 'ATLAS_2JET_7TEV_R06_M12Y', 'variant': 'legacy'},
        {'dataset': 'ATLAS_PH_13TEV_XSEC', 'cfac': ['EWK'], 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_7TEV_TCHANNEL-XSEC', 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_13TEV_TCHANNEL-XSEC', 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_7TEV_T-Y-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_7TEV_TBAR-Y-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_8TEV_T-RAP-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_8TEV_TBAR-RAP-NORM', 'variant': 'legacy'},
        {'dataset': 'CMS_WPWM_7TEV_ELECTRON_ASY'},
        {'dataset': 'CMS_WPWM_7TEV_MUON_ASY', 'variant': 'legacy'},
        {'dataset': 'CMS_Z0_7TEV_DIMUON_2D'},
        {'dataset': 'CMS_WPWM_8TEV_MUON_Y', 'variant': 'legacy'},
        {'dataset': 'CMS_Z0J_8TEV_PT-Y', 'cfac': ['NRM'], 'variant': 'legacy_10'},
        {'dataset': 'CMS_2JET_7TEV_M12Y'},
        {'dataset': 'CMS_1JET_8TEV_PTY', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_7TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_8TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_13TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_8TEV_LJ_DIF_YTTBAR-NORM', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_5TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT-NORM', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_13TEV_2L_DIF_YT', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_13TEV_LJ_2016_DIF_YTTBAR', 'variant': 'legacy'},
        {'dataset': 'CMS_SINGLETOP_7TEV_TCHANNEL-XSEC', 'variant': 'legacy'},
        {'dataset': 'CMS_SINGLETOP_8TEV_TCHANNEL-XSEC', 'variant': 'legacy'},
        {'dataset': 'CMS_SINGLETOP_13TEV_TCHANNEL-XSEC', 'variant': 'legacy'},
        {'dataset': 'LHCB_Z0_7TEV_DIELECTRON_Y'},
        {'dataset': 'LHCB_Z0_8TEV_DIELECTRON_Y'},
        {'dataset': 'LHCB_WPWM_7TEV_MUON_Y', 'cfac': ['NRM']},
        {'dataset': 'LHCB_Z0_7TEV_MUON_Y', 'cfac': ['NRM']},
        {'dataset': 'LHCB_WPWM_8TEV_MUON_Y', 'cfac': ['NRM']},
        {'dataset': 'LHCB_Z0_8TEV_MUON_Y', 'cfac': ['NRM']},
        {'dataset': 'LHCB_Z0_13TEV_DIMUON-Y'},
        {'dataset': 'LHCB_Z0_13TEV_DIELECTRON-Y'},
    ]

    # dataset_40_new=[{'dataset': 'NMC_NC_NOTFIXED_DW_EM-F2', 'variant': 'legacy'}]

    dataset_40_new = [
        {'dataset': 'NMC_NC_NOTFIXED_DW_EM-F2', 'variant': 'legacy'},
        {'dataset': 'NMC_NC_NOTFIXED_P_EM-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'SLAC_NC_NOTFIXED_P_DW_EM-F2', 'variant': 'legacy'},
        {'dataset': 'SLAC_NC_NOTFIXED_D_DW_EM-F2', 'variant': 'legacy'},
        {'dataset': 'BCDMS_NC_NOTFIXED_P_DW_EM-F2', 'variant': 'legacy'},
        {'dataset': 'BCDMS_NC_NOTFIXED_D_DW_EM-F2', 'variant': 'legacy'},
        {'dataset': 'CHORUS_CC_NOTFIXED_PB_DW_NU-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'CHORUS_CC_NOTFIXED_PB_DW_NB-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'NUTEV_CC_NOTFIXED_FE_DW_NU-SIGMARED', 'cfac': ['MAS'], 'variant': 'legacy'},
        {'dataset': 'NUTEV_CC_NOTFIXED_FE_DW_NB-SIGMARED', 'cfac': ['MAS'], 'variant': 'legacy'},
        {'dataset': 'HERA_NC_318GEV_EM-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'HERA_NC_225GEV_EP-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'HERA_NC_251GEV_EP-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'HERA_NC_300GEV_EP-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'HERA_NC_318GEV_EP-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'HERA_CC_318GEV_EM-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'HERA_CC_318GEV_EP-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'HERA_NC_318GEV_EAVG_CHARM-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'HERA_NC_318GEV_EAVG_BOTTOM-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'DYE866_Z0_800GEV_DW_RATIO_PDXSECRATIO', 'variant': 'legacy'},
        {'dataset': 'DYE866_Z0_800GEV_PXSEC', 'variant': 'legacy'},
        {'dataset': 'DYE605_Z0_38P8GEV_DW_PXSEC', 'variant': 'legacy'},
        {'dataset': 'DYE906_Z0_120GEV_DW_PDXSECRATIO', 'cfac': ['ACC'], 'variant': 'legacy'},
        {'dataset': 'CDF_Z0_1P96TEV_ZRAP', 'variant': 'legacy'},
        {'dataset': 'D0_Z0_1P96TEV_ZRAP', 'variant': 'legacy'},
        {'dataset': 'D0_WPWM_1P96TEV_ASY', 'variant': 'legacy'},
        {'dataset': 'ATLAS_WPWM_7TEV_36PB_ETA', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_7TEV_36PB_ETA', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_7TEV_49FB_HIMASS', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_7TEV_LOMASS_M', 'variant': 'legacy'},
        {'dataset': 'ATLAS_WPWM_7TEV_46FB_CC-ETA', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_7TEV_46FB_CC-Y', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_7TEV_46FB_CF-Y', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_8TEV_HIMASS_M-Y', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_8TEV_LOWMASS_M-Y', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_13TEV_TOT', 'cfac': ['NRM'], 'variant': 'legacy'},
        {'dataset': 'ATLAS_WPWM_13TEV_TOT', 'cfac': ['NRM'], 'variant': 'legacy'},
        {'dataset': 'ATLAS_WJ_8TEV_WP-PT', 'variant': 'legacy'},
        {'dataset': 'ATLAS_WJ_8TEV_WM-PT', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0J_8TEV_PT-M', 'variant': 'legacy_10'},
        {'dataset': 'ATLAS_Z0J_8TEV_PT-Y', 'variant': 'legacy_10'},
        {'dataset': 'ATLAS_TTBAR_7TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'ATLAS_TTBAR_8TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'ATLAS_TTBAR_13TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'ATLAS_TTBAR_8TEV_LJ_DIF_YT-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_TTBAR_8TEV_LJ_DIF_YTTBAR-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_TTBAR_8TEV_2L_DIF_YTTBAR-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_1JET_8TEV_R06_PTY', 'variant': 'legacy_decorrelated'},
        {'dataset': 'ATLAS_2JET_7TEV_R06_M12Y', 'variant': 'legacy'},
        {'dataset': 'ATLAS_PH_13TEV_XSEC', 'cfac': ['EWK'], 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_7TEV_TCHANNEL-XSEC', 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_13TEV_TCHANNEL-XSEC', 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_7TEV_T-Y-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_7TEV_TBAR-Y-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_8TEV_T-RAP-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_8TEV_TBAR-RAP-NORM', 'variant': 'legacy'},
        {'dataset': 'CMS_WPWM_7TEV_ELECTRON_ASY'},
        {'dataset': 'CMS_WPWM_7TEV_MUON_ASY', 'variant': 'legacy'},
        {'dataset': 'CMS_Z0_7TEV_DIMUON_2D'},
        {'dataset': 'CMS_WPWM_8TEV_MUON_Y', 'variant': 'legacy'},
        {'dataset': 'CMS_Z0J_8TEV_PT-Y', 'cfac': ['NRM'], 'variant': 'legacy_10'},
        {'dataset': 'CMS_2JET_7TEV_M12Y'},
        {'dataset': 'CMS_1JET_8TEV_PTY', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_7TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_8TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_13TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_8TEV_LJ_DIF_YTTBAR-NORM', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_5TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT-NORM', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_13TEV_2L_DIF_YT', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_13TEV_LJ_2016_DIF_YTTBAR', 'variant': 'legacy'},
        {'dataset': 'CMS_SINGLETOP_7TEV_TCHANNEL-XSEC', 'variant': 'legacy'},
        {'dataset': 'CMS_SINGLETOP_8TEV_TCHANNEL-XSEC', 'variant': 'legacy'},
        {'dataset': 'CMS_SINGLETOP_13TEV_TCHANNEL-XSEC', 'variant': 'legacy'},
        {'dataset': 'LHCB_Z0_7TEV_DIELECTRON_Y'},
        {'dataset': 'LHCB_Z0_8TEV_DIELECTRON_Y'},
        {'dataset': 'LHCB_WPWM_7TEV_MUON_Y', 'cfac': ['NRM']},
        {'dataset': 'LHCB_Z0_7TEV_MUON_Y', 'cfac': ['NRM']},
        {'dataset': 'LHCB_WPWM_8TEV_MUON_Y', 'cfac': ['NRM']},
        {'dataset': 'LHCB_Z0_8TEV_MUON_Y', 'cfac': ['NRM']},
        {'dataset': 'LHCB_Z0_13TEV_DIMUON-Y'},
        {'dataset': 'LHCB_Z0_13TEV_DIELECTRON-Y'},
    ]

    dataset_hh_new = [
        {'dataset': 'CDF_Z0_1P96TEV_ZRAP', 'variant': 'legacy'},
        {'dataset': 'D0_Z0_1P96TEV_ZRAP', 'variant': 'legacy'},
        {'dataset': 'D0_WPWM_1P96TEV_ASY', 'variant': 'legacy'},
        {'dataset': 'ATLAS_WPWM_7TEV_36PB_ETA', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_7TEV_36PB_ETA', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_7TEV_49FB_HIMASS', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_7TEV_LOMASS_M', 'variant': 'legacy'},
        {'dataset': 'ATLAS_WPWM_7TEV_46FB_CC-ETA', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_7TEV_46FB_CC-Y', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_7TEV_46FB_CF-Y', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_8TEV_HIMASS_M-Y', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_8TEV_LOWMASS_M-Y', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_13TEV_TOT', 'cfac': ['NRM'], 'variant': 'legacy'},
        {'dataset': 'ATLAS_WPWM_13TEV_TOT', 'cfac': ['NRM'], 'variant': 'legacy'},
        {'dataset': 'ATLAS_WJ_8TEV_WP-PT', 'variant': 'legacy'},
        {'dataset': 'ATLAS_WJ_8TEV_WM-PT', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0J_8TEV_PT-M', 'variant': 'legacy_10'},
        {'dataset': 'ATLAS_Z0J_8TEV_PT-Y', 'variant': 'legacy_10'},
        {'dataset': 'ATLAS_TTBAR_7TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'ATLAS_TTBAR_8TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'ATLAS_TTBAR_13TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'ATLAS_TTBAR_8TEV_LJ_DIF_YT-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_TTBAR_8TEV_LJ_DIF_YTTBAR-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_TTBAR_8TEV_2L_DIF_YTTBAR-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_1JET_8TEV_R06_PTY', 'variant': 'legacy_decorrelated'},
        {'dataset': 'ATLAS_2JET_7TEV_R06_M12Y', 'variant': 'legacy'},
        {'dataset': 'ATLAS_PH_13TEV_XSEC', 'cfac': ['EWK'], 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_7TEV_TCHANNEL-XSEC', 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_13TEV_TCHANNEL-XSEC', 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_7TEV_T-Y-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_7TEV_TBAR-Y-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_8TEV_T-RAP-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_8TEV_TBAR-RAP-NORM', 'variant': 'legacy'},
        {'dataset': 'CMS_WPWM_7TEV_ELECTRON_ASY'},
        {'dataset': 'CMS_WPWM_7TEV_MUON_ASY', 'variant': 'legacy'},
        {'dataset': 'CMS_Z0_7TEV_DIMUON_2D'},
        {'dataset': 'CMS_WPWM_8TEV_MUON_Y', 'variant': 'legacy'},
        {'dataset': 'CMS_Z0J_8TEV_PT-Y', 'cfac': ['NRM'], 'variant': 'legacy_10'},
        {'dataset': 'CMS_2JET_7TEV_M12Y'},
        {'dataset': 'CMS_1JET_8TEV_PTY', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_7TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_8TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_13TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_8TEV_LJ_DIF_YTTBAR-NORM', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_5TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT-NORM', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_13TEV_2L_DIF_YT', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_13TEV_LJ_2016_DIF_YTTBAR', 'variant': 'legacy'},
        {'dataset': 'CMS_SINGLETOP_7TEV_TCHANNEL-XSEC', 'variant': 'legacy'},
        {'dataset': 'CMS_SINGLETOP_8TEV_TCHANNEL-XSEC', 'variant': 'legacy'},
        {'dataset': 'CMS_SINGLETOP_13TEV_TCHANNEL-XSEC', 'variant': 'legacy'},
        {'dataset': 'LHCB_Z0_7TEV_DIELECTRON_Y'},
        {'dataset': 'LHCB_Z0_8TEV_DIELECTRON_Y'},
        {'dataset': 'LHCB_WPWM_7TEV_MUON_Y', 'cfac': ['NRM']},
        {'dataset': 'LHCB_Z0_7TEV_MUON_Y', 'cfac': ['NRM']},
        {'dataset': 'LHCB_WPWM_8TEV_MUON_Y', 'cfac': ['NRM']},
        {'dataset': 'LHCB_Z0_8TEV_MUON_Y', 'cfac': ['NRM']},
        {'dataset': 'LHCB_Z0_13TEV_DIMUON-Y'},
        {'dataset': 'LHCB_Z0_13TEV_DIELECTRON-Y'},
    ]

    dataset_hh_heralight_new = [
        {'dataset': 'HERA_NC_318GEV_EM-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'HERA_NC_225GEV_EP-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'HERA_NC_251GEV_EP-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'HERA_NC_300GEV_EP-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'HERA_NC_318GEV_EP-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'HERA_CC_318GEV_EM-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'HERA_CC_318GEV_EP-SIGMARED', 'variant': 'legacy'},
        {'dataset': 'CDF_Z0_1P96TEV_ZRAP', 'variant': 'legacy'},
        {'dataset': 'D0_Z0_1P96TEV_ZRAP', 'variant': 'legacy'},
        {'dataset': 'D0_WPWM_1P96TEV_ASY', 'variant': 'legacy'},
        {'dataset': 'ATLAS_WPWM_7TEV_36PB_ETA', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_7TEV_36PB_ETA', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_7TEV_49FB_HIMASS', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_7TEV_LOMASS_M', 'variant': 'legacy'},
        {'dataset': 'ATLAS_WPWM_7TEV_46FB_CC-ETA', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_7TEV_46FB_CC-Y', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_7TEV_46FB_CF-Y', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_8TEV_HIMASS_M-Y', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_8TEV_LOWMASS_M-Y', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0_13TEV_TOT', 'cfac': ['NRM'], 'variant': 'legacy'},
        {'dataset': 'ATLAS_WPWM_13TEV_TOT', 'cfac': ['NRM'], 'variant': 'legacy'},
        {'dataset': 'ATLAS_WJ_8TEV_WP-PT', 'variant': 'legacy'},
        {'dataset': 'ATLAS_WJ_8TEV_WM-PT', 'variant': 'legacy'},
        {'dataset': 'ATLAS_Z0J_8TEV_PT-M', 'variant': 'legacy_10'},
        {'dataset': 'ATLAS_Z0J_8TEV_PT-Y', 'variant': 'legacy_10'},
        {'dataset': 'ATLAS_TTBAR_7TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'ATLAS_TTBAR_8TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'ATLAS_TTBAR_13TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'ATLAS_TTBAR_8TEV_LJ_DIF_YT-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_TTBAR_8TEV_LJ_DIF_YTTBAR-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_TTBAR_8TEV_2L_DIF_YTTBAR-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_1JET_8TEV_R06_PTY', 'variant': 'legacy_decorrelated'},
        {'dataset': 'ATLAS_2JET_7TEV_R06_M12Y', 'variant': 'legacy'},
        {'dataset': 'ATLAS_PH_13TEV_XSEC', 'cfac': ['EWK'], 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_7TEV_TCHANNEL-XSEC', 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_13TEV_TCHANNEL-XSEC', 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_7TEV_T-Y-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_7TEV_TBAR-Y-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_8TEV_T-RAP-NORM', 'variant': 'legacy'},
        {'dataset': 'ATLAS_SINGLETOP_8TEV_TBAR-RAP-NORM', 'variant': 'legacy'},
        {'dataset': 'CMS_WPWM_7TEV_ELECTRON_ASY'},
        {'dataset': 'CMS_WPWM_7TEV_MUON_ASY', 'variant': 'legacy'},
        {'dataset': 'CMS_Z0_7TEV_DIMUON_2D'},
        {'dataset': 'CMS_WPWM_8TEV_MUON_Y', 'variant': 'legacy'},
        {'dataset': 'CMS_Z0J_8TEV_PT-Y', 'cfac': ['NRM'], 'variant': 'legacy_10'},
        {'dataset': 'CMS_2JET_7TEV_M12Y'},
        {'dataset': 'CMS_1JET_8TEV_PTY', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_7TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_8TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_13TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_8TEV_LJ_DIF_YTTBAR-NORM', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_5TEV_TOT_X-SEC', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT-NORM', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_13TEV_2L_DIF_YT', 'variant': 'legacy'},
        {'dataset': 'CMS_TTBAR_13TEV_LJ_2016_DIF_YTTBAR', 'variant': 'legacy'},
        {'dataset': 'CMS_SINGLETOP_7TEV_TCHANNEL-XSEC', 'variant': 'legacy'},
        {'dataset': 'CMS_SINGLETOP_8TEV_TCHANNEL-XSEC', 'variant': 'legacy'},
        {'dataset': 'CMS_SINGLETOP_13TEV_TCHANNEL-XSEC', 'variant': 'legacy'},
        {'dataset': 'LHCB_Z0_7TEV_DIELECTRON_Y'},
        {'dataset': 'LHCB_Z0_8TEV_DIELECTRON_Y'},
        {'dataset': 'LHCB_WPWM_7TEV_MUON_Y', 'cfac': ['NRM']},
        {'dataset': 'LHCB_Z0_7TEV_MUON_Y', 'cfac': ['NRM']},
        {'dataset': 'LHCB_WPWM_8TEV_MUON_Y', 'cfac': ['NRM']},
        {'dataset': 'LHCB_Z0_8TEV_MUON_Y', 'cfac': ['NRM']},
        {'dataset': 'LHCB_Z0_13TEV_DIMUON-Y'},
        {'dataset': 'LHCB_Z0_13TEV_DIELECTRON-Y'},
    ]

    #   For certain pbeam datasets cannnot load fk tables in default routine so do separately
    load_fk = True

    # dataset_40_new=[{'dataset': 'DYE906_Z0_120GEV_DW_PDXSECRATIO', 'cfac': ['ACC'], 'variant': 'legacy'}]
    # 30.474249518879027
    # If true then group datasets together for dynamic tolerance
    dynT_group = True
    # If true then remove datasets with n < 5 from dynamic tolerance
    dynT_ngt5 = True
    preds_stored = {}
    datapath = ''
    theories_path = ''
    newmin = True  # This should always be set to True
    dataset_ii_global = ''
    pdf_dict = []


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
    """

    datasets: tuple
    # NOTE: intersection cuts are being ignored, but should've been added upon construction

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

        cds = [ds.load_commondata() for ds in datasets]
        covmat = dataset_inputs_covmat_from_systematics(
            cds, _list_of_central_values=central_values, use_weights_in_covmat=False
        )
        return covmat

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


# Limite the shared global data to what's available in this dictionary
shared_global_data = {"data": None, "posdata": None}

sensible_positivity_cuts = [
    {"dataset": "NNPDF_POS_2P24GEV_FLL", "rule": "x > 5.0e-7"},
    {"dataset": "NNPDF_POS_2P24GEV_F2C", "rule": "x < 0.74"},
    {"dataset": "NNPDF_POS_2P24GEV_XGL", "rule": "x > 0.1"},
    {"dataset": "NNPDF_POS_2P24GEV_XUQ", "rule": "x > 0.1"},
    {"dataset": "NNPDF_POS_2P24GEV_XUB", "rule": "x > 0.1"},
    {"dataset": "NNPDF_POS_2P24GEV_XDQ", "rule": "x > 0.1"},
    {"dataset": "NNPDF_POS_2P24GEV_XDB", "rule": "x > 0.1"},
    {"dataset": "NNPDF_POS_2P24GEV_XSQ", "rule": "x > 0.1"},
    {"dataset": "NNPDF_POS_2P24GEV_XSB", "rule": "x > 0.1"},
]


def shared_populate_data():

    config = {"theoryid": fit_pars.theoryidi, "use_cuts": "internal"}

    datasets = []
    for dinput in fit_pars.dataset_40:
        ds = API.dataset(dataset_input=dinput, **config)
        datasets.append(ds)

    positivity_datasets = API.posdatasets(
        posdatasets=fit_pars.pos_data31 + fit_pars.pos_data40,
        **config,
        added_filter_rules=sensible_positivity_cuts
    )

    shared_global_data["data"] = DataHolder(tuple(datasets))
    shared_global_data["posdata"] = DataHolder(tuple(positivity_datasets))


# Instantiate all global configurations
# Change this perhaps to a namedtuple or something better
global_configuration = {
    "chi2": Chi2Pars(),
    "min": MinPars(),
    "closure": PDFClosure(),
    "dload": DloadPars(),
}
# basis_pars = None
# pdf_pars = None
# inout_pars = None
chi2_pars = global_configuration["chi2"]
min_pars = global_configuration["min"]
pdf_closure = global_configuration["closure"]
dload_pars = global_configuration["dload"]
