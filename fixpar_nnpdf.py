import warnings as wa
from validphys import core
from validphys.api import API
from validphys.fkparser import load_fktable
from validphys.loader import Loader
from validphys.convolution import predictions, linear_predictions, central_predictions
import validphys.convolution as conv
# from validphys.results import experiments_index
import validphys.results as rs
from validphys.covmats import covmat_from_systematics
import numpy as np
from validphys.commondataparser import load_commondata
import validphys.commondata as cdat
import pandas as pd
from validphys.calcutils import calc_chi2
import scipy.linalg as la
import os
import lhapdf
import shutil as sh
from scipy.integrate import quadrature
from scipy.integrate import fixed_quad
from validphys.covmats import dataset_inputs_covmat_from_systematics
from validphys.pseudodata import make_replica
from scipy.optimize import minimize
from scipy.optimize import least_squares
import itertools as IT
from reportengine.utils import yaml_safe
import time
import sys
import argparse
import pathlib
from validphys.loader import _get_nnpdf_profile
from validphys.lhaindex import get_lha_datapath

lhapdf.setVerbosity(0)
sys.path.append("src/")

from global_pars import *
from outputs import *
from inputs import *
from pdfs import *
from chebyshevs import *
from data_theory import *
from chi2s import *
from levmar import *
from lhapdf_funs import *
from error_calc import *

if DEBUG:
# # Fake LHAPDF folder, no longer needed
    TEMP_LHAPDF = pathlib.Path("tmp_lhapdf")
    TEMP_LHAPDF.mkdir(exist_ok=True)
    lhapdf.pathsAppend(TEMP_LHAPDF.as_posix())


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="Path to the config file")

args = parser.parse_args()

##################
# Loading yaml config
##################
with open(args.config, 'r') as file:
        config = yaml_safe.load(file)

inp = config.get("inputs", None)
inout_pars.inputnam=inp.get("input file")
inout_pars.covinput=inp.get("cov input file")
inout_pars.readcov=inp.get("readcov")

outp = config.get("outputs", None)
inout_pars.label=outp.get("label")


bas = config.get("basis pars", None)
basis_pars.g_second_term=bas.get("g_second_term")
basis_pars.g_cheb7=bas.get("g_cheb7")
basis_pars.asp_fix=bas.get("asp_fix")
basis_pars.dvd_eq_uvd=bas.get("dvd_eq_uvd")
basis_pars.t8_int=bas.get("t8_int")
if basis_pars.t8_int is None:
    basis_pars.t8_int=False
basis_pars.Cheb_8=bas.get("Cheb_8")
if basis_pars.Cheb_8 is None:
    basis_pars.Cheb_8=False

if basis_pars.Cheb_8:
    basis_pars.i_uv_min=0
    basis_pars.i_uv_max+=2
    basis_pars.i_dv_max+=4
    basis_pars.i_sea_max+=6
    basis_pars.i_sp_max+=8
    basis_pars.i_g_max+=10
    basis_pars.i_sm_max+=10
    basis_pars.i_dbub_max+=12
    basis_pars.i_ch_max+=14
    basis_pars.i_dv_min=basis_pars.i_uv_max
    basis_pars.i_sea_min=basis_pars.i_dv_max
    basis_pars.i_sp_min=basis_pars.i_sea_max
    basis_pars.i_g_min=basis_pars.i_sp_max
    basis_pars.i_sm_min=basis_pars.i_g_max
    basis_pars.i_dbub_min=basis_pars.i_sm_max
    basis_pars.i_ch_min=basis_pars.i_dbub_max

pdfp = config.get("pdf pars", None)
pdf_pars.lhin=pdfp.get("lhin")
pdf_pars.uselha=pdfp.get("uselha")
pdf_pars.PDFlabel_lhin=pdfp.get("PDFlabel_lhin")

pdfcl = config.get("pdf closure", None)
pdf_closure.pdfscat=pdfcl.get("pdfscat")
pdf_closure.pdlabel=pdfcl.get("pdlabel")
pdf_closure.pdpdf=pdfcl.get("pdpdf")

pdflag = config.get("pseudodata flags", None)
inout_pars.pdout=pdflag.get("pdout")
inout_pars.pdin=pdflag.get("pdin")
inout_pars.pd_output=pdflag.get("pd_output")
if inout_pars.pd_output is None:
    inout_pars.pd_output=False

fitp = config.get("fit pars")
fit_pars.fixpar=fitp.get("fixpar")
fit_pars.theoryidi=fitp.get("theoryidi")
fit_pars.nnpdf_pos=fitp.get("nnpdf_pos")
fit_pars.pos_40=fitp.get("pos_40")
fit_pars.pos_nogluon=fitp.get("pos_nogluon")
if fit_pars.pos_nogluon is None:
    fit_pars.pos_nogluon=False
fit_pars.pos_gluononly=fitp.get("pos_gluononly")
if fit_pars.pos_gluononly is None:
    fit_pars.pos_gluononly=False
fit_pars.pos_dyc=fitp.get("pos_dyc")
if fit_pars.pos_dyc is None:
    fit_pars.pos_dyc=False
fit_pars.pseud=fitp.get("pseud")
fit_pars.irep=fitp.get("irep")
fit_pars.lhrep=fitp.get("lhrep")
if fit_pars.lhrep is None:
    fit_pars.lhrep=0
fit_pars.nmcpd_diag=fitp.get("nmcpd_diag")
chi2_pars.t0_noderivin=fitp.get("t0_noderivin")
fit_pars.deld_const=fitp.get("deld_const")
fit_pars.dset_type=fitp.get("dset_type")
tollm=fitp.get("tollm")
if tollm is not None:
    min_pars.tollm=tollm
fit_pars.nlo_cuts=fitp.get("nlo_cuts")
if fit_pars.nlo_cuts is None:
    fit_pars.nlo_cuts=False

# Make sure that new_min is True
fit_pars.newmin=True

fit_pars.dynT_group=fitp.get("dynT_group")
if fit_pars.dynT_group is None:
    fit_pars.dynT_group=False

fit_pars.dynT_ngt5=fitp.get("dynT_ngt5")
if fit_pars.dynT_ngt5 is None:
    fit_pars.dynT_ngt5=False

chi2_pars.dynamic_tol=fitp.get("dynamic_tol")

if chi2_pars.dynamic_tol is None:
    chi2_pars.dynamic_tol=False

chi2_pars.t2_err=fitp.get("t2_err")

if chi2_pars.t2_err is None:
    chi2_pars.t2_err=1.


if fit_pars.pos_nogluon:
    fit_pars.pos_data31=fit_pars.pos_data31_nogluon
    fit_pars.pos_data40=fit_pars.pos_data40_nogluon

if fit_pars.pos_gluononly:
    fit_pars.pos_data31=fit_pars.pos_data31_gluononly
    fit_pars.pos_data40=fit_pars.pos_data40_gluononly

if fit_pars.pos_dyc:
    fit_pars.pos_40=False
    fit_pars.pos_data31=fit_pars.pos_data31_dyc

if fit_pars.dset_type is None:
    fit_pars.dset_type='global'

if fit_pars.dset_type == 'HERAonly':
    fit_pars.dataset_40=fit_pars.dataset_HERAonly
    fit_pars.imaxdat=len(fit_pars.dataset_40)
    for i in range(0,fit_pars.imaxdat):
        fit_pars.cftrue[i]=False
elif fit_pars.dset_type == 'noHERA':
    fit_pars.dataset_40=fit_pars.dataset_noHERA
    fit_pars.imaxdat=len(fit_pars.dataset_40)
    for i in range(10,19):
        fit_pars.cftrue[i]=True
    fit_pars.systrue[36]=False
    fit_pars.systrue[37]=False
    fit_pars.systrue[57]=False
    fit_pars.systrue[27]=True
    fit_pars.systrue[28]=True
    fit_pars.systrue[48]=True
elif fit_pars.dset_type == 'noLHC':
    fit_pars.dataset_40=fit_pars.dataset_noLHC
    fit_pars.imaxdat=len(fit_pars.dataset_40)
elif fit_pars.dset_type == 'hhcollideronly':
    fit_pars.dataset_40=fit_pars.dataset_hhcollideronly
    fit_pars.imaxdat=len(fit_pars.dataset_40)
    for i in range(0,8):
        fit_pars.cftrue[i]=True
    for i in range(10,19):
        fit_pars.cftrue[i]=True
    fit_pars.systrue[36]=False
    fit_pars.systrue[37]=False
    fit_pars.systrue[57]=False
    fit_pars.systrue[13]=True
    fit_pars.systrue[14]=True
    fit_pars.systrue[34]=True
elif fit_pars.dset_type == 'lowenergyDIS':
    fit_pars.dataset_40=fit_pars.dataset_lowenergyDIS
    fit_pars.imaxdat=len(fit_pars.dataset_40)
elif fit_pars.dset_type == 'lowenergyDY':
    fit_pars.dataset_40=fit_pars.dataset_lowenergyDY
    fit_pars.imaxdat=len(fit_pars.dataset_40)
    for i in range(0,4):
        fit_pars.cftrue[i]=True
elif fit_pars.dset_type == 'lowenergyDISDY':
    fit_pars.dataset_40=fit_pars.dataset_lowenergyDISDY
    fit_pars.imaxdat=len(fit_pars.dataset_40)
    for i in range(10,19):
        fit_pars.cftrue[i]=True
elif fit_pars.dset_type == 'lowenergyDISDY_HERAonly':
    fit_pars.dataset_40=fit_pars.dataset_lowenergyDISDY_HERAonly
    fit_pars.imaxdat=len(fit_pars.dataset_40)
elif fit_pars.dset_type == 'test':
    fit_pars.dataset_40=fit_pars.dataset_test
    fit_pars.imaxdat=len(fit_pars.dataset_40)
    for i in range(0,fit_pars.imaxdat):
        fit_pars.cftrue[i]=False
elif fit_pars.dset_type == 'global_new':
    fit_pars.dataset_40=fit_pars.dataset_40_new
    fit_pars.imaxdat=len(fit_pars.dataset_40)
    fit_pars.cftrue=np.zeros((82),dtype=bool)
    for i in range (0,len(fit_pars.dataset_40)):
        if "cfac" in fit_pars.dataset_40[i]:
            fit_pars.cftrue[i]=True
if inout_pars.pd_output:
    inout_pars.pd_output_lab=inout_pars.label
    inout_pars.label+='_irep'+str(fit_pars.irep)

if fit_pars.theoryidi==212:
    fit_pars.dataset_40=fit_pars.dataset_40_nlo
    fit_pars.imaxdat=len(fit_pars.dataset_40)

pdf_pars.lhapdfdir = get_lha_datapath() + "/"
if DEBUG:
    pdf_pars.tmp_lhapdfdir = TEMP_LHAPDF
profile=_get_nnpdf_profile(None)
fit_pars.datapath=pathlib.Path(profile["data_path"])
# fit_pars.theories_path=pathlib.Path(profile["theories_path"])

fit_pars.newmin=True

if chi2_pars.diff_4:
    chi2_pars.diff_2=True

if not pdf_pars.uselha:
    pdf_pars.lhin=False

if inout_pars.pdin:
    inout_pars.pdout=False

if pdf_pars.lhin:
    fit_pars.fixpar=True

fit_pars.alphas=0.118

tzero=time.process_time()

print('inputnam = ',inout_pars.inputnam)

if inout_pars.readcov:
    chi2_pars.t0=True
    # fit_pars.pos_const=False
    if fit_pars.nnpdf_pos:
        fit_pars.pos_const=True
    else:
        fit_pars.pos_const=False
    (afi,hess,jac)=readincov()
    if chi2_pars.dynamic_tol:
        # hesserror_dynamic_tol(afi,hess,jac)
        hesserror_dynamic_tol_new(afi,hess,jac)
    else:
        hesserror_new(afi,hess,jac)
        # hesserror(afi,hess,jac)

else:
    afi=readin()

outputfile=open('outputs/buffer/'+inout_pars.label+'.dat','a')
outputfile.write("inputnam = ")
outputfile.write(inout_pars.inputnam)
outputfile.write("\n")

# msht_kfs()

print(tzero)

use_levmar=True

# Prepare the initial PDF that might be used for the computation
# of the chi2, the minimization, etc.
if pdf_pars.lhin:
    # If using directly an LHAPDF input, load said PDF
    vp_pdf = API.pdf(pdf = pdf_pars.PDFlabel_lhin)
else:
    # NB the name is irrelevant since the PDF object doesn't leave the program
    pdfname = f"{inout_pars.label}_run+{pdf_pars.idir}"

    # Take the free parameters and create the parameter set
    parin = initpars()
    pdf_parameters_raw = parset(afi, parin, are_free = pdf_pars.par_isf)
    # Modify it so the sumrules work (is this independent from the parametrization or need the function below?)
    pdf_parameters = sumrules(pdf_parameters_raw)

    vp_pdf = MSHTPDF(name = pdfname, pdf_parameters = pdf_parameters, pdf_function = "msht")



if inout_pars.readcov:

    print('finish!')

elif len(afi)==0:  # no free pars

    if inout_pars.pdout:

        print('Output PD to file...')
        chi2_pars.t0=False
        fit_pars.pos_const=False
        chi2expi=chi2min(afi, vp_pdf=vp_pdf)

    elif(chi2_pars.chi2ind):
        chi2_pars.t0=True
        fit_pars.pos_const=False
        chi2t0i=chi2min(afi, vp_pdf=vp_pdf)

        print('chi2(t0) in (no pos pen):',chi2t0i,chi2t0i/chi2_pars.ndat)
        t1=time.process_time()
        print('time= ',t1-tzero)
    else:

        chi2_pars.t0=False
        fit_pars.pos_const=False
        chi2expi=chi2min(afi, vp_pdf=vp_pdf)
        chi2_pars.t0=True
        chi2t0i=chi2min(afi, vp_pdf=vp_pdf)
        fit_pars.pos_const=True


        if(inout_pars.pdin):

            chi2posi=chi2min(afi, vp_pdf=vp_pdf)
            pospeni=chi2posi-chi2t0i

            # print('chi2(exp) in:',chi2expi,(chi2expi)/chi2_pars.ndat)
            # t1=time.process_time()
            # print('time= ',t1-tzero)

            print('chi2(exp) in:',chi2expi+pospeni,(chi2expi+pospeni)/chi2_pars.ndat)
            print('chi2(t0) in:',(chi2t0i+pospeni),(chi2t0i+pospeni)/chi2_pars.ndat)
            print('pos penalty in = ',pospeni,pospeni/chi2_pars.ndat)
            print('chi2(exp) in (no pos pen):',chi2expi,chi2expi/chi2_pars.ndat)
            print('chi2(t0) in (no pos pen):',chi2t0i,chi2t0i/chi2_pars.ndat)
            t1=time.process_time()
            print('time= ',t1-tzero)

            gridout()
            parsout()
            plotout()
            evgrido()
            resout_nofit(pospeni,chi2t0i,chi2expi,chi2_pars.ndat)


        else:
            fit_pars.pos_const=True

            chi2posi=chi2min(afi, vp_pdf=vp_pdf)
            pospeni=chi2posi-chi2t0i

            print('chi2(exp) in:',chi2expi+pospeni,(chi2expi+pospeni)/chi2_pars.ndat)
            print('chi2(t0) in:',(chi2t0i+pospeni),(chi2t0i+pospeni)/chi2_pars.ndat)
            print('pos penalty in = ',pospeni,pospeni/chi2_pars.ndat)
            print('chi2(exp) in (no pos pen):',chi2expi,chi2expi/chi2_pars.ndat)
            print('chi2(t0) in (no pos pen):',chi2t0i,chi2t0i/chi2_pars.ndat)
            t1=time.process_time()
            print('time= ',t1-tzero)

            gridout()
            parsout()
            plotout()
            evgrido()
            resout_nofit(pospeni,chi2t0i,chi2expi,chi2_pars.ndat)

elif(use_levmar):

    print('Using LM algorithm...')
    print('afi = ', afi)

    if(pdf_closure.pdpdf):

        chi2_pars.t0=False
        fit_pars.pos_const=False
        chi2=chi2min(afi)
        print('chi2 in:',chi2,(chi2)/chi2_pars.ndat)

        chi2_pars.diff_2=True
        # afo=levmar(afi)
        afo=levmar(afi)

        print('afo = ', afo)

    else:
        # Compute the experimental chi2
        chi2_pars.t0=False
        fit_pars.pos_const=False
        chi2expi=chi2min(afi, vp_pdf = vp_pdf)

        # Now the t0 chi2
        chi2_pars.t0=True
        chi2t0i=chi2min(afi, vp_pdf = vp_pdf)

        # And now the positivity
        fit_pars.pos_const=True
        chi2posi=chi2min(afi, vp_pdf = vp_pdf)

        # TODO: these three calls should have t0/pos as input parameter to the chi2 function
        # and possibily also the PDF *object* should be an input
        pospeni=chi2posi-chi2t0i

        # Now compute the loss functions (ie., chi2+positivity)
        loss_exp = chi2expi+pospeni
        loss_t0 = chi2posi
        ndat = chi2_pars.ndat
        print(f"chi2(exp) in:{loss_exp:.8} {loss_exp/ndat:.5}")
        print(f"chi2(t0) in:{loss_t0:.8} {loss_t0/ndat:.5}")
        print(f"chi2(exp) in:{pospeni:.5}")

        chi2_pars.t0=True
        if fit_pars.nnpdf_pos:
            fit_pars.pos_const=True
        else:
            fit_pars.pos_const=False

        chi2_pars.diff_4=False

        # chi2_pars.t0=False
        # chi2_pars.diff_2=False
        # afo=levmar(afi)
        # print('afo = ', afo)

        chi2_pars.t0=False
        dload_pars.dflag=1
        chi2_pars.diff_2=False


        if chi2_pars.t0_noderivin:
            chi2_pars.t0_noderiv=True
            chi2_pars.t0=False
            dload_pars.dcov=1


        if fit_pars.nnpdf_pos:
            # TODO: there are a few calls to levmar here that need to be treated
            if(inout_pars.pdin):
                chi2_pars.t0=False
                chi2_pars.t0_noderiv=False
            fit_pars.lampos=1e1
            print('lampos = 1e1')
            outputfile=open('outputs/buffer/'+inout_pars.label+'.dat','a')
            outputfile.write("lampos = 1e1")
            outputfile.write(inout_pars.inputnam)
            outputfile.write("\n")
            afo=levmar(afi)
            fit_pars.lampos=1e2
            print('lampos = 1e2')
            outputfile=open('outputs/buffer/'+inout_pars.label+'.dat','a')
            outputfile.write("lampos = 1e2")
            outputfile.write(inout_pars.inputnam)
            outputfile.write("\n")
            afo=levmar(afo)
            fit_pars.lampos=1e3
            print('lampos = 1e3')
            outputfile=open('outputs/buffer/'+inout_pars.label+'.dat','a')
            outputfile.write("lampos = 1e3")
            outputfile.write(inout_pars.inputnam)
            outputfile.write("\n")
            afo=levmar(afo)
            if chi2_pars.t0_noderivin:
                print('lampos = 1e3, t0 full')
                fit_pars.lampos=1e3
                chi2_pars.t0_noderiv=False
                chi2_pars.t0=True
                chi2_pars.diff_2=True
                outputfile=open('outputs/buffer/'+inout_pars.label+'.dat','a')
                outputfile.write("lampos = 1e3")
                outputfile.write(inout_pars.inputnam)
                outputfile.write("\n")
                afo=levmar(afo)
        else:

            # chi2_pars.diff_2=True

            if(inout_pars.pdin):
                chi2_pars.t0=False
                chi2_pars.t0_noderiv=False
                # chi2_pars.t0=True
                # chi2_pars.t0_noderiv=True
            if inout_pars.pd_output:
                chi2_pars.t0=False
                chi2_pars.t0_noderiv=False


            chi2_pars.t0=False
            chi2_pars.t0_noderiv=False
            # chi2_pars.diff_2=True

            # TODO this should open a context manager and pass down the file already opened
            outputfile=open('outputs/buffer/'+inout_pars.label+'.dat','a')
            outputfile.write("diff2=False")
            print("diff2=False")
            outputfile.write(inout_pars.inputnam)
            outputfile.write("\n")
            afo = levmar(afi)

            if chi2_pars.t0_noderivin and not inout_pars.pdin:
                chi2_pars.t0_noderiv=False
                chi2_pars.t0=True
            if inout_pars.pd_output:
                chi2_pars.t0=False
                chi2_pars.t0_noderiv=False

        print('afo = ', afo)

    if(pdf_closure.pdpdf):

        gridout()
        parsout()
        plotout()
        evgrido()

        t1=time.process_time()
        print('time= ',t1-tzero)

    else:

        gridout()
        parsout()
        plotout()
        evgrido()

        fit_pars.pos_const=False
        chi2t0f=chi2min(afo)
        chi2_pars.t0=False
        chi2expf=chi2min(afo)
        fit_pars.pos_const=True
        chi2posf=chi2min(afo)
        pospenf=chi2posf-chi2expf

        print('chi2(exp) in:',chi2expi+pospeni,(chi2expi+pospeni)/chi2_pars.ndat)
        print('chi2(t0) in:',(chi2t0i+pospeni),(chi2t0i+pospeni)/chi2_pars.ndat)
        print('pos penalty in = ',pospeni,pospeni/chi2_pars.ndat)
        print('chi2(exp) out:',chi2expf+pospenf,(chi2expf+pospenf)/chi2_pars.ndat)
        print('chi2(exp) out (no pos pen):',chi2expf,(chi2expf)/chi2_pars.ndat)
        print('chi2(t0) out:',(chi2t0f+pospenf),(chi2t0f+pospenf)/chi2_pars.ndat)
        print('chi2(t0) out (no pos pen):',chi2t0f,(chi2t0f)/chi2_pars.ndat)
        print('pos penalty out = ',pospenf,pospenf/chi2_pars.ndat)


        resout(pospeni,pospenf,chi2t0i,chi2expi,chi2t0f,chi2expf,chi2_pars.ndat)
        t1=time.process_time()
        print('time= ',t1-tzero)

else:
    chi2_pars.t0=False
    fit_pars.pos_const=False
    chi2expi=chi2min(afi)
    chi2_pars.t0=True
    chi2t0i=chi2min(afi)
    ##    fit_pars.pos_const=True
    fit_pars.pos_const=True
    chi2posi=chi2min(afi)
    pospeni=chi2posi-chi2t0i
    print('chi2(exp) in:',chi2expi,chi2expi/chi2_pars.ndat)
    print('chi2(t0) in:',chi2t0i,chi2t0i/chi2_pars.ndat)
    print('pos penalty = ',pospeni)
    res = minimize(chi2min, afi, method='Nelder-Mead', tol=1e-4, options = {'maxiter': 10000, 'maxfev': 10000})
    # res = minimize(chi2min, afi, method='Newton-CG', jac=jac_fun, hess=hess_fun, options = {'disp': True})
#    res = minimize(chi2min, afi, method='CG', jac=jac_fun, options = {'maxiter': 10000})
    print('pars out =',res.x)
    print(chi2min(res.x),chi2min(res.x)/chi2_pars.ndat)
    gridout()
    parsout()
    plotout()
    evgrido()
    fit_pars.pos_const=False
    chi2t0f=chi2min(res.x)
    chi2_pars.t0=False
    chi2expf=chi2min(res.x)
    fit_pars.pos_const=True
    chi2posf=chi2min(res.x)
    pospenf=chi2posf-chi2expf
    print('chi2(exp) in:',chi2expi,chi2expi/chi2_pars.ndat)
    print('chi2(t0) in:',chi2t0i,chi2t0i/chi2_pars.ndat)
    print('pos penalty in = ',pospeni)
    print('chi2(exp)= ',chi2expf,chi2expf/chi2_pars.ndat)
    print('chi2(t0)= ',chi2t0f,chi2t0f/chi2_pars.ndat)
    print('pos penalty= ',pospenf)
    resout(pospeni,pospenf,chi2t0i,chi2expi,chi2t0f,chi2expf,chi2_pars.ndat)
    t1=time.process_time()
    print('time= ',t1-tzero)


# TODO Remove TEMP_LHAPDF at the end
