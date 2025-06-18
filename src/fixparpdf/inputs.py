import numpy as np

from fixparpdf.global_pars import *
from fixparpdf.outputs import BUFFER_F


def readincov():

    basis_pars.n_pars=73
    if basis_pars.Cheb_8:
         basis_pars.n_pars=73+14

    readjac=True

    inputfile='outputs/cov/'+inout_pars.covinput

    # disttot=np.loadtxt(inputfile,skiprows=2,max_rows=64)
    disttot=np.loadtxt(inputfile,skiprows=2,max_rows=73)

    # pdf_pars.parsin=disttot[0:64,0].flatten()
    # pdf_pars.par_isf=disttot[0:64,1].flatten()
    pdf_pars.parsin=disttot[0:73,0].flatten()
    pdf_pars.par_isf=disttot[0:73,1].flatten()

    pdf_pars.npar_free=0
    afin=np.zeros((1))
    pdf_pars.par_free_i=np.zeros((1),dtype=np.int8)
    for i in range(0,len(pdf_pars.parsin)):
        if pdf_pars.par_isf[i]==1:
            afin=np.append(afin,pdf_pars.parsin[i])
            pdf_pars.npar_free+=1
            pdf_pars.par_free_i=np.append(pdf_pars.par_free_i,i)

    afin=np.delete(afin,0)
    pdf_pars.par_free_i=np.delete(pdf_pars.par_free_i,0)

    print(afin)

    # hess=np.loadtxt(inputfile,skiprows=67,max_rows=53)
    # Old input file
    # hess=np.loadtxt(inputfile,skiprows=67,max_rows=pdf_pars.npar_free)
    # New free charm input file
    hess=np.loadtxt(inputfile,skiprows=76,max_rows=pdf_pars.npar_free)

#    hessin=np.loadtxt(inputfile,skiprows=68+pdf_pars.npar_free,max_rows=pdf_pars.npar_free)

    # corrmatcalc(hess,afin)

    if readjac:
        # jac=np.loadtxt(inputfile,skiprows=68+pdf_pars.npar_free,max_rows=pdf_pars.npar_free)
        jac=np.loadtxt(inputfile,skiprows=77+pdf_pars.npar_free,max_rows=pdf_pars.npar_free)
#        jac=np.loadtxt(inputfile,skiprows=68+2*pdf_pars.npar_free,max_rows=pdf_pars.npar_free)
        # jac=np.loadtxt(inputfile,skiprows=68+53,max_rows=53)
    else:
        jac=0.

    return (afin,hess,jac)

def readin():
    """Read the input file inputs::input_file

    Uses ``np.loadtxt`` to read the file a few times once per flavour.
    These parameters are the initial parameters of the fit.
    For each set of parameters, the first column represents the parameter value
    and the second whether it is a free parameter (1) or whether it should be considered fixed (0)
    """

    parfree_def=False

    inputfile='input/'+inout_pars.inputnam

    # distuv=np.loadtxt(inputfile,skiprows=1,max_rows=8)
    nuv=basis_pars.i_uv_max-basis_pars.i_uv_min-1
    distuv=np.loadtxt(inputfile,skiprows=1,max_rows=nuv)

    if parfree_def:
        distuv[:,1]=1

    distuv=np.vstack([[1.0, 0],distuv]) # so numbers match code (calculated norm)
    # distdv=np.loadtxt(inputfile,skiprows=10,max_rows=8)
    ndv=basis_pars.i_dv_max-basis_pars.i_dv_min-1
    distdv=np.loadtxt(inputfile,skiprows=2+nuv,max_rows=ndv)

    if parfree_def:
        distdv[:,1]=1

    distdv=np.vstack([[1.0, 0],distdv])

    if basis_pars.dvd_eq_uvd:
        distdv[1,1]=0.

    # distsea=np.loadtxt(inputfile,skiprows=19,max_rows=9)
    nsea=basis_pars.i_sea_max-basis_pars.i_sea_min
    distsea=np.loadtxt(inputfile,skiprows=3+nuv+ndv,max_rows=nsea)

    if parfree_def:
        distsea[:,1]=1

    # distsp=np.loadtxt(inputfile,skiprows=29,max_rows=9)
    nsp=basis_pars.i_sp_max-basis_pars.i_sp_min
    distsp=np.loadtxt(inputfile,skiprows=4+nuv+ndv+nsea,max_rows=nsp)

    if parfree_def:
        distsp[:,1]=1

    if basis_pars.asp_fix:
        distsp[1,1]=0.

    if basis_pars.t8_int:
        distsp[0,1]=0.

    # distg=np.loadtxt(inputfile,skiprows=39,max_rows=9)
    ng=basis_pars.i_g_max-basis_pars.i_g_min-1
    distg=np.loadtxt(inputfile,skiprows=5+nuv+ndv+nsea+nsp,max_rows=ng)

    if parfree_def:
        distg[:,1]=1

    distg=np.vstack([[1.0, 0],distg])
    # distsm=np.loadtxt(inputfile,skiprows=49,max_rows=3)
    # distsm1=np.loadtxt(inputfile,skiprows=52,max_rows=6)
    nsm=basis_pars.i_sm_max-basis_pars.i_sm_min-1
    distsm=np.loadtxt(inputfile,skiprows=6+nuv+ndv+nsea+nsp+ng,max_rows=3)

    if parfree_def:
        distsm[:,1]=1
        distsm[1,1]=0

    distsm1=np.loadtxt(inputfile,skiprows=9+nuv+ndv+nsea+nsp+ng,max_rows=nsm-3)
    distsm=np.vstack([distsm,[1.0, 0]])
    distsm=np.vstack([distsm,distsm1])
    # distdbub=np.loadtxt(inputfile,skiprows=59,max_rows=8)
    ndbub=basis_pars.i_dbub_max-basis_pars.i_dbub_min
    distdbub=np.loadtxt(inputfile,skiprows=7+nuv+ndv+nsea+nsp+ng+nsm,max_rows=ndbub)

    if parfree_def:
        distdbub[:,1]=1

    # if num_lines==67:
    #     distcharm=np.zeros((9,2))
    # else:
    #     # distcharm=np.loadtxt(inputfile,skiprows=68,max_rows=9)
    #     nchm=basis_pars.i_ch_max-basis_pars.i_ch_min
    #     distcharm=np.loadtxt(inputfile,skiprows=8+nuv+ndv+nsea+nsp+ng+nsm+ndbub,max_rows=nchm)


#   Set charm to zero if p. charm theory!
    nchm=basis_pars.i_ch_max-basis_pars.i_ch_min    
    if fit_pars.theoryidi==211 or fit_pars.theoryidi==212 or fit_pars.theoryidi==40001000 or fit_pars.theoryidi==50001000:
        if basis_pars.Cheb_8:
            distcharm=np.zeros((11,2))
        else:
            distcharm=np.zeros((9,2))
    elif fit_pars.theoryidi==200:
        distcharm=np.loadtxt(inputfile,skiprows=8+nuv+ndv+nsea+nsp+ng+nsm+ndbub,max_rows=nchm)

    ntot=4+nuv+ndv+nsea+nsp+ng+nsm+ndbub+nchm
    basis_pars.n_pars=ntot

    # disttot=np.vstack([distuv,distdv,distsea,distsp,distg,distsm,distdbub])
    disttot=np.vstack([distuv,distdv,distsea,distsp,distg,distsm,distdbub,distcharm])

    if fit_pars.fixpar:
        disttot[:,1]=0.
    

    # pdf_pars.parsin=disttot[0:64,0].flatten()    
    # pdf_pars.par_isf=disttot[0:64,1].flatten()

    # pdf_pars.parsin=disttot[0:73,0].flatten()    
    # pdf_pars.par_isf=disttot[0:73,1].flatten()
    pdf_pars.parsin=disttot[0:ntot,0].flatten()    
    pdf_pars.par_isf=disttot[0:ntot,1].flatten()


    pdf_pars.npar_free=0
    afin=np.zeros((1))
    pdf_pars.par_free_i=np.zeros((1),dtype=np.int8)
    for i in range(0,len(pdf_pars.parsin)):
        if pdf_pars.par_isf[i]==1:
            afin=np.append(afin,pdf_pars.parsin[i])
            pdf_pars.npar_free+=1
            pdf_pars.par_free_i=np.append(pdf_pars.par_free_i,i)

    afin=np.delete(afin,0)
    pdf_pars.par_free_i=np.delete(pdf_pars.par_free_i,0)
    
    outputfile = (BUFFER_F / f"{inout_pars.label}.dat").open("w")
    outputfile.write("Starting new buffer...")
    outputfile.write("\n")

    return afin

def readin_Cheb8():

    inputfile='input/'+inout_pars.inputnam

    with open(inputfile, 'r') as fp:
        x = fp.readlines()
        num_lines = len([l for l in x if l.strip(' \n') != ''])

    distuv=np.loadtxt(inputfile,skiprows=1,max_rows=8)
    distuv=np.vstack([[1.0, 0],distuv]) # so numbers match code (calculated norm)
    distdv=np.loadtxt(inputfile,skiprows=10,max_rows=8)
    distdv=np.vstack([[1.0, 0],distdv])

    if basis_pars.dvd_eq_uvd:
        distdv[1,1]=0.

    distsea=np.loadtxt(inputfile,skiprows=19,max_rows=9)
    distsp=np.loadtxt(inputfile,skiprows=29,max_rows=9)

    if basis_pars.asp_fix:
        distsp[1,1]=0.

    if basis_pars.t8_int:
        distsp[0,1]=0.

    distg=np.loadtxt(inputfile,skiprows=39,max_rows=9)
    distg=np.vstack([[1.0, 0],distg])
    distsm=np.loadtxt(inputfile,skiprows=49,max_rows=3)
    distsm1=np.loadtxt(inputfile,skiprows=52,max_rows=6)
    distsm=np.vstack([distsm,[1.0, 0]])
    distsm=np.vstack([distsm,distsm1])
    distdbub=np.loadtxt(inputfile,skiprows=59,max_rows=8)

    if num_lines==67:
        distcharm=np.zeros((9,2))
    else:
        distcharm=np.loadtxt(inputfile,skiprows=68,max_rows=9)

#   Set charm to zero if p. charm theory!
    if fit_pars.theoryidi==211 or fit_pars.theoryidi==40001000 or fit_pars.theoryidi==50001000:
        distcharm=np.zeros((9,2))

    if fit_pars.theoryidi==212:
        distcharm=np.zeros((9,2))

    # disttot=np.vstack([distuv,distdv,distsea,distsp,distg,distsm,distdbub])
    disttot=np.vstack([distuv,distdv,distsea,distsp,distg,distsm,distdbub,distcharm])

    if fit_pars.fixpar:
        disttot[:,1]=0.
    
    # pdf_pars.parsin=disttot[0:64,0].flatten()    
    # pdf_pars.par_isf=disttot[0:64,1].flatten()

    pdf_pars.parsin=disttot[0:73,0].flatten()    
    pdf_pars.par_isf=disttot[0:73,1].flatten()


    pdf_pars.npar_free=0
    afin=np.zeros((1))
    pdf_pars.par_free_i=np.zeros((1),dtype=np.int8)
    for i in range(0,len(pdf_pars.parsin)):
        if pdf_pars.par_isf[i]==1:
            afin=np.append(afin,pdf_pars.parsin[i])
            pdf_pars.npar_free+=1
            pdf_pars.par_free_i=np.append(pdf_pars.par_free_i,i)

    afin=np.delete(afin,0)
    pdf_pars.par_free_i=np.delete(pdf_pars.par_free_i,0)
    

    outputfile=open('outputs/buffer/'+inout_pars.label+'.dat','w')
    outputfile.write("Starting new buffer...")
    outputfile.write("\n")

    return afin



