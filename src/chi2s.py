from global_pars import *
from validphys.api import API
import scipy.linalg as la
from data_theory import *
from validphys.calcutils import calc_chi2
from validphys.covmats import dataset_inputs_t0_covmat_from_systematics, dataset_inputs_covmat_from_systematics
from pdfs import *
from inputs import *
from lhapdf_funs import *
import time
import scipy.stats as st
from scipy.stats import multivariate_normal

def xgrid_calc():

    xgridtot=[]

    for i in range (0,76):

        output='input/xarr/grid'+str(i)+'.dat'
        inputfile=output
        xcheck=np.loadtxt(inputfile)
        xgridtot=np.append(xgridtot,xcheck)
        xgridtot=np.sort(xgridtot)
      
    xgridtot=np.unique(xgridtot)

    return xgridtot

def af_matcalc(afree):

    afree_mat=np.zeros([len(afree),len(afree)]) 
    for i in range(0,len(afree)):
        for j in range(0,len(afree)):
            afree_mat[i,j]=afree[i]*afree[j]

    return afree_mat

def chi2min_fun(afree,jac_calc,hess_calc):


    err=False
    
    parin=initpars()

    pdfparsii=parset(afree,parin)

    dload_pars.xarr_tot=xgrid_calc()
    
    if not jac_calc and not hess_calc:
        err=parcheck(pdfparsii)
        if(err):
            jac=np.zeros((pdf_pars.npar_free))
            hess=np.zeros((pdf_pars.npar_free,pdf_pars.npar_free))
            hessp=np.zeros((pdf_pars.npar_free,pdf_pars.npar_free))
            out=1e50
            return(out,jac,hess,err,hessp)

    pdf_pars.pdfparsi=sumrules(pdfparsii)

    if fit_pars.deld_const:
        pdf_pars.deld_arr=np.zeros((4*pdf_pars.npar_free+1))
        pdf_pars.delu_arr=np.zeros((4*pdf_pars.npar_free+1))
        pdf_pars.deld_arr[0]=pdf_pars.pdfparsi[10]
        pdf_pars.delu_arr[0]=pdf_pars.pdfparsi[1]


    jac=np.zeros((pdf_pars.npar_free))
    hess=np.zeros((pdf_pars.npar_free,pdf_pars.npar_free))
    hessp=np.zeros((pdf_pars.npar_free,pdf_pars.npar_free))
    pdf_pars.parinarr=np.zeros((4*pdf_pars.npar_free+1,len(pdf_pars.pdfparsi)))
    

    pdflabel_arr=np.empty(pdf_pars.npar_free+1, dtype='U256')
    pdflabel_marr=np.empty(pdf_pars.npar_free+1, dtype='U256')
    if chi2_pars.diff_4:
        pdflabel_m2arr=np.empty(pdf_pars.npar_free+1, dtype='U256')
        pdflabel_p2arr=np.empty(pdf_pars.npar_free+1, dtype='U256')

    if(jac_calc):

        eps_arr=np.zeros((pdf_pars.npar_free+1))
        for ip in range(0,pdf_pars.npar_free+1):

            idir_j=pdf_pars.idir+ip
            name=inout_pars.label+'_run'+str(idir_j)
            # name=inout_pars.label+'_irep'+str(fit_pars.irep)+'_run'+str(idir_j)
            pdflabel_arr[ip]=name

            if ip==0:
                parin1=pdf_pars.pdfparsi.copy()
                pdf_pars.parinarr[0,:]=parin1
            else:
                parin=pdf_pars.pdfparsi.copy() 
                (parin1,eps_arr[ip])=parinc(parin,ip-1,1)
                pdf_pars.parinarr[ip,:]=parin1

                if fit_pars.deld_const:
                    pdf_pars.deld_arr[ip]=parin1[10]
                    pdf_pars.delu_arr[ip]=parin1[1]
                
                
            if(pdf_pars.uselha):
                initlha(name,pdf_pars.lhapdfdir)
                pdf_pars.PDFlabel=name
                writelha(name,pdf_pars.lhapdfdir,parin1)


        if chi2_pars.diff_2 or fit_pars.pos_const:
            for ip in range(1,pdf_pars.npar_free+1):
                idir_j=pdf_pars.idir+pdf_pars.npar_free+ip
                name=inout_pars.label+'_run'+str(idir_j)
                # name=inout_pars.label+'_irep'+str(fit_pars.irep)+'_run'+str(idir_j)
                pdflabel_marr[ip]=name

                if ip==0:
                    parin1=pdf_pars.pdfparsi.copy()
                else:
                    parin=pdf_pars.pdfparsi.copy()
                    (parin1,eps_arr[ip])=parinc(parin,ip-1,2)
                    pdf_pars.parinarr[pdf_pars.npar_free+ip,:]=parin1

                if fit_pars.deld_const:
                    pdf_pars.deld_arr[pdf_pars.npar_free+ip]=parin1[10]
                    pdf_pars.delu_arr[pdf_pars.npar_free+ip]=parin1[1]

                if(pdf_pars.uselha):
                    
                    initlha(name,pdf_pars.lhapdfdir)
                    pdf_pars.PDFlabel=name
                    writelha(name,pdf_pars.lhapdfdir,parin1) 


        if chi2_pars.diff_4:
            for ip in range(1,pdf_pars.npar_free+1):
                idir_j=pdf_pars.idir+2*pdf_pars.npar_free+ip
                name=inout_pars.label+'_run'+str(idir_j)
                # name=inout_pars.label+'_irep'+str(fit_pars.irep)+'_run'+str(idir_j)
                pdflabel_m2arr[ip]=name

                if ip==0:
                    parin1=pdf_pars.pdfparsi.copy()
                else:
                    parin=pdf_pars.pdfparsi.copy()
                    (parin1,eps_arr[ip])=parinc(parin,ip-1,3)
                    pdf_pars.parinarr[pdf_pars.npar_free*2+ip,:]=parin1

                if fit_pars.deld_const:
                    pdf_pars.deld_arr[pdf_pars.npar_free*2+ip]=parin1[10]
                    pdf_pars.delu_arr[pdf_pars.npar_free*2+ip]=parin1[1]

                if(pdf_pars.uselha):

                    initlha(name,pdf_pars.lhapdfdir)
                    pdf_pars.PDFlabel=name
                    writelha(name,pdf_pars.lhapdfdir,parin1)

            for ip in range(1,pdf_pars.npar_free+1):
                idir_j=pdf_pars.idir+3*pdf_pars.npar_free+ip
                name=inout_pars.label+'_run'+str(idir_j)
                # name=inout_pars.label+'_irep'+str(fit_pars.irep)+'_run'+str(idir_j)
                pdflabel_p2arr[ip]=name

                if ip==0:
                    parin1=pdf_pars.pdfparsi.copy()
                else:
                    parin=pdf_pars.pdfparsi.copy()
                    (parin1,eps_arr[ip])=parinc(parin,ip-1,4)
                    pdf_pars.parinarr[pdf_pars.npar_free*3+ip,:]=parin1

                if fit_pars.deld_const:
                    pdf_pars.deld_arr[pdf_pars.npar_free*3+ip]=parin1[10]
                    pdf_pars.delu_arr[pdf_pars.npar_free*3+ip]=parin1[1]

                if(pdf_pars.uselha):
                    
                    initlha(name,pdf_pars.lhapdfdir)
                    pdf_pars.PDFlabel=name
                    writelha(name,pdf_pars.lhapdfdir,parin1)


        if chi2_pars.diff_4:
            (out0,out1,jac,hessd4)=jaccalc_d4(pdflabel_arr,pdflabel_marr,pdflabel_m2arr,pdflabel_p2arr,eps_arr,hess_calc,fit_pars.imindat,fit_pars.imaxdat)
            out=out0+out1
            hess=hessd4.copy()


        elif chi2_pars.diff_2:        
            (out0,out1,jac,hessd2)=jaccalc_d2(pdflabel_arr,pdflabel_marr,eps_arr,hess_calc,fit_pars.imindat,fit_pars.imaxdat)
            out=out0+out1
            hess=hessd2.copy()

        else:
            print('JACCALC')
            (jac,hess,out0,out1,hessp)=jaccalc(pdflabel_arr,pdflabel_marr,eps_arr,hess_calc)
            out=out0+out1
            

        if(pdf_pars.uselha):
            for ip in range(0,pdf_pars.npar_free+1):
                idir_j=pdf_pars.idir+ip
                name=inout_pars.label+'_run'+str(idir_j)
                # name=inout_pars.label+'_irep'+str(fit_pars.irep)+'_run'+str(idir_j)
                dellha(name)
                
            pdf_pars.idir+=pdf_pars.npar_free+1

        if chi2_pars.diff_2 or fit_pars.pos_const:
            if(pdf_pars.uselha):
                for ip in range(0,pdf_pars.npar_free):
                    idir_j=pdf_pars.idir+ip
                    name=inout_pars.label+'_run'+str(idir_j)
                    # name=inout_pars.label+'_irep'+str(fit_pars.irep)+'_run'+str(idir_j)
                    dellha(name)

                pdf_pars.idir+=pdf_pars.npar_free
        
        if chi2_pars.diff_4:

            if(pdf_pars.uselha):
                
                for ip in range(0,pdf_pars.npar_free):
                    idir_j=pdf_pars.idir+ip
                    name=inout_pars.label+'_run'+str(idir_j)
                    # name=inout_pars.label+'_irep'+str(fit_pars.irep)+'_run'+str(idir_j)
                    dellha(name)
                    
                pdf_pars.idir+=pdf_pars.npar_free
                for ip in range(0,pdf_pars.npar_free):
                    idir_j=pdf_pars.idir+ip
                    name=inout_pars.label+'_run'+str(idir_j)
                    # name=inout_pars.label+'_irep'+str(fit_pars.irep)+'_run'+str(idir_j)
                    dellha(name)

                pdf_pars.idir+=pdf_pars.npar_free
    else:

        name=inout_pars.label+'_run'+str(pdf_pars.idir) 
        # name=inout_pars.label+'_irep'+str(fit_pars.irep)+'_run'+str(pdf_pars.idir)


        parin1=pdf_pars.pdfparsi.copy() 

        pdf_pars.parinarr[0,:]=parin1
        pdf_pars.iPDF=0

        # write to temp lhapdf grid to be used by nnpdf code
        if(pdf_pars.uselha):
            initlha(name,pdf_pars.lhapdfdir)
            pdf_pars.PDFlabel=name
            writelha(name,pdf_pars.lhapdfdir,parin1)    
        # calculate chi2
        chi=chi2totcalc()
        out=chi[0]+chi[1] # exp + positivity
        out0=chi[0]
        if(pdf_pars.uselha):
            dellha(name) # delete temporary folder
        pdf_pars.idir+=1 # iterate up so new folder

        print('chi2/N_dat=',out/chi2_pars.ndat)                                                                                             
        print('chi2tot (no pos)=',out0)                                                                                       
        print('pos pen =',chi[1])
        print('chi2tot (no pos)/N_dat=',out0/chi2_pars.ndat)  
        
    return(out,jac,hess,err,hessp)

def chi2corr(imin,imax):

    if(pdf_closure.pdpdf):
        (out,theorytot,cov,covin)=chi2corr_pdf()
    else:
        (out,theorytot,cov,covin)=chi2corr_global(imin,imax)

    return (out,theorytot,cov,covin)
    

def chi2corr_pdf():

    L0_old=False

    npdf=8
    nx=500
#    nx=120

    chi2_pars.ndat=nx*npdf

    chi2_pars.ndat=nx

    xmin=1e-5
    xmax=0.99

    # xmax=0.8
    # xmin=1e-3

    if L0_old:
        nx=400
        inputfile='outputs/pseudodata/'+pdf_closure.pdlabel+'.dat'
        distin=np.loadtxt(inputfile)
        xtot=distin[0:nx,0].flatten()


        # xmin=1e-3
        # xmax=0.8

    lxmin=np.log(xmin)
    lxmax=np.log(xmax)
    delerr=0.01

    if fit_pars.theoryidi==211 or fit_pars.theoryidi==40001000:
        qin=1.
    elif fit_pars.theoryidi==200:
        qin=1.65


    parr=pdf_pars.parinarr[pdf_pars.iPDF].copy()
    indices=[3,4,5,6,7,8]
    chebsum=np.sum(parr[indices])


    theorytot=np.zeros((1))
    if(inout_pars.pdout):
        errtot=np.zeros((1))
        truetot=np.zeros((1))
        xtot=np.zeros((1))

    if(pdf_pars.uselha):
        pset = lhapdf.getPDFSet(pdf_pars.PDFlabel)
        pdfs = pset.mkPDFs()

        # for ip in range(0,npdf):
    for ip in range(0,1):
        for ix in range(0,nx):
            lx=lxmin+(lxmax-lxmin)*(ix-1)/nx
            x=np.exp(lx)

            if L0_old:
                x=xtot[ix]
            
            if(inout_pars.pdout):
                if(pdf_pars.uselha):
                    if ip == 0:
                        gluon=pdfs[0].xfxQ(0,x,qin)
                    elif ip == 1:
                        gluon=pdfs[0].xfxQ(2,x,qin)-pdfs[0].xfxQ(-2,x,qin)
                    elif ip == 2:
                        gluon=pdfs[0].xfxQ(1,x,qin)-pdfs[0].xfxQ(-1,x,qin)
                    elif ip == 3:
                        gluon=2.*(pdfs[0].xfxQ(-1,x,qin)+pdfs[0].xfxQ(-2,x,qin))
                        gluon=gluon+pdfs[0].xfxQ(3,x,qin)+pdfs[0].xfxQ(-3,x,qin)
                    elif ip == 4:
                        gluon=pdfs[0].xfxQ(3,x,qin)+pdfs[0].xfxQ(-3,x,qin)
                    elif ip == 5:
                        gluon=pdfs[0].xfxQ(-1,x,qin)/pdfs[0].xfxQ(-2,x,qin)
                    elif ip == 6:
                        gluon=pdfs[0].xfxQ(3,x,qin)-pdfs[0].xfxQ(-3,x,qin)
                    elif ip == 7:
                        gluon=pdfs[0].xfxQ(4,x,qin)+pdfs[0].xfxQ(-4,x,qin)
                else:
                    if ip == 0:
                        gluon=pdfs_msht(0,pdf_pars.pdfparsi,x)
                    elif ip == 1:
                        gluon=pdfs_msht(2,pdf_pars.pdfparsi,x)-pdfs_msht(-2,pdf_pars.pdfparsi,x)
                    elif ip == 2:
                        gluon=pdfs_msht(1,pdf_pars.pdfparsi,x)-pdfs_msht(-1,pdf_pars.pdfparsi,x)
                    elif ip == 3:
                        gluon=2.*(pdfs_msht(-1,pdf_pars.pdfparsi,x)+pdfs_msht(-2,pdf_pars.pdfparsi,x))
                        gluon=gluon+pdfs_msht(3,pdf_pars.pdfparsi,x)+pdfs_msht(-3,pdf_pars.pdfparsi,x)
                    elif ip == 4:
                        gluon=pdfs_msht(3,pdf_pars.pdfparsi,x)+pdfs_msht(-3,pdf_pars.pdfparsi,x)
                    elif ip == 5:
                        gluon=pdfs_msht(-1,pdf_pars.pdfparsi,x)/pdfs_msht(-2,pdf_pars.pdfparsi,x)
                    elif ip == 6:
                        gluon=pdfs_msht(3,pdf_pars.pdfparsi,x)-pdfs_msht(-3,pdf_pars.pdfparsi,x)
                    elif ip == 7:
                        gluon=pdfs_msht(4,pdf_pars.pdfparsi,x)+pdfs_msht(-4,pdf_pars.pdfparsi,x)
            else:
                if(pdf_pars.uselha):
                    if ip == 0:
                        gluon=pdfs[0].xfxQ(0,x,qin)
                    elif ip == 1:
                        gluon=pdfs[0].xfxQ(2,x,qin)-pdfs[0].xfxQ(-2,x,qin)
                    elif ip == 2:
                        gluon=pdfs[0].xfxQ(1,x,qin)-pdfs[0].xfxQ(-1,x,qin)
                    elif ip == 3:
                        gluon=2.*(pdfs[0].xfxQ(-1,x,qin)+pdfs[0].xfxQ(-2,x,qin))
                        gluon=gluon+pdfs[0].xfxQ(3,x,qin)+pdfs[0].xfxQ(-3,x,qin)
                    elif ip == 4:
                        gluon=pdfs[0].xfxQ(3,x,qin)+pdfs[0].xfxQ(-3,x,qin)
                    elif ip == 5:
                        gluon=pdfs[0].xfxQ(-1,x,qin)/pdfs[0].xfxQ(-2,x,qin)
                    elif ip == 6:
                        gluon=pdfs[0].xfxQ(3,x,qin)-pdfs[0].xfxQ(-3,x,qin)
                    elif ip == 7:
                        gluon=pdfs[0].xfxQ(4,x,qin)+pdfs[0].xfxQ(-4,x,qin)
                else:
                    if ip == 0:
                        gluon=pdfs_msht(0,pdf_pars.parinarr[pdf_pars.iPDF],x)
                    elif ip == 1:
                        gluon=pdfs_msht(2,pdf_pars.parinarr[pdf_pars.iPDF],x)-pdfs_msht(-2,pdf_pars.parinarr[pdf_pars.iPDF],x)
                    elif ip == 2:
                        gluon=pdfs_msht(1,pdf_pars.parinarr[pdf_pars.iPDF],x)-pdfs_msht(-1,pdf_pars.parinarr[pdf_pars.iPDF],x)
                    elif ip == 3:
                        gluon=2.*(pdfs_msht(-1,pdf_pars.parinarr[pdf_pars.iPDF],x)+pdfs_msht(-2,pdf_pars.parinarr[pdf_pars.iPDF],x))
                        gluon=gluon+pdfs_msht(3,pdf_pars.parinarr[pdf_pars.iPDF],x)+pdfs_msht(-3,pdf_pars.parinarr[pdf_pars.iPDF],x)
                    elif ip == 4:
                        gluon=pdfs_msht(3,pdf_pars.parinarr[pdf_pars.iPDF],x)+pdfs_msht(-3,pdf_pars.parinarr[pdf_pars.iPDF],x)
                    elif ip == 5:
                        gluon=pdfs_msht(-1,pdf_pars.parinarr[pdf_pars.iPDF],x)/pdfs_msht(-2,pdf_pars.parinarr[pdf_pars.iPDF],x)
                    elif ip == 6:
                        gluon=pdfs_msht(3,pdf_pars.parinarr[pdf_pars.iPDF],x)-pdfs_msht(-3,pdf_pars.parinarr[pdf_pars.iPDF],x)
                    elif ip == 7:
                        gluon=pdfs_msht(4,pdf_pars.parinarr[pdf_pars.iPDF],x)+pdfs_msht(-4,pdf_pars.parinarr[pdf_pars.iPDF],x)

            if(inout_pars.pdout):
                error=np.abs(gluon*delerr)
                if error < 1e-5:
                    error=1e-5
                errtot=np.append(errtot,error)

                xtot=np.append(xtot,x)

                if(pdf_closure.pdfscat):
                    gluono=gluon
                    gluon=gluon+np.random.normal()*error
                    truetot=np.append(truetot,gluono)
                else:
                    truetot=np.append(truetot,gluon)

            theorytot=np.append(theorytot,gluon)

    theorytot=np.delete(theorytot,0)

    if(inout_pars.pdout):
        errtot=np.delete(errtot,0)
        truetot=np.delete(truetot,0)
        xtot=np.delete(xtot,0)

    if(inout_pars.pdout):
        outputfile=open('outputs/pseudodata/'+pdf_closure.pdlabel+'.dat','w')
        for i in range (0,len(theorytot)):
            outputfile.write(str(theorytot[i]))
            outputfile.write(' ')
            outputfile.write(str(errtot[i]))
            outputfile.write(' ')
            outputfile.write(str(truetot[i]))
            outputfile.write(' ')
            outputfile.write(str(xtot[i]))
            outputfile.write('\n')

    if inout_pars.pdin:
        if dload_pars.dflag==1:

            if L0_old:
                nx=400
                inputfile='outputs/pseudodata/'+pdf_closure.pdlabel+'.dat'
                distin=np.loadtxt(inputfile)
                xtot=distin[0:nx,0].flatten()
                dattot=distin[0:nx,1].flatten()
                truetot=dattot
                errtot=dattot*delerr
                # print(errtot)
                # exit()
                # for i in range(0,len(errtot)):
                #     error=errtot[i]
                #     if error < 1e-4:
                #         errtot[i]=1e-4
            else:
                inputfile='outputs/pseudodata/'+pdf_closure.pdlabel+'.dat'
                print('testdir =',pdf_closure.pdlabel)
                distin=np.loadtxt(inputfile)
                dattot=distin[0:len(theorytot),0].flatten()
                errtot=distin[0:len(theorytot),1].flatten()
                truetot=distin[0:len(theorytot),2].flatten()
                xtot=distin[0:len(theorytot),3].flatten()


            for i in range(0,len(theorytot)): 
                
                if np.abs(errtot[i]) < 1e-4:
                    errtot[i]=1e-4

            dload_pars.darr_gl=dattot
            dload_pars.err_gl=errtot
            dload_pars.true_gl=truetot
            dload_pars.x_gl=xtot
            dload_pars.dflag=0
        else:
            dattot=dload_pars.darr_gl
            errtot=dload_pars.err_gl
            truetot=dload_pars.true_gl
            xtot=dload_pars.x_gl
        
    cov=np.zeros((len(theorytot),len(theorytot)))
    for i in range (0,len(theorytot)):
        cov[i,i]=np.power(errtot[i],2)

    if(inout_pars.pdin):

        if L0_old:
            for i in range(0,len(theorytot)):
                if xtot[i] > 0.5 or xtot[i] < 1e-3:
                    theorytot[i]=dattot[i]

        diffs=theorytot-dattot
        out=calc_chi2(la.cholesky(cov, lower=True), diffs)
        diffs_true=np.abs((theorytot-truetot)/truetot)
        out_true=np.sum(diffs_true)/chi2_pars.ndat

    else:
        diffs=0.
        out=0.


    if pdf_pars.iPDF == 0 and inout_pars.pdin:
#        print('wtcheb= ',wtcheb)
        print('av diff =',out_true)
        # print((theorytot-truetot)/truetot)
        outputfile=open('outputs/pseudodata/pdfcomp/'+pdf_closure.pdlabel+'.dat','w')
        for i in range (0,len(theorytot)):
            outputfile.write(str(xtot[i]))
            outputfile.write(' ')
            outputfile.write(str(theorytot[i]))
            outputfile.write(' ')
            outputfile.write(str(truetot[i]))
            outputfile.write(' ')
            outputfile.write(str((theorytot[i]-truetot[i])/truetot[i]*100.))
            outputfile.write('\n')


    covin=la.inv(cov)

    return (out,theorytot,cov,covin)

def chi2corr_ind_plot(imin,imax):


    for i in range(imin,imax+1):
        dataset_testii=fit_pars.dataset_40[i] 
        # print(dataset_testii)

    if(chi2_pars.t0):
        cov_gl=dload_pars.covt0
        covin_gl=dload_pars.covt0_inv
    else:
        cov_gl=dload_pars.covexp
        covin_gl=dload_pars.covexp_inv

    cov_ind=cov_gl[chi2_pars.idat_low_arr[imin]:chi2_pars.idat_up_arr[imax],chi2_pars.idat_low_arr[imin]:chi2_pars.idat_up_arr[imax]]
    cov=cov_ind
    cov_inv=la.inv(cov)

    dattot=dload_pars.darr_gl[chi2_pars.idat_low_arr[imin]:chi2_pars.idat_up_arr[imax]]

    output='BCDMSP_dwsh_dat.dat'
    output='outputs/chi2ind_group/'+output
    with open(output,'w') as outputfile:

        for i in range(0,len(dattot)): 
            print(dattot[i],np.sqrt(cov_ind[i,i]))
            L=[str(f'{dattot[i]:10}'),' ',str(f'{np.sqrt(cov_ind[i,i]):6}')]
            outputfile.writelines(L)
            outputfile.write('\n')

    exit()

def chi2corr_ind_group(imin,imax):


    for i in range(imin,imax+1):
        dataset_testii=fit_pars.dataset_40[i] 
        # print(dataset_testii)

    if(chi2_pars.t0):
        cov_gl=dload_pars.covt0
        covin_gl=dload_pars.covt0_inv
    else:
        cov_gl=dload_pars.covexp
        covin_gl=dload_pars.covexp_inv

    cov_ind=cov_gl[chi2_pars.idat_low_arr[imin]:chi2_pars.idat_up_arr[imax],chi2_pars.idat_low_arr[imin]:chi2_pars.idat_up_arr[imax]]
    cov=cov_ind
    cov_inv=la.inv(cov)

    theory=dload_pars.tharr_gl[chi2_pars.idat_low_arr[imin]:chi2_pars.idat_up_arr[imax]]
    dattot=dload_pars.darr_gl[chi2_pars.idat_low_arr[imin]:chi2_pars.idat_up_arr[imax]]


    chi2_pars.ndat=len(dattot)

    diffs=np.array(dattot-theory)
    out=diffs@cov_inv@diffs

    # diffs_diag=np.power(diffs,2)/np.diag(cov)
    # print('diag=',sum(diffs_diag),len(diffs_diag))
    # print(dattot,theory,np.sqrt(np.diag(cov)))

    return (out,chi2_pars.ndat)

def chi2corr_lab(i):

    dscomb=[fit_pars.dataset_40[i]]
    dataset_testii=fit_pars.dataset_40[i]                                                                                

    inptt = {                                                                                                                 
        "dataset_input": dataset_testii,                                                                                      
        "use_cuts": "internal",                                                                                               
        "theoryid": fit_pars.theoryidi,                                                                                    
    } 

    dlabel=API.dataset(**inptt)

    return dlabel

def chi2corr_ind(i):


    dscomb=[fit_pars.dataset_40[i]]
    dataset_testii=fit_pars.dataset_40[i]                                                                                

    inptt = {                                                                                                                 
        "dataset_input": dataset_testii,                                                                                      
        "use_cuts": "internal",                                                                                               
        "theoryid": fit_pars.theoryidi,                                                                                    
    } 

    dlabel=API.dataset(**inptt)

    # theory=theory_calc(i,dataset_testii,inptt,fit_pars.cftrue[i])

    # dattot=dat_calc_rep(dscomb,cov)
    # lcd = API.loaded_commondata_with_cuts(dataset_input=dataset_testii,theoryid=fit_pars.theoryidi,use_cuts='internal')
    # cval=lcd.central_values

    if(chi2_pars.t0):
        # inpt0 = dict(dataset_inputs=dscomb, theoryid=fit_pars.theoryidi, use_cuts="internal", t0pdfset=pdf_pars.PDFlabel, use_t0=True)
        # cov = API.dataset_inputs_t0_covmat_from_systematics(**inpt0)
        cov_gl=dload_pars.covt0
        covin_gl=dload_pars.covt0_inv
    else:
        # inp = dict(dataset_inputs=dscomb, theoryid=fit_pars.theoryidi, use_cuts="internal")
        # cov = API.dataset_inputs_covmat_from_systematics(**inp)
        cov_gl=dload_pars.covexp
        covin_gl=dload_pars.covexp_inv
        

    cov_ind=cov_gl[chi2_pars.idat_low_arr[i]:chi2_pars.idat_up_arr[i],chi2_pars.idat_low_arr[i]:chi2_pars.idat_up_arr[i]]
    cov=cov_ind
    cov_inv=la.inv(cov)
    # cov_inv=covin_gl[chi2_pars.idat_low_arr[i]:chi2_pars.idat_up_arr[i],chi2_pars.idat_low_arr[i]:chi2_pars.idat_up_arr[i]]


    theory=dload_pars.tharr_gl[chi2_pars.idat_low_arr[i]:chi2_pars.idat_up_arr[i]]
    # dattot=dat_calc_rep(dscomb,cov)
    dattot=dload_pars.darr_gl[chi2_pars.idat_low_arr[i]:chi2_pars.idat_up_arr[i]]


    chi2_pars.ndat=len(dattot)

    diffs=np.array(dattot-theory)
    # out=calc_chi2(la.cholesky(cov, lower=True), diffs)
    out=diffs@cov_inv@diffs

    return (out,chi2_pars.ndat,dlabel)

def chi2corr_global(imin,imax):

    if fit_pars.nlo_cuts:
        # intersection=[{"dataset_inputs": dload_pars.dscomb, "theoryid": 212}]
        intersection=[{"dataset_inputs": dload_pars.dscomb, "theoryid": 200}]


    din=[fit_pars.dataset_40[imin]]

    # print('CHI2CORR')


    if dload_pars.dflag==1:
        # print('DLOAD')
        for i in range(imin+1,imax+1):
            din.append(fit_pars.dataset_40[i])
            # print(i,fit_pars.dataset_40[i])
        dload_pars.dscomb=din
        
    
    dload_pars.ifk=0
    if dload_pars.dflag==1:
        chi2_pars.idat=0

    for i in range(imin,imax+1):
        dataset_testii=fit_pars.dataset_40[i]  
        print(fit_pars.dataset_40[i])
        inptt = {                                                                                                                 
                "dataset_input": dataset_testii,                                                                                      
                "use_cuts": "internal",                                                                                               
                "theoryid": fit_pars.theoryidi,                                                                         
            }  

               
        theory=theory_calc(i,dataset_testii,inptt,fit_pars.cftrue[i])
        fit_pars.preds_stored[str(dataset_testii["dataset"])]=theory

        if dload_pars.dflag==1:
            chi2_pars.idat_low_arr[i]=chi2_pars.idat
            chi2_pars.idat+=len(theory)
            chi2_pars.idat_up_arr[i]=chi2_pars.idat

        # print('theory1 - ', i)
        if i==imin:
            theorytot=theory
        else:
            theorytoti=np.concatenate((theorytot,theory))
            theorytot=theorytoti
    
    if(chi2_pars.t0):
        print('t0 cov1...')
        if fit_pars.nlo_cuts:
            inpt0 = dict(dataset_inputs=dload_pars.dscomb, theoryid=fit_pars.theoryidi, use_cuts="fromintersection", cuts_intersection_spec=intersection, t0pdfset=pdf_pars.PDFlabel, use_t0=True)
        else:   
            inpt0 = dict(dataset_inputs=dload_pars.dscomb, theoryid=fit_pars.theoryidi, use_cuts="internal", t0pdfset=pdf_pars.PDFlabel, use_t0=True)
        try:
            cov = calc_covmat_t0(inpt0)
            covin=la.inv(cov)
        except la.LinAlgError as err:
            print('t0 cov may be ill behaved, trying exp cov instead...')
            cov=dload_pars.covexp
            covin=dload_pars.covexp_inv
        dload_pars.covt0=cov
        dload_pars.covt0_inv=covin
        print('...finish')
    else:
        if chi2_pars.t0_noderiv:
            if dload_pars.dcov==1:
                print('t0 cov2...')
                if fit_pars.nlo_cuts:
                    inpt0 = dict(dataset_inputs=dload_pars.dscomb, theoryid=fit_pars.theoryidi, use_cuts="fromintersection", cuts_intersection_spec=intersection, t0pdfset=pdf_pars.PDFlabel, use_t0=True)
                else:   
                    inpt0 = dict(dataset_inputs=dload_pars.dscomb, theoryid=fit_pars.theoryidi, use_cuts="internal", t0pdfset=pdf_pars.PDFlabel, use_t0=True)
                try:
                    cov = calc_covmat_t0(inpt0)
                    covin=la.inv(cov)
                except la.LinAlgError as err:
                    print('t0 cov may be ill behaved, trying exp cov instead...')
                    cov=dload_pars.covexp
                    covin=dload_pars.covexp_inv
                dload_pars.covt0=cov
                dload_pars.covt0_inv=covin
                print('...finish')
                dload_pars.dcov=0
        else:
            if dload_pars.dflag==1:
                print('exp cov...')
                if fit_pars.nlo_cuts:
                    print('NLO CUTS')
                    inp = dict(dataset_inputs=dload_pars.dscomb, theoryid=fit_pars.theoryidi, use_cuts="fromintersection", cuts_intersection_spec=intersection)
                else:   
                    inp = dict(dataset_inputs=dload_pars.dscomb, theoryid=fit_pars.theoryidi, use_cuts="internal")
                cov = API.dataset_inputs_covmat_from_systematics(**inp) 
                covin=la.inv(cov)
                dload_pars.covexp=cov
                dload_pars.covexp_inv=covin
                print('...finish')

    if dload_pars.dflag==1:

        print('DLOAD')

        if fit_pars.pseud:
            fit_pars.pseud=False
            dattot0=dat_calc_rep(dload_pars.dscomb,cov)
            fit_pars.pseud=True
            dattot=dat_calc_rep(dload_pars.dscomb,cov)
        else:  
            dattot=dat_calc_rep(dload_pars.dscomb,cov)
        chi2_pars.ndat=len(dattot)
        dload_pars.darr_gl=dattot

    else:

        print('NO DLOAD')

        dattot=dload_pars.darr_gl
        if chi2_pars.t0_noderiv:
            cov=dload_pars.covt0
            covin=dload_pars.covt0_inv
        elif not chi2_pars.t0:
            cov=dload_pars.covexp
            covin=dload_pars.covexp_inv

    if inout_pars.pdout:
        outputfile=open('outputs/pseudodata/'+pdf_closure.pdlabel+'.dat','w')


#       pseudodata calculate using internal NNPDF routing - shift defined wrt real data so redefine as wrt pseudodata
        if fit_pars.pseud: 
            p_data=theorytot+dattot-dattot0
            for i in range (0,len(dattot)):
                outputfile.write(str(p_data[i]))
                outputfile.write('\n')
        # calculated using own routine
        else:
            for i in range (0,len(theorytot)):
                outputfile.write(str(theorytot[i]))
                outputfile.write('\n')
            
    if dload_pars.dflag==1:
        if inout_pars.pdin:
            inputfile='outputs/pseudodata/'+pdf_closure.pdlabel+'.dat'
            print('testdir =',pdf_closure.pdlabel)
            distin=np.loadtxt(inputfile)
            dattot=distin
            dload_pars.darr_gl=dattot
            # dload_pars.dflag=0

            if(pdf_closure.pdfscat):

                covin=la.inv(cov)
                lam,eig = la.eigh(covin)
                cov_d=la.inv(eig)@covin@eig
                
                dattot_d=la.inv(eig)@dattot            
                
                for i in range(0,len(cov)):
                    dattot_d[i]=dattot_d[i]+np.random.normal()*np.sqrt(1./cov_d[i,i])
                
                dattot=eig@dattot_d
            
                dload_pars.darr_gl=dattot
                
        dload_pars.dflag=0
 
    else:
        dattot=dload_pars.darr_gl

    chi2_pars.ndat=len(dattot)   

    diffs=dattot-theorytot

    dload_pars.tharr_gl=theorytot

    dattotr=2.*dattot-1.
    theorytotr=2.*theorytot-1.

    if fit_pars.nmcpd_diag:

        ndat_nmcpd=121
        cov_d=cov.copy()

        for i in range(0,ndat_nmcpd):
            for j in range(0,ndat_nmcpd):
                if i==j:
                    cov_d[i,j]=cov[i,j]
                else:
                    cov_d[i,j]=0.

        cov=cov_d.copy()

    try:
        out=calc_chi2(la.cholesky(cov, lower=True), diffs)
    except la.LinAlgError as err:
        print(err)
        print('t0 cov may be ill behaved, trying exp cov instead...')
        inp = dict(dataset_inputs=dload_pars.dscomb, theoryid=fit_pars.theoryidi, use_cuts="internal")
        cov = API.dataset_inputs_covmat_from_systematics(**inp)
        try:
            out=calc_chi2(la.cholesky(cov, lower=True), diffs)
            print('out=', out)
            print('cov=',cov)
            print('diffs=',diffs)
            print('dattot =',dattot)
            print('theorytot =',theorytot)
        except la.LinAlgError as erra:
            print(erra)
            print('No, theory ill behaved - set chi^2=1e50')
            out=1e50
            print('out=', out)
    except ValueError as err1:
        print(err1)
        print('t0 cov ill behaved, trying exp cov instead...')
        inp = dict(dataset_inputs=dload_pars.dscomb, theoryid=fit_pars.theoryidi, use_cuts="internal")
        cov = API.dataset_inputs_covmat_from_systematics(**inp)
        try:
            out=calc_chi2(la.cholesky(cov, lower=True), diffs)
            print('out=', out)
            print('cov=',cov)
            print('diffs=',diffs)
            print('dattot =',dattot)
            print('theorytot =',theorytot)
        except ValueError as err1a:
            print(err1a)
            print('No, theory ill behaved - set chi^2=1e50')
            out=1e50
            print('out=', out)

    return (out,theorytot,cov,covin)


def hess_fun(afree):

    hess_calc=True
    jac_calc=True
    out=chi2min_fun(afree,jac_calc,hess_calc)[2]
    print('Hessian = ',out)
    return out

def jac_fun(afree):

    hess_calc=False
    jac_calc=True
    out=chi2min_fun(afree,jac_calc,hess_calc)[1]
    print('Jacobian = ',out)
    return out

def chi2min(afree):


    hess_calc=False
    jac_calc=False
    outarr=chi2min_fun(afree,jac_calc,hess_calc)
    out=outarr[0]
    return out
    
def chilim_calc(nd):

    n=np.rint(nd)

    frac=1E-3

    chisq=0.
    sum=0.
    i=0
    cl50=0.
    cl68=0.

    while cl68==0.:
        chisq+=frac*n
        sum+=st.chi2.pdf(chisq,nd)*frac*n
        # print(sum,chisq)
        if sum > 0.5 and i==0:
            cl50=chisq
            i=1
        if sum > 0.68:
            cl68=chisq
        

    return (cl50,cl68)

def chilim_sort():

    output='test.dat'
    output='outputs/chi2ind_group/'+inout_pars.label+'.dat'
    with open(output,'w') as outputfile:

        for i in range(0,len(chi2_pars.chi_ind_arr)): 
            print(i,chi2_pars.clnd_arr[i],chi2_pars.cldataset_arr[i],chi2_pars.chi_ind_arr[i])
            L=[str(f'{i:2}'),' ',str(f'{chi2_pars.clnd_arr[i]:4}'),' ',str(f'{chi2_pars.cldataset_arr[i]:45}'),' ',str(f'{chi2_pars.chi_ind_arr[i]:5}')]
            outputfile.writelines(L)
            outputfile.write('\n')
            # print(chi2_pars.chi0_ind_arr[i])

    print(sum(chi2_pars.chi_ind_arr))


    # exit()

def chilim_fill(nd,chi,dlab):

    (cl50,cl68)=chilim_calc(nd)

    if chi2_pars.L0:
        chilim=cl68-cl50
    else:   
        chilim=(cl68/cl50-1.)*chi
    # print(chi,chilim)

    chi2_pars.chilim_arr.append(chilim)
    chi2_pars.cldataset_arr.append(dlab.name)
    chi2_pars.clnd_arr.append(nd)
    chi2_pars.chi0_ind_arr.append(chi)
    # print(i,chilim/chi,cl68,cl50,cl68/cl50,nd)

    

def chi2totcalc():
    
    chiarr=np.zeros(fit_pars.imaxdat-fit_pars.imindat)
    chiarr[0]=chi2corr(fit_pars.imindat,fit_pars.imaxdat-1)[0]

    ndtot=0
    chi2totind=0.
    if chi2_pars.chi2ind:
        chi2_pars.chi_ind_arr=[]
        chi2_pars.idat=0.

        for i in range(fit_pars.imindat,fit_pars.imaxdat): 

            (chi,nd,dlab)=chi2corr_ind(i)
            # print(dlab,chi)

# LHL !!! Pretty sure this - for dynamic tolerance - will only work for old version (4.0.6.1188) of code - to update

            if fit_pars.dynT_group:
                chi2ind_calc=True
                

                if str(dlab) in ['CMSTTBARTOT7TEV','ATLAS_TTB_DIFF_8TEV_LJ_TRAPNORM','ATLASTTBARTOT7TEV','ATLASZPT8TEVMDIST','HERACOMBNCEM','HERACOMB_SIGMARED_C','ATLASWZRAP11CC','ATLAS_WP_JET_8TEV_PT']:

                    if str(dlab)=='HERACOMBNCEM':
                        dlab.name='HERA inclusive'
                        (chi,nd)=chi2corr_ind_group(i,i+6)
                    elif str(dlab)=='HERACOMB_SIGMARED_C':
                        dlab.name='HERA Heavy Flavour'
                        # (chii,ndi)=chi2corr_ind_group(i,i)
                        # chi2corr_ind_plot(i,i)
                        (chi,nd)=chi2corr_ind_group(i,i+1)
                    elif str(dlab)=='ATLASWZRAP11CC':
                        dlab.name='ATLASWZRAP11 (CC+CF)'
                        (chi,nd)=chi2corr_ind_group(i,i+1)
                    elif str(dlab)=='ATLAS_WP_JET_8TEV_PT':
                        dlab.name='ATLAS_WPM_JET_8TEV_PT'
                        (chi,nd)=chi2corr_ind_group(i,i+1)
                    elif str(dlab)=='ATLASZPT8TEVMDIST':
                        dlab.name='ATLASZPT8TEV'
                        (chi,nd)=chi2corr_ind_group(i,i+1)
                    elif str(dlab)=='ATLASTTBARTOT7TEV':
                        dlab.name='ATLASTTBARTOT'
                        (chi,nd)=chi2corr_ind_group(i,i+2)
                    elif str(dlab)=='ATLAS_TTB_DIFF_8TEV_LJ_TRAPNORM':
                        dlab.name='ATLAS_TTB_DIFF_8TEV_LJ'
                        (chi,nd)=chi2corr_ind_group(i,i+1)
                    elif str(dlab)=='CMSTTBARTOT7TEV':
                        dlab.name='CMSTTBARTOT'
                        (chi,nd)=chi2corr_ind_group(i,i+2)

                    chi2ind_calc=False
                    if fit_pars.dynT_ngt5:
                        if nd > 4:
                            chi2_pars.chi_ind_arr.append(chi)
                            chi2totind+=chi
                            ndtot+=nd
                            if chi2_pars.calc_cl:
                                 chilim_fill(nd,chi,dlab)
                    else:
                        chi2_pars.chi_ind_arr.append(chi)
                        chi2totind+=chi
                        ndtot+=nd
                        if chi2_pars.calc_cl:
                            chilim_fill(nd,chi,dlab)

                if str(dlab) in ['CMSTTBARTOT13TEV','CMSTTBARTOT8TEV','ATLAS_TTB_DIFF_8TEV_LJ_TTRAPNORM','ATLAS_TTBARTOT_13TEV_FULLLUMI','ATLASTTBARTOT8TEV','ATLASZPT8TEVYDIST','ATLAS_WM_JET_8TEV_PT','ATLASWZRAP11CF','HERACOMB_SIGMARED_B','HERACOMBNCEP460','HERACOMBNCEP575','HERACOMBNCEP820','HERACOMBNCEP920','HERACOMBCCEM','HERACOMBCCEP']:
                    chi2ind_calc=False

                if chi2ind_calc:

                    if fit_pars.dynT_ngt5:
                        if nd > 4:
                            chi2_pars.chi_ind_arr.append(chi)
                            chi2totind+=chi
                            ndtot+=nd
                            if chi2_pars.calc_cl:
                                chilim_fill(nd,chi,dlab)
                    else:
                        (chi,nd,dlab)=chi2corr_ind(i)
                        chi2_pars.chi_ind_arr.append(chi)
                        chi2totind+=chi
                        ndtot+=nd
                        if chi2_pars.calc_cl:
                            chilim_fill(nd,chi,dlab)
            else:
                (chi,nd,dlab)=chi2corr_ind(i)
                chi2_pars.chi_ind_arr.append(chi)
                chi2totind+=chi
                ndtot+=nd
                if chi2_pars.calc_cl:
                    chilim_fill(nd,chi,dlab)

    out0=np.sum(chiarr)

    out1=0.
    if fit_pars.pos_const:
        out31=pos_calc(fit_pars.pos_data31)
        if(fit_pars.pos_40):
            out40=pos_calc(fit_pars.pos_data40)
            chi2pos=out31+out40
            out1=chi2pos
        else:
            out1=out31
        
    chi2_pars.chi_pos1=out1

    if chi2_pars.calc_cl:
        chi2_pars.chi_pos0=out1
        # chilim_sort()
    chi2_pars.calc_cl=False

    if fit_pars.deld_const:
        # dv
        out0=out0+del_pen(pdf_pars.deld_arr[0])[0]
        # uv
        out0=out0+del_pen(pdf_pars.delu_arr[0])[0]


    return(out0,out1)

def hess_ij_calc_d2(diffi,diffj,cov,covin):

    dattot=dload_pars.darr_gl

    diffp=diffi+diffj
    diffm=diffi-diffj


    outp=diffp@covin@diffp
    outm=diffm@covin@diffm

    out=outp-outm
    out=out/4.
    out=out*2.

    return out

def hess_ij_calc_not0_new(theoryi,theoryj,outi,outj,cov,covin):

    dattot=dload_pars.darr_gl

    diffij=theoryi-theoryj
    # outij=calc_chi2(la.cholesky(cov, lower=True), diffij)
    outij=diffij@covin@diffij
    outa=outi/2.+outj/2.-outij

    out=outa

    return out

def hess_ij_calc_new(theory0,theoryi,theoryj,cov0,covi,covj,out1j,out1i,outi,outj):

    dattot=dload_pars.darr_gl

    diffij=theoryi-theoryj
    outij=calc_chi2(la.cholesky(cov0, lower=True), diffij)

    outa=outi+outj-outij

    diff1=theory0-dattot
    diff2i=theoryi-theory0
    out2i=calc_chi2(la.cholesky(covj, lower=True), diff2i)
    diff3i=theoryi-dattot
    out3i=calc_chi2(la.cholesky(covj, lower=True), diff3i)

    out10=calc_chi2(la.cholesky(cov0, lower=True), diff1)
    out20=outi
    out30=calc_chi2(la.cholesky(cov0, lower=True), diff3i)

    out1ii=out1i-out10
    out2i=out2i-out20
    out3i=out3i-out30

    outb=out3i-out2i-out1ii

    diff2j=theoryj-theory0
    out2j=calc_chi2(la.cholesky(covi, lower=True), diff2j)
    diff3j=theoryj-dattot
    out3j=calc_chi2(la.cholesky(covi, lower=True), diff3j)

    out20=out2j
    out30=calc_chi2(la.cholesky(cov0, lower=True), diff3j)

    out1jj=out1j-out10
    out2j=out2j-out20
    out3j=out3j-out30

    outc=out3j-out2j-out1jj

    out=outa+outb+outc

    return out
 

def hess_ii_calc_d2(diff2,cov,covin):
    
    dattot=dload_pars.darr_gl
#    diff2=tp-tm
    # out2=calc_chi2(la.cholesky(cov, lower=True), diff2)
    out2=diff2@covin@diff2
    out=2.*out2

    return out

def hess_ii_calc_not0(theory0,theoryi,cov,covin):

    dattot=dload_pars.darr_gl

    diff2=theoryi-theory0
    # out2=calc_chi2(la.cholesky(cov, lower=True), diff2)
    out2=diff2@covin@diff2

    out=2.*out2

    return out

def hess_ii_calc_t0(theory0,theoryi,cov0,covi,out10):

    dattot=dload_pars.darr_gl
        
    diff1=theory0-dattot
    out1i=calc_chi2(la.cholesky(covi, lower=True), diff1)
    diff2=theoryi-theory0
    out2i=calc_chi2(la.cholesky(covi, lower=True), diff2)
    diff3=theoryi-dattot
    out3i=calc_chi2(la.cholesky(covi, lower=True), diff3)

    out20=calc_chi2(la.cholesky(cov0, lower=True), diff2)
    out30=calc_chi2(la.cholesky(cov0, lower=True), diff3)

    out1=out1i-out10
    out2=out2i-out20*2.
    out3=out3i-out30

    out=2.*(out3-out2-out1)

    return(out,out1i,out20)
    
def betacalc_not0(theory0,theoryi,cov0):

    print('test')

    dattot=dload_pars.darr_gl

    diff1=theory0-dattot
    out1=calc_chi2(la.cholesky(cov0, lower=True), diff1)
    diff2=theoryi-theory0
    out2=calc_chi2(la.cholesky(cov0, lower=True), diff2)
    diff3=theoryi-dattot
    out3=calc_chi2(la.cholesky(cov0, lower=True), diff3)

    out=out3-out2-out1

    return out

def betacalc(theory0,theoryi,cov0,covi):

    dattot=dload_pars.darr_gl

    diff1=theory0-dattot
    out1=calc_chi2(la.cholesky(cov0, lower=True), diff1)
    diff2=theoryi-theory0
    out2=calc_chi2(la.cholesky(cov0, lower=True), diff2)
    diff3=theoryi-dattot
    out3=calc_chi2(la.cholesky(cov0, lower=True), diff3)

    diffc=theory0-dattot
    outc=calc_chi2(la.cholesky(covi, lower=True), diffc)
    
    out=out3-out2-out1
    out=out3-out2-2.*out1+outc  # outc-out1: dcov/dpar
    
    return out

def jaccalc_d4(label_arr,label_arrm,label_arrm2,label_arrp2,eps_arr,hess_calc,il,ih):

    imax=fit_pars.imaxdat

    chiarr=np.zeros((pdf_pars.npar_free+1))
    chiarrp=np.zeros((pdf_pars.npar_free+1))
    chiarrm=np.zeros((pdf_pars.npar_free+1))
    chiarrm2=np.zeros((pdf_pars.npar_free+1))
    chiarrp2=np.zeros((pdf_pars.npar_free+1))

    hessarr=np.zeros((pdf_pars.npar_free+1,pdf_pars.npar_free+1))

    pdf_pars.PDFlabel=label_arr[0].strip()

    pdf_pars.iPDF=0
    (chiarr[0],theory0,cov0,cov0in)=chi2corr(il,ih-1)
    difft=[0.]

    out0=chiarr[0]

    for ip in range(1,pdf_pars.npar_free+1):
        pdf_pars.PDFlabel=label_arr[ip].strip()
        pdf_pars.iPDF=ip
        (chiarrp[ip],theoryp,cov,covin)=chi2corr(il,ih-1)
        pdf_pars.PDFlabel=label_arrm[ip].strip()
        pdf_pars.iPDF=ip+pdf_pars.npar_free
        (chiarrm[ip],theorym,cov,covin)=chi2corr(il,ih-1)
        pdf_pars.PDFlabel=label_arrm2[ip].strip()
        pdf_pars.iPDF=ip+pdf_pars.npar_free*2
        (chiarrm2[ip],theorym2,cov,covin)=chi2corr(il,ih-1)
        pdf_pars.PDFlabel=label_arrp2[ip].strip()
        pdf_pars.iPDF=ip+pdf_pars.npar_free*3
        (chiarrp2[ip],theoryp2,cov,covin)=chi2corr(il,ih-1)

        theoryd=theorym2-8.*theorym+8.*theoryp-theoryp2
        difft.append(theoryd)


    for ip in range(1,pdf_pars.npar_free+1):

        chiarr[ip]=chiarrm2[ip]-8.*chiarrm[ip]+8.*chiarrp[ip]-chiarrp2[ip]
        chiarr[ip]=chiarr[ip]/eps_arr[ip]/12.


    chiarr=np.delete(chiarr,0)
    jacarr=chiarr

    if hess_calc:

        for ip in range(1,pdf_pars.npar_free+1):

            if ip==1:
                tii0=hess_ii_calc_d2(difft[ip],cov0,cov0in)
                tii=[tii0]
            else:
                tii0=hess_ii_calc_d2(difft[ip],cov0,cov0in)
                tii.append(tii0)

            for jp in range(1,ip+1):

                if ip==jp:
                    hii=tii[ip-1]
                    hii=hii/np.power(eps_arr[ip],2)/144.
                    hessarr[ip,jp]=hii
                else:
                    hij=hess_ij_calc_d2(difft[ip],difft[jp],cov0,cov0in)
                    hij=hij/eps_arr[ip]/eps_arr[jp]/144.
                    hessarr[ip,jp]=hij

        hessarr=np.delete(hessarr,0,0)
        hessarr=np.delete(hessarr,0,1)
        hessarr=hessarr+hessarr.T-np.diag(hessarr.diagonal())

    out1=0.
    if fit_pars.pos_const:
#        for ip in range(0,pdf_pars.npar_free+1):                                                                                                              
        pdf_pars.PDFlabel=label_arr[ip].strip()
        out31=pos_calc(fit_pars.pos_data31)
        if(fit_pars.pos_40):
            out40=pos_calc(fit_pars.pos_data40)
            chi2pos=out31+out40
            out1=chi2pos
        else:
            out1=out31
    #            chiarr[ip]=chiarr[ip]+out1    



    return (out0,out1,jacarr,hessarr)

def jaccalc_d0(label_arr,eps_arr,il,ih):

    chiarr=np.zeros((pdf_pars.npar_free+1))
    pdf_pars.PDFlabel=label_arr[0].strip()

    print(pdf_pars.PDFlabel)

    (chiarr[0],theory0,cov0,cov0in)=chi2corr(il,ih-1)
    difft=[0.]

    out0=chiarr[0]

    for ip in range(1,pdf_pars.npar_free+1):
        pdf_pars.PDFlabel=label_arr[ip].strip()
        (chiarr[ip],theoryp,cov,covin)=chi2corr(il,ih-1)

    for ip in range(1,pdf_pars.npar_free+1):

        chiarr[ip]=chiarr[ip]-out0
        chiarr[ip]=chiarr[ip]/eps_arr[ip]

    chiarr=np.delete(chiarr,0)
    jacarr=chiarr

    return jacarr

def jaccalc(label_arr,label_arrm,eps_arr,hess_calc):

    imax=fit_pars.imaxdat

    chiarr=np.zeros((pdf_pars.npar_free+1))
    jacarr=np.zeros((pdf_pars.npar_free+1))
    hessarr=np.zeros((pdf_pars.npar_free+1,pdf_pars.npar_free+1))

    pdf_pars.PDFlabel=label_arr[0].strip()
    pdf_pars.iPDF=0
    (chiarr[0],theory0,cov0,cov0in)=chi2corr(fit_pars.imindat,imax-1)
    
    tarr=[theory0]
    covarr=[cov0]

    print(pdf_pars.npar_free)

    for ip in range(1,pdf_pars.npar_free+1):
        print(ip)
        pdf_pars.iPDF=ip
        pdf_pars.PDFlabel=label_arr[ip].strip()
        (chiarr[ip],theory,cov,covin)=chi2corr(fit_pars.imindat,imax-1)
        # chiarr[ip]=chiarr[0]
        # theory=theory0
        # cov=cov0
        # cov
        tarr.append(theory)
        if(chi2_pars.uset0cov):
            covarr.append(cov)
        
    for ip in range(1,pdf_pars.npar_free+1):

#        test=betacalc(theory0,tarr[ip],cov0,covarr[ip])
#        test=test/eps_arr[ip]
#        jacarr[ip]=test

        test1=chiarr[ip]-chiarr[0]
        # print('jact',ip,chiarr[ip],chiarr[0])
#        print('diff :', test1,eps_arr[ip])
        test1=test1/eps_arr[ip]
        jacarr[ip]=test1   
#        print('jac :',test1)

        if(hess_calc):

            if chi2_pars.uset0cov:

                if ip==1:
                    (tiia,tiib,tiic)=hess_ii_calc_t0(theory0,tarr[ip],cov0,covarr[ip],chiarr[0])
                    tii=[tiia]
                    ti1=[tiib]
                    ti2=[tiic]
                else:
                    (tiia,tiib,tiic)=hess_ii_calc_t0(theory0,tarr[ip],cov0,covarr[ip],chiarr[0])
                    tii.append(tiia)
                    ti1.append(tiib)
                    ti2.append(tiic)
                    
            else:
                
                if ip==1:
                    tii0=hess_ii_calc_not0(theory0,tarr[ip],cov0,cov0in)
                    tii=[tii0]
                else:
                    tii0=hess_ii_calc_not0(theory0,tarr[ip],cov0,cov0in)
                    tii.append(tii0)
                    
            for jp in range(1,ip+1):
                # print(ip,jp)
                if ip==jp:
                    hii=tii[ip-1]
                    hii=hii/np.power(eps_arr[ip],2)
                    hessarr[ip,jp]=hii
                else:
                    if(chi2_pars.uset0cov):
                        hij=hess_ij_calc_new(theory0,tarr[ip],tarr[jp],cov0,covarr[ip],covarr[jp],ti1[ip-1],ti1[jp-1],ti2[ip-1],ti2[jp-1])
                    else:
                        # print('hij...')
                        hij=hess_ij_calc_not0_new(tarr[ip],tarr[jp],tii[ip-1],tii[jp-1],cov0,cov0in)
                        # print('...done')
                    hij=hij/eps_arr[ip]/eps_arr[jp]
                    hessarr[ip,jp]=hij
    
    jacarr=np.delete(jacarr,0)

    hessarr=np.delete(hessarr,0,0)
    hessarr=np.delete(hessarr,0,1)
    hessarr=hessarr+hessarr.T-np.diag(hessarr.diagonal()) 
    
    out0=chiarr[0]
    
    chiarr=np.zeros((pdf_pars.npar_free+1))
    chiarrm=np.zeros((pdf_pars.npar_free+1))
    penarr=np.zeros((pdf_pars.npar_free+1))

    if fit_pars.pos_const:
        for ip in range(0,pdf_pars.npar_free+1):
            pdf_pars.PDFlabel=label_arr[ip].strip()
            out31=pos_calc(fit_pars.pos_data31)
            if(fit_pars.pos_40):
                out40=pos_calc(fit_pars.pos_data40)
                chi2pos=out31+out40
                out1=chi2pos
            else:
                out1=out31
            chiarr[ip]=chiarr[ip]+out1

    if fit_pars.pos_const:
        for ip in range(1,pdf_pars.npar_free+1):
            pdf_pars.PDFlabel=label_arrm[ip].strip()
            out31m=pos_calc(fit_pars.pos_data31)            
            if(fit_pars.pos_40):
                out40m=pos_calc(fit_pars.pos_data40)
                chi2posm=out31m+out40m
                out1m=chi2posm
            else:
                out1m=out31m

            chiarrm[ip]=chiarrm[ip]+out1m    

    out1=chiarr[0]

    if fit_pars.deld_const:
        (chi0d,chi0u,diffd,diffu,hessd,hessu,idv,iuv)=del_pen_calc()
        out1=out1+chi0d
        out1=out1+chi0u
        # print(chi0d,chi0u,diffd,diffu,hessd,hessu)
        print('deld_const...')
        # print(chi0d,chi0u)
        if idv > 0:
            # print('idv=',idv,jacarr[idv-1])
            jacarr[idv-1]=jacarr[idv-1]+diffd   
            # print('idv=',idv,jacarr[idv-1])         
            # print(hessarr[idv-1,idv-1])
            hessarr[idv-1,idv-1]=hessarr[idv-1,idv-1]+hessd
            # print(idv,diffd,hessd)
        if iuv > 0:
            # print('iuv=',iuv,jacarr[iuv-1])
            jacarr[iuv-1]=jacarr[iuv-1]+diffu
            # print('iuv=',iuv,jacarr[iuv-1])
            # print(hessarr[iuv-1,iuv-1])
            hessarr[iuv-1,iuv-1]=hessarr[iuv-1,iuv-1]+hessu
            # print(iuv,diffu,hessu)

    hessparr=np.zeros((pdf_pars.npar_free+1,pdf_pars.npar_free+1))
    
    if fit_pars.pos_const:

        for ip in range(1,pdf_pars.npar_free+1):
           
            hessparr[ip,ip]=(chiarr[ip]-2.*chiarr[0]+chiarrm[ip])/np.power(eps_arr[ip],2)
            chiarr[ip]=chiarr[ip]-chiarrm[ip]
            chiarr[ip]=chiarr[ip]/eps_arr[ip]/2.

        
    if chi2_pars.add_hessp:

        for ip in range(1,pdf_pars.npar_free+1):
            for jp in range(1,pdf_pars.npar_free+1):
                if ip==jp:
                    hessarr[ip-1,ip-1]=hessarr[ip-1,ip-1]+hessparr[ip,ip]


    chiarr=np.delete(chiarr,0)

    hessparr=np.delete(hessparr,0,0)
    hessparr=np.delete(hessparr,0,1)
    

    jacarr=jacarr+chiarr

    
    if(chi2_pars.jac_b):

        jacarr_b=np.zeros((pdf_pars.npar_free+1))
        if chi2_pars.uset0cov:
            jacarr_bt0=np.zeros((pdf_pars.npar_free+1))
        
        for ip in range(1,pdf_pars.npar_free+1):
            jacarr_b[ip]=betacalc_not0(theory0,tarr[ip],cov0)
            jacarr_b[ip]=jacarr_b[ip]/eps_arr[ip]

            if chi2_pars.uset0cov:
                jacarr_bt0[ip]=betacalc(theory0,tarr[ip],cov0,covarr[ip])
                jacarr_bt0[ip]=jacarr_bt0[ip]/eps_arr[ip]
        
        jacarr_b=np.delete(jacarr_b,0)
        if chi2_pars.uset0cov:
            jacarr_bt0=np.delete(jacarr_bt0,0)
        
        jacarr=jacarr_b.copy()
        if chi2_pars.uset0cov:
            jacarr=jacarr_bt0.copy()

    return(jacarr,hessarr,out0,out1,hessparr)
    
def jaccalc_d2(label_arr,label_arrm,eps_arr,hess_calc,il,ih):

    difft=np.zeros((pdf_pars.npar_free+1))
    chiarr=np.zeros((pdf_pars.npar_free+1))
    chiarrm=np.zeros((pdf_pars.npar_free+1))
    hessarr=np.zeros((pdf_pars.npar_free+1,pdf_pars.npar_free+1))

    pdf_pars.PDFlabel=label_arr[0].strip()

    pdf_pars.iPDF=0
    (chiarr[0],theory0,cov0,cov0in)=chi2corr(il,ih-1)
    difft=[0.]

    out0=chiarr[0]



    for ip in range(1,pdf_pars.npar_free+1):
        pdf_pars.PDFlabel=label_arr[ip].strip()
        pdf_pars.iPDF=ip
        (chiarr[ip],theoryp,cov,covin)=chi2corr(il,ih-1)
        pdf_pars.PDFlabel=label_arrm[ip].strip()
        pdf_pars.iPDF=ip+pdf_pars.npar_free
        (chiarrm[ip],theorym,cov,covin)=chi2corr(il,ih-1)

        theoryd=theoryp-theorym
        difft.append(theoryd)

    for ip in range(1,pdf_pars.npar_free+1):

        chiarr[ip]=chiarr[ip]-chiarrm[ip]
        chiarr[ip]=chiarr[ip]/eps_arr[ip]/2.

    chiarr=np.delete(chiarr,0)
    jacarr=chiarr
    


    if hess_calc:

        for ip in range(1,pdf_pars.npar_free+1):
        

            if ip==1:
                tii0=hess_ii_calc_d2(difft[ip],cov0,cov0in)
                tii=[tii0]
            else:
                tii0=hess_ii_calc_d2(difft[ip],cov0,cov0in)
                tii.append(tii0)

            for jp in range(1,ip+1):

                if ip==jp:
                    hii=tii[ip-1]
                    hii=hii/np.power(eps_arr[ip],2)/4.
                    hessarr[ip,jp]=hii
                else:
#                    hij=hess_ij_calc_d2(tarrp[ip],tarrm[ip],tarrp[jp],tarrm[jp],cov0)
                    hij=hess_ij_calc_d2(difft[ip],difft[jp],cov0,cov0in)
                    hij=hij/eps_arr[ip]/eps_arr[jp]/4.
                    hessarr[ip,jp]=hij

        hessarr=np.delete(hessarr,0,0)
        hessarr=np.delete(hessarr,0,1)
        hessarr=hessarr+hessarr.T-np.diag(hessarr.diagonal())
    
    
    out1=0.
    if fit_pars.pos_const:
        chiarr=np.zeros((pdf_pars.npar_free+1))
        chiarrm=np.zeros((pdf_pars.npar_free+1))
        hessparr=np.zeros((pdf_pars.npar_free+1,pdf_pars.npar_free+1))

        for ip in range(1,pdf_pars.npar_free+1):

            pdf_pars.PDFlabel=label_arr[ip].strip()
            out31=pos_calc(fit_pars.pos_data31)
            if(fit_pars.pos_40):
                out40=pos_calc(fit_pars.pos_data40)
                chi2pos=out31+out40
                out1=chi2pos
            else:
                out1=out31
                pdf_pars.PDFlabel=label_arrm[ip].strip()
            
            chiarr[ip]=out1

            pdf_pars.PDFlabel=label_arrm[ip].strip()
            out31=pos_calc(fit_pars.pos_data31)
            if(fit_pars.pos_40):
                out40=pos_calc(fit_pars.pos_data40)
                chi2pos=out31+out40
                out1=chi2pos
            else:
                out1=out31
                pdf_pars.PDFlabel=label_arrm[ip].strip()
            
            chiarrm[ip]=out1

        for ip in range(1,pdf_pars.npar_free+1):

            hessparr[ip,ip]=(chiarr[ip]-2.*chiarr[0]+chiarrm[ip])/np.power(eps_arr[ip],2)
            chiarr[ip]=chiarr[ip]-chiarrm[ip]
            chiarr[ip]=chiarr[ip]/eps_arr[ip]/2.

        if chi2_pars.add_hessp:

            for ip in range(1,pdf_pars.npar_free+1):
                for jp in range(1,pdf_pars.npar_free+1):
                    if ip==jp:
                        hessarr[ip-1,ip-1]=hessarr[ip-1,ip-1]+hessparr[ip,ip]

        chiarr=np.delete(chiarr,0)
        jacarr=jacarr+chiarr

    if fit_pars.deld_const:
        (chi0d,chi0u,diffd,diffu,hessd,hessu,idv,iuv)=del_pen_calc()
        out1=out1+chi0d
        out1=out1+chi0u
        print(chi0d,chi0u)
        if idv > 0:
            jacarr[idv-1]=jacarr[idv-1]+diffd
            # print(id,diffd)
            hessarr[idv-1,idv-1]=hessarr[idv-1,idv-1]+hessd
        if iuv > 0:
            jacarr[iuv-1]=jacarr[iuv-1]+diffu
            # print(iu,diffu)
            hessarr[iuv-1,iuv-1]=hessarr[iuv-1,iuv-1]+hessu

    return (out0,out1,jacarr,hessarr)


def hess_zeros(hess):

    hessi=hess.copy()
    dimh=len(hess)
    
    result = np.all((hess == 0), axis=1)
    
    for i in range(len(result)):
        if result[i]:
            print('Row: ', i)
            print(hess[i,:])
#            hessi[i,:]=1e-10
#            hessi[:,i]=1e-10

    idx = np.argwhere(np.all(hessi[..., :] == 0, axis=0))
#    print('idx',idx)
    hessd = np.delete(hessi, idx, axis=1)
    hessd1=np.delete(hessd,idx,axis=0)
#    print('hessi',hessi)
#    print('hessd',hessd)
#    print('hessd1',hessd1)

    hessin=la.inv(hessd1)

    result=hessin.copy()

    output_size=(dimh,dimh)
    indices=(idx,idx)
    
    result = np.zeros(output_size)
    existing_indices = [np.setdiff1d(np.arange(axis_size), axis_indices,assume_unique=True)
                        for axis_size, axis_indices in zip(output_size, indices)]
    result[np.ix_(*existing_indices)] = hessin 

    hessd2=result
        
#    print('hessd2',hessd2)
   
    return hessd2

#  allows theory values in covariance matrix to be overwritten if need be (for MSHT)

def calc_covmat_t0(inpt0):

    lcd = API.dataset_inputs_loaded_cd_with_cuts(dataset_inputs=dload_pars.dscomb,theoryid=fit_pars.theoryidi,use_cuts='internal')
    dtest=API.data_input(**inpt0)

    list_preds=list(fit_pars.preds_stored.keys())

    preds=[]
    for label in list_preds:
        preds.append(np.array(fit_pars.preds_stored[label]))

    cov = dataset_inputs_covmat_from_systematics(lcd,dtest,True,None,preds)

    return cov
