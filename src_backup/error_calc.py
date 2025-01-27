from global_pars import *
import numpy as np
import scipy.linalg as la
from chi2s import *
from outputs import *
from scipy.optimize import newton

def hesserror_dynamic_tol(afi,hess,jaci):

    chi2_pars.chi2ind=True

    delchisq_exact=True
    tol=np.sqrt(10.0)
    tol=1.0

    tol=np.sqrt(chi2_pars.t2_err)

    inhess=True
    pdfout=True
    delchi50=False
    chi50=False


    output='outputs/evscans/'+inout_pars.label+'.dat'

    if(inhess):
        hessinv=la.inv(hess)
        lam,eig = la.eigh(hessinv)
        hessin_d=la.inv(eig)@hessinv@eig

    lam0,eig0=la.eigh(hess)
    hess_d=la.inv(eig0)@hess@eig0

    if inhess:
        lam,eig = la.eigh(hessinv)
    else:
        lam0,eig=la.eigh(hess)
        lam=1./lam0        
        lam=np.flip(lam)
        eig=np.flip(eig,1)

#### Remove negative eigenvalues
    nlam_neg=np.count_nonzero(lam < 0, axis=0)
    if(nlam_neg > 0):
        lam_noneg=np.delete(lam,np.s_[-nlam_neg:])
        lam=lam_noneg.copy()
        eig_noneg=eig[:, :-nlam_neg]
        eig=eig_noneg.copy()

    eig=eig*np.sqrt(lam)

    
    af=afi.copy()
    chiarr=np.zeros((1))
    
    imax=1
    imaxd=imax

    chi0=chi2min(afi)
    

    if pdfout:
        evgrido()
        gridout()
        fit_pars.irep=fit_pars.irep+1

    with open(output,'w') as outputfile:
        outputfile.write('imax= ')
        outputfile.write(str(imax))
        outputfile.write('\n')

    chiarr=np.zeros((1))
    chiarr_t=np.zeros((1))
    t_chi_arr=np.zeros((1))
       


    for j in range(0,len(lam)):

        print('j = ',j)
        
        # Delete grids with forbidden pars at t=1
        if chi50:
            if delchi50:
                fit_pars.irep=fit_pars.irep-2

        chi50=False
        eig0=eig[:,j].flatten()

        print('eig0 = ',eig0)

        for i in range(-imaxd,imax+1):
            
            t0=0.
            chiarr=np.zeros((1))

            print('i=',i)

            if i==0:
                chi2t=chi0
                # chiarr=np.append(chiarr,chi2t)
            else:

                idelmax=40
                t_chi=0.
                delt=0.5*tol
                chi2t_tm=0.
                deltachi2_limo=np.zeros((len(chi2_pars.chi_ind_arr)))   
                for idel in range(1,idelmax+1):
                    t_chi=t_chi+delt
                    afint=af.copy()+i*t_chi*eig0.copy()
                    chi2t_t=chi2min(afint)-chi0
                    print('t_chi = ',t_chi)
                    print('delchi = ',chi2t_t)
                    deltachi2=np.array(chi2_pars.chi_ind_arr)-np.array(chi2_pars.chi0_ind_arr)
                    # print(deltachi2)
                    deltachi2_lim=deltachi2/np.array(chi2_pars.chilim_arr)
                    dchi2max=np.max(deltachi2_lim)
                    arg_dchi2max=np.argmax(deltachi2_lim)
                    # print(deltachi2_lim)
                    print(np.max(deltachi2_lim))
                    print(np.argmax(deltachi2_lim))

                    if dchi2max > 1.:
                        yi=deltachi2_lim[arg_dchi2max]
                        yo=deltachi2_limo[arg_dchi2max]
                        mt=yi-yo
                        mt=mt/(np.power(t_chi,2)-np.power(t_chi-delt,2))
                        t_chi=(1.-yo)/mt+np.power(t_chi-delt,2)
                        t_chi=np.sqrt(t_chi)
                        afint=af.copy()+i*t_chi*eig0.copy()
                        chi2t_t=chi2min(afint)
                        print('t_chi final = ',t_chi)
                        print('delchi final = ',chi2t_t-chi0)
                        break

                    deltachi2_limo=deltachi2_lim        

                with open(output,'a') as outputfile:
                    if j==0 and i==-imax:
                        outputfile.write('chi0 = ')
                        outputfile.write(str(chi0))
                        outputfile.write('\n')
                        outputfile.write('Delta chi^2, t, limiting dataset')
                        outputfile.write('\n')


                    outputfile.write(str(chi2t_t-chi0))
                    outputfile.write(' ')
                    outputfile.write(str(t_chi)) 
                    outputfile.write(' ')
                    outputfile.write(chi2_pars.cldataset_arr[arg_dchi2max])    
                    outputfile.write('\n')       
                    if pdfout:
                        evgrido()
                        gridout()
                        fit_pars.irep=fit_pars.irep+1


def hesserror_dynamic_tol_new(afi,hess,jaci):

    output='outputs/evscans/'+inout_pars.label+'.dat'

    chi2_pars.chi2ind=True
    chi2_pars.chitotind=False
    chi2_pars.L0=False

    # hessinv=la.inv(hess)
    # lamt,eigt = la.eigh(hessinv)

    pdfout=True 
    delchisq_exact=True
    # For closure tests use experimental cov matrix (quicker)
    if(inout_pars.pdin):
        chi2_pars.t0=False

    # chi2_pars.t0=False

    tol_exact=0.01
    tol=np.sqrt(10.0)
    tol=1.0

    tol=np.sqrt(chi2_pars.t2_err)

#   Calculate eigenvalues + eigenvector of Hessian and then invert/flip so correspond to C=H^-1 with appropriate ordering
    lam0,eig=la.eigh(hess)
    lam=1./lam0        
    lam=np.flip(lam)
    eig=np.flip(eig,1)

#### Remove negative eigenvalues
    nlam_neg=np.count_nonzero(lam < 0, axis=0)
    if(nlam_neg > 0):
        lam_noneg=np.delete(lam,np.s_[-nlam_neg:])
        lam=lam_noneg.copy()
        eig_noneg=eig[:, :-nlam_neg]
        eig=eig_noneg.copy()

#   Rescale
    eig=eig*np.sqrt(lam)
  
#   Central chi^2 - from full covariance matrix


    af=afi.copy()
    chi0=chi2min(afi)
    
    # exit()

#   Output central grid and increase replica number ready for eigenvector generation

    if pdfout:
        evgrido()
        gridout()
        fit_pars.irep=fit_pars.irep+1

#   Header for evscan file

    with open(output,'w') as outputfile:
        L = ['chi2_0 = ',str(f'{chi0:.5f}'),', neig = ',str(len(lam))]
        outputfile.writelines(L) 
        outputfile.write('\n')


#   Loop over eigenvectors
    

    # fit_pars.irep=43*2+1

    for j in range(0,len(lam)):
    # for j in range(45,46):

        print('j = ',j)

        # jth eigenvector       
        eig0=eig[:,j].flatten()


        # +/- directions
        # for i in [-1,1]:
        for i in [-1,1]:

            delt=1.0
            idelmax=20
            deltachi2_limo=np.zeros((len(chi2_pars.chi_ind_arr)))   

            for k in range (1,idelmax):

                tchi=delt*k*i
                print('tchi = ',tchi)
                (deltachi2_lim,dchi2max,arg_dchi2max,chi2t)=get_chi2_ind(af,tchi,eig0.copy())

                if chi2t > 1e40:
                    print('shift outside allowed region -> reduce t')
                    chi2=chi2t
                    while chi2 > 1e40:
                        tchi/=2.
                        print('tchi = ',tchi)
                        afin=af.copy()+tchi*eig0.copy()
                        chi2=chi2min(afin) 
                    delt=np.abs(tchi)/5.
                    tchi=delt*k*i
                    print('tchi = ',tchi)
                    (deltachi2_lim,dchi2max,arg_dchi2max,chi2t)=get_chi2_ind(af,tchi,eig0.copy())

                if dchi2max > 1e1:
                    tchi_i=tchi-delt*i
                    tchi_del=delt
                    print('limit far surpassed -> reduce t by hand')
                    dchi2=dchi2max
                    while dchi2 > 1e1:
                        tchi_del/=2.
                        tchi=tchi_i+tchi_del*i
                        print('tchi = ',tchi)
                        (deltachi2_lim,dchi2,arg_dchi2max,chi2t)=get_chi2_ind(af,tchi,eig0.copy())
                    print('tchi out = ',tchi)


                # test=newton_func_ind(tchi,af,eig0,arg_dchi2max)
                if dchi2max > 1.:
                    try:
                        tchi_out=newton(newton_func_ind,tchi,args=(af,eig0,arg_dchi2max), maxiter=40, tol=0.01)
                    except RuntimeError as err:
                        print('newton method failing, try brute force...')
                        dchi2=10.
                        while dchi2 > 1.:
                            tchi-=0.05
                            print('tchi = ',tchi)
                            (deltachi2_lim,dchi2,arg_dchi2max,chi2t)=get_chi2_ind(af,tchi,eig0.copy())
                        tchi_out=tchi

                    print('tchi_out = ',tchi_out)
                    # afin=af.copy()+tchi_out*eig0.copy()
                    (deltachi2_lim,dchi2,arg_dchi2max,chi2t)=get_chi2_ind(af,tchi_out,eig0.copy())
                    # delchi2t=chi2min(afin)-chi0
                    delchi2t=chi2t-chi0
                    print('delchi2 out = ',delchi2t)
                    if delchi2t > 1e10:
                        print('tchi leads to unstable region of parameter space - set t=0')
                        tchi_out=1e-10
                        delchi2t=0.
                    # if np.abs(delchi2t-np.power(tol,2)) < tol_exact:
                    #     print('Now |delchi2 - T^2| < ',tol_exact,' - done')
                    evscan_output_dyT(pdfout,j,tchi_out,delchi2t,arg_dchi2max)
                    break
                    # exit()

# Function defined so that is zero at deltachi^2_i=deltachi^2_lim for constraining dataset

def newton_func_ind(x,afi,eig,arg_dchi2max):

    af=afi.copy()+eig*x
    chi2t_t=chi2min(af)
    deltachi2=np.array(chi2_pars.chi_ind_arr)-np.array(chi2_pars.chi0_ind_arr)
    deltachi2_lim=deltachi2/np.array(chi2_pars.chilim_arr)
    lim=deltachi2_lim[arg_dchi2max]
    dchi=deltachi2[arg_dchi2max]
    out=lim-1.

    print(arg_dchi2max)
    print('Newton - chi2 =',dchi)
    print('Newton - chi2/lim =',lim)
    print('Newton - tchi =',x)

    dload_pars.tchi_newton=x

    return out

def bruteforce_ind(x,afi,eig,arg_dchi2max):

    af=afi.copy()+eig*x
    chi2t_t=chi2min(af)
    deltachi2=np.array(chi2_pars.chi_ind_arr)-np.array(chi2_pars.chi0_ind_arr)
    deltachi2_lim=deltachi2/np.array(chi2_pars.chilim_arr)
    lim=deltachi2_lim[arg_dchi2max]
    dchi=deltachi2[arg_dchi2max]
    out=lim-1.

    print('Brute force - chi2 =',dchi)
    print('Brute force - chi2/lim =',lim)
    print('Brute Force - tchi =',x)

    return out


def get_chi2_ind(af,t,eig):

    afint=af.copy()+t*eig
    chi2t_t=chi2min(afint)
    deltachi2=np.array(chi2_pars.chi_ind_arr)-np.array(chi2_pars.chi0_ind_arr)
    deltachi2_lim=deltachi2/np.array(chi2_pars.chilim_arr)
    dchi2max=np.max(deltachi2_lim)
    arg_dchi2max=np.argmax(deltachi2_lim)
    # print(deltachi2)
    print('deltachi^2/lim = ',dchi2max)
    # print(chi2_pars.chilim_arr)
    # print(np.max(deltachi2_lim))
    print('limiting dataset = ',np.argmax(deltachi2_lim))
    # print(deltachi2[np.argmax(deltachi2_lim)])
    # exit()

    return(deltachi2_lim,dchi2max,arg_dchi2max,chi2t_t)
    

def hessfix(hess):

    inputfile='input/fixinput.dat'
    distuv=np.loadtxt(inputfile,skiprows=1,max_rows=9,usecols=1)
    distdv=np.loadtxt(inputfile,skiprows=11,max_rows=9,usecols=1)
    distsea=np.loadtxt(inputfile,skiprows=21,max_rows=9,usecols=1)
    distsp=np.loadtxt(inputfile,skiprows=31,max_rows=9,usecols=1)
    distg=np.loadtxt(inputfile,skiprows=41,max_rows=10,usecols=1)
    distsm=np.loadtxt(inputfile,skiprows=52,max_rows=9,usecols=1)
    distdbub=np.loadtxt(inputfile,skiprows=62,max_rows=8,usecols=1)
    distcharm=np.loadtxt(inputfile,skiprows=71,max_rows=9,usecols=1)

    disttot=np.concatenate([distuv,distdv,distsea,distsp,distg,distsm,distdbub,distcharm])


#   remove pars from array when fixed

    pdf_pars.npar_free=0
    afin=np.zeros((1))
    dist_afin=np.zeros((1))
    pdf_pars.par_free_i=np.zeros((1),dtype=np.int8)
    for i in range(0,len(pdf_pars.parsin)):
        if pdf_pars.par_isf[i]==1:
            dist_afin=np.append(dist_afin,disttot[i])
        if pdf_pars.par_isf[i]==1 and disttot[i] == 0:
            afin=np.append(afin,pdf_pars.parsin[i])
            pdf_pars.npar_free+=1
            pdf_pars.par_free_i=np.append(pdf_pars.par_free_i,i)
        else:
            pdf_pars.par_isf[i]=0

    afin=np.delete(afin,0)
    dist_afin=np.delete(dist_afin,0)
    pdf_pars.par_free_i=np.delete(pdf_pars.par_free_i,0)

    hessout=hess.copy()

    mask=(dist_afin > 0)
    # print('')
    # print(mask)
    # print('')
    hessout=np.delete(hessout, mask, axis=0)
    hessout=np.delete(hessout, mask, axis=1)

    # for i in range (0,len(afin)+1):
    #     if dist_afin[i] > 0:
    #         hessout=np.delete(hessout, (i), axis=0)
    #         hessout=np.delete(hessout, (i), axis=1)

    #     # for j in range (0,len(afin)+1):
    #     #     if dist_afin[i] > 0:
    #     #         print(i,j)
    #             # hessout[i,j]=1e50
    #             # hessout[j,i]=1e50

    # print('')
    # print(hess)
    # print('')
    # print(hessout)

    # mask= (hessout < 1e50)
    # idx=mask.any()

    # # print(mask)
    # # print(idx)
    # # print('')

    # hessout=np.delete(hessout, np.where(hessout > 1e40)[0],axis=0)

    # print('')
    # print(hessout)
    # print('')
    # print(hessout.shape,hess.shape)

    # exit()

    return (hessout,afin)

def hesserror_new(afi,hess,jaci):


    msht_fix=False

    if msht_fix:
        (hess,afi)=hessfix(hess)

    output='outputs/evscans/'+inout_pars.label+'.dat'

    # hessinv=la.inv(hess)
    # lamt,eigt = la.eigh(hessinv)

    pdfout=True 
    delchisq_exact=True
    # For closure tests use experimental cov matrix (quicker)
    if(inout_pars.pdin):
        chi2_pars.t0=False


    tol_exact=0.01
    tol=np.sqrt(10.0)
    tol=1.0

    tol=np.sqrt(chi2_pars.t2_err)

#   Calculate eigenvalues + eigenvector of Hessian and then invert/flip so correspond to C=H^-1 with appropriate ordering
    lam0,eig=la.eigh(hess)
    lam=1./lam0        
    lam=np.flip(lam)
    eig=np.flip(eig,1)



#### Remove negative eigenvalues
    nlam_neg=np.count_nonzero(lam < 0, axis=0)

    if(nlam_neg > 0):
        lam_noneg=np.delete(lam,np.s_[-nlam_neg:])
        lam=lam_noneg.copy()
        eig_noneg=eig[:, :-nlam_neg]
        eig=eig_noneg.copy()


#   Rescale
    eig=eig*np.sqrt(lam)

    # print(lam)
    # print('eig')
    # print(eig[:,0].flatten())
    # exit()

#   Central chi^2

    af=afi.copy()
    chi0=chi2min(afi)
    chi0pos=chi2_pars.chi_pos1



#   Output central grid and increase replica number ready for eigenvector generation

    if pdfout:
        evgrido()
        gridout()
        fit_pars.irep=fit_pars.irep+1

#   Header for evscan file

    with open(output,'w') as outputfile:
        L = ['chi2_0 = ',str(f'{chi0:.5f}'),', neig = ',str(len(lam))]
        outputfile.writelines(L) 
        outputfile.write('\n')

    # fit_pars.irep=41

#   Loop over eigenvectors

    # for j in range(0,1):
    for j in range(0,len(lam)):

        print('j = ',j)

        # jth eigenvector       
        eig0=eig[:,j].flatten()



        # +/- directions
        # for i in [-1,1]:
        for i in [-1,1]:
            
            # new parameter
            tchi=i*tol

            afin=af.copy()+tchi*eig0.copy()

            print(tchi)
            print(eig0)

            print('delpar = ',afin-af)
            chi2t=chi2min(afin) 
            delchi2t=chi2t-chi0
            print('chidel=',delchi2t)
            print(chi2t,chi0)

            if delchi2t > 1e40:
                print('shift outside allowed region -> reduce t')
                delchi2=delchi2t
                while delchi2 > 1e40:
                    tchi/=2.
                    print('tchi = ',tchi)
                    afin=af.copy()+tchi*eig0.copy()
                    chi2t=chi2min(afin) 
                    delchi2=chi2t-chi0
                delchi2t=delchi2

            if not delchisq_exact:
                print('fixed t - output')
                delchi2pos=chi2_pars.chi_pos1-chi0pos
                evscan_output(pdfout,j,tchi,delchi2t,delchi2pos)
            elif delchi2t < 0.:
                print('delchi^2 less than 0 - minimum not reached!')
                (tchi,delchi2)=tchi_findmin(delchi2t,chi0,tchi,af,eig0,tol)
                # parsout()
                if delchi2 > np.power(tol,2)*5.:
                    (delchi2,tchi)=dchi2_toolarge(tol,tchi,af,eig0,chi0,delchi2)
                delchi2pos=chi2_pars.chi_pos1-chi0pos
                evscan_output(pdfout,j,tchi,delchi2,delchi2pos)
                # exit()
            elif np.abs(delchi2t-np.power(tol,2)) < tol_exact:
                print('|delchi2 - T^2| < ',tol_exact,' - done')
                delchi2pos=chi2_pars.chi_pos1-chi0pos
                evscan_output(pdfout,j,tchi,delchi2t,delchi2pos)
            else:
                # If delchi^2 really big just keep dividing t/2 before applying Newton's method (quicker)
                if delchi2t > np.power(tol,2)*5.:
                    (delchi2t,tchi)=dchi2_toolarge(tol,tchi,af,eig0,chi0,delchi2t)
                    if delchi2t < 0.:
                        (tchi,delchi2t)=tchi_findmin(delchi2t,chi0,tchi,af,eig0,tol)

                # tchi_out=tchi_bruteforce(delchi2t,chi0,tchi,af,eig0,tol)
                print('|delchi2 - T^2| > ',tol_exact, " -> Use Newton's method")
                try:
                    tchi_out=newton(newton_func,tchi,args=(af,eig0,tol,chi0), maxiter=40, tol=0.01)
                    # If min has taken t to the other eigenvector direction use brute force instead
                    if tchi_out/(i*tol) < 0:
                        print('tchi has wrong sign!')
                        tchi_out=tchi_bruteforce(delchi2t,chi0,tchi,af,eig0,tol)
                    print('tchi out = ',tchi_out)
                    afin=af.copy()+tchi_out*eig0.copy()
                    delchi2t=chi2min(afin)-chi0
                    print('delchi2 out = ',delchi2t)
                    if delchi2t > np.power(tol,2)*5.:
                        (delchi2t,tchi)=dchi2_toolarge(tol,tchi,af,eig0,chi0,delchi2t)
                    # if np.abs(delchi2t-np.power(tol,2)) < tol_exact:
                    #     print('Now |delchi2 - T^2| < ',tol_exact,' - done')
                    delchi2pos=chi2_pars.chi_pos1-chi0pos
                    evscan_output(pdfout,j,tchi_out,delchi2t,delchi2pos)
                except RuntimeError as err:
                    print(err)
                    tchi_out=dload_pars.tchi_newton
                    print('tchi = ',tchi_out)
                    afin=af.copy()+tchi_out*eig0.copy()
                    delchi2t=chi2min(afin)-chi0
                    delchi2pos=chi2_pars.chi_pos1-chi0pos
                    evscan_output(pdfout,j,tchi_out,delchi2t,delchi2pos)


def dchi2_toolarge(tol,tchi,af,eig0,chi0,delchi2in):

    delchi2=delchi2in
    print("delchi2 very big - divide t by 2 before applying Newton's method")
    while delchi2 > np.power(tol,2)*5.:
        tchi/=2.
        print('tchi = ',tchi)
        afin=af.copy()+tchi*eig0.copy()
        chi2t=chi2min(afin) 
        delchi2=chi2t-chi0
        print('delchi2 = ',delchi2)
    print('delchi2 = ',delchi2)

    return (delchi2,tchi)

def tchi_findmin(delchi,chi0,tchi,afi,eig,tol):     

    print('Starting findmin...')

    tmin=0.
    delt=0.1*tchi
    imax=40    

    dchi_min=1e40
    t=0.

    printmin=True

    for i in range(1,imax):
        t+=delt
        print('tchi = ',t)
        af=afi.copy()+t*eig
        dchi_f=chi2min(af)-chi0 
        print('delchi2 = ',dchi_f)
        print('dchi_min = ',dchi_min)
        if dchi_min > dchi_f:
            dchi_min=dchi_f
            if printmin:
                if chi2_pars.chidel_min > dchi_f:
                    parsout()
        elif printmin:
            print('min reached!')
            t-=delt 
            af=afi.copy()+t*eig
            dchi_f=chi2min(af)-chi0 
            if chi2_pars.chidel_min > dchi_f:
                chi2_pars.chidel_min=dchi_f
            print('tchi = ',t)
            print('delchi2 = ',dchi_f) 
            printmin=False         
            output='outputs/evscans/'+inout_pars.label+'.dat'
            with open(output,'a') as outputfile:
                outputfile.write('New min = ')
                outputfile.write(str(dchi_f))
                outputfile.write('\n')
        
        if dchi_f > np.power(tol,2)+dchi_min:
            print('tol reached')
            print('tchi = ',t)
            print('delchi2 = ',dchi_f)  
            break

    return (t,dchi_f)

                
def tchi_bruteforce(delchi,chi0,tchi,afi,eig,tol):

    print('Use brute force...')

    if delchi > np.power(tol,2):
        tmin=0.
    else:
        tmin=tchi

    imax=40
    delt=0.01*tchi

    t=tmin
    t_i=t
    dchi_i=delchi

    tout=t

    tarr=np.array([t])
    deltarr=np.array([delchi])

    for i in range(1,imax):
        t+=delt
        af=afi.copy()+t*eig
        dchi_f=chi2min(af)-chi0
        print('t = ',t)
        print('delchi2 = ',dchi_f)
        tarr=np.append(tarr,t)
        deltarr=np.append(deltarr,dchi_f)
        if dchi_f > np.power(tol,2):
            # delt2=np.power(t,2)-np.power(t_i,2)
            # mchi=(dchi_f-dchi_i)/delt2
            # tchi_down=quad_int_tchi(tol,t_i,dchi_i,mchi)
            # tchi_up=quad_int_tchi(tol,t,dchi_f,mchi)
            # af_up=afi.copy()+tchi_up*eig
            # dchi_up=chi2min(af_up)-chi0
            # af_down=afi.copy()+tchi_down*eig
            # dchi_down=chi2min(af_down)-chi0
            # tarr=np.append(tarr,tchi_up)
            # deltarr=np.append(deltarr,dchi_up)
            # tarr=np.append(tarr,tchi_down)
            # deltarr=np.append(deltarr,dchi_down)
            break
        else:
            dchi_i=dchi_f
            t_i=t

    tolarr=np.ones((len(tarr)))*np.power(tol,2)
    deltarr=np.abs(deltarr-tolarr)
    it=np.argmin(deltarr)
    tchi_out=tarr[it]

    return tchi_out

def quad_int_tchi(y,a,b,m):

    x=(np.power(y,2)-b)/m+a
    x=np.sqrt(x)

    return x


# Function that is zero at correct t=T - for use with Newton's method

def newton_func(x,afi,eig,tol,chi0):

    af=afi.copy()+eig*x
    chi2t=chi2min(af) 
    out=chi2t-chi0-np.power(tol,2)

    print('Newton - chidel =',chi2t-chi0)
    print('Newton - tchi =',x)

    dload_pars.tchi_newton=x

    return out

# def tchi_newton_iterator(afi,eig,tol):

#     af=afi.copy()+eig

#     return tout\

def evscan_output_dyT(pdfout,j,tchi,delchi2t,arg_dchi2max):

    output='outputs/evscans/'+inout_pars.label+'.dat'

    if delchi2t < 0:
        delchi2t_sign=-np.sqrt(-delchi2t)*np.sign(tchi)
    else:   
        delchi2t_sign=np.sqrt(delchi2t)*np.sign(tchi)
  
    dlab=chi2_pars.cldataset_arr[arg_dchi2max]

    with open(output,'a') as outputfile:
 
        neig=np.rint((j+1)*np.sign(tchi))

        if np.abs(tchi) < 1e-9:
            tchi=0.

        if np.sign(tchi) > 0:
            L=['eig[',str(neig),']  : ',' t = ',str(f'{tchi:.5f}'),'   T = ',str(f'{delchi2t_sign:.5f}'),' Limiting dataset = ',dlab]
        else:       
            L=['eig[',str(neig),'] : ',' t = ',str(f'{tchi:.5f}'),'  T = ',str(f'{delchi2t_sign:.5f}'),' Limiting dataset = ',dlab]
        outputfile.writelines(L)
        outputfile.write('\n')

    if pdfout:
        evgrido()
        gridout()
        fit_pars.irep=fit_pars.irep+1

def evscan_output(pdfout,j,tchi,delchi2t,delchi2pos):

    output='outputs/evscans/'+inout_pars.label+'.dat'

    if delchi2t < 0:
        delchi2t_sign=-np.sqrt(-delchi2t)*np.sign(tchi)
    else:   
        delchi2t_sign=np.sqrt(delchi2t)*np.sign(tchi)

    with open(output,'a') as outputfile:
 

        neig=np.rint((j+1)*np.sign(tchi))

        if np.sign(tchi) > 0:
            if fit_pars.nnpdf_pos:
                L=['eig[',str(neig),']  : ',' t = ',str(f'{tchi:.5f}'),'   T = ',str(f'{delchi2t_sign:.5f}'), ' T^2(pos) = ',str(f'{delchi2pos:.5f}')]   
            else:
                L=['eig[',str(neig),']  : ',' t = ',str(f'{tchi:.5f}'),'   T = ',str(f'{delchi2t_sign:.5f}')]
        else:    
            if fit_pars.nnpdf_pos:
                L=['eig[',str(neig),'] : ',' t = ',str(f'{tchi:.5f}'),'  T = ',str(f'{delchi2t_sign:.5f}'), ' T^2(pos) = ',str(f'{delchi2pos:.5f}')]
            else:
                L=['eig[',str(neig),'] : ',' t = ',str(f'{tchi:.5f}'),'  T = ',str(f'{delchi2t_sign:.5f}')]
        outputfile.writelines(L)
        outputfile.write('\n')

    if pdfout:
        evgrido()
        gridout()
        fit_pars.irep=fit_pars.irep+1



def hesserror(afi,hess,jaci):


    msht_fix=False
    delchisq_exact=True
    tol=np.sqrt(10.0)
    tol=1.0

    tol=np.sqrt(chi2_pars.t2_err)

    inhess=False
    pdfout=True
    delp=False
    delchi50=False
    chi50=False

    if(inout_pars.pdin):
        chi2_pars.t0=False

    output='outputs/evscans/'+inout_pars.label+'.dat'

    if(inhess):
        hessinv=la.inv(hess)
        lam,eig = la.eigh(hessinv)
        hessin_d=la.inv(eig)@hessinv@eig

    lam0,eig0=la.eigh(hess)
    hess_d=la.inv(eig0)@hess@eig0

    if msht_fix:
        (hess,afi)=hessfix(hess)



    if delp:

        jaci_d=la.inv(eig)@jaci

        delpar=np.zeros((len(jaci)))
        for j in range(0,len(jaci)):
            delpar[j]=jaci_d[j]*hessin_d[j,j]
            #        print(j,jaci_d[j],jaci_d[j]*hessin_d[j,j],hessin_d[j,j])
            

    if inhess:
        lam,eig = la.eigh(hessinv)
    else:
        lam0,eig=la.eigh(hess)
        lam=1./lam0        
        lam=np.flip(lam)
        eig=np.flip(eig,1)
   

#### Remove negative eigenvalues
    nlam_neg=np.count_nonzero(lam < 0, axis=0)
    if(nlam_neg > 0):
        lam_noneg=np.delete(lam,np.s_[-nlam_neg:])
        lam=lam_noneg.copy()
        eig_noneg=eig[:, :-nlam_neg]
        eig=eig_noneg.copy()

    eig=eig*np.sqrt(lam)


    # print(1./lam)
    # exit()

    # for j in range(0,len(lam)): 
    #     eig0=eig[:,j].flatten()
    #     print(j,eig0[42],eig0[1])
    
    # exit()

    # eig0=eig[:,1].flatten()
    # afin=afi.copy()+eig0.copy()*0.8
    

    #    print(afi)
    #    print(afi+delpar[30]*eig0.copy())
    #    print(delpar[30]*eig0.copy()/afi)
    #   
    
    #    chi0t=chi2min(afi)
    #    print(chi0t)
    
    # chi0t=chi2min(afi)
    # eig0=eig[:,23].flatten()
    # afin=afi.copy()-eig0.copy()*1.
    # chi0=chi2min(afin)
    # parsout()
    # print(chi0,chi0t)
    # exit()
    
    af=afi.copy()
    
    chiarr=np.zeros((1))
    
    imax=1
    imaxd=imax

    if delp:
        imaxd=0
        delchisq_exact=False

    chi0=chi2min(afi)


    if pdfout:
        evgrido()
        gridout()
        fit_pars.irep=fit_pars.irep+1

    with open(output,'w') as outputfile:
        outputfile.write('imax= ')
        outputfile.write(str(imax))
        outputfile.write('\n')

    chiarr=np.zeros((1))
    chiarr_t=np.zeros((1))
    t_chi_arr=np.zeros((1))
    
    for j in range(0,len(lam)):
    # for j in range(0,1):

                    

        print('j = ',j)
        

        # Delete grids with forbidden pars at t=1
        if chi50:
            if delchi50:
                fit_pars.irep=fit_pars.irep-2

        chi50=False
                    
        eig0=eig[:,j].flatten()

        print('eig0 = ',eig0)

        # chiarr=np.zeros((1))
        # chiarr_t=np.zeros((1))
        # t_chi_arr=np.zeros((1))
        
        for i in range(-imaxd,imax+1):
            

            t0=0.
            chiarr=np.zeros((1))

            print('i=',i)

            if i==0:
                chi2t=chi0
                # chiarr=np.append(chiarr,chi2t)
            else:
                if delp:
                    afin=af.copy()+delpar[j]*eig0.copy()
                else:
                    afin=af.copy()+i*tol*eig0.copy()
                print('af = ',afin)
                chi2t=chi2min(afin) 
                print('chidel=',chi2t-chi0)
                tol0=tol
                brute=False
                chidel=chi2t-chi0

                if delchisq_exact:
                    if chi2t > 1e40:
                        t_chi=tol0-0.01
                        tol0=tol0-0.01
                        idelmax=100
                        chi50=True
                    elif chidel < 0:
                        # Do brute force if negative
                        idelmax=0 
                        brute=True
                    else:
                        t_chi=np.power(tol,2)/np.sqrt(chidel)
                        idelmax=10

                    if brute:
                        delchit=100.
                    else:   
                        afint=af.copy()+i*t_chi*eig0.copy()
                        chi2t_t=chi2min(afint)
                        delchit=chi2t_t-chi0

                    if np.abs(delchit-np.power(tol,2)) > 0.01:
                        for idel in range(0,idelmax):
                            print('chidel not close enough to tol, iterate...')
                            print('chidel_t =',chi2t_t-chi0)
                            if chi2t > 1e40:
                                t_chi=tol0-0.01*tol
                                tol0=tol0-0.01*tol
                                idelmax+=1
                            else:
                                delchit=chi2t_t-chi0        
                                if delchit < 0:
                                    print('delchi negative - move min and shift')
                                    t0=t_chi
                                    t_chi=t_chi+0.1
                                    afint=af.copy()+i*t_chi*eig0.copy()
                                    chi2t_t=chi2min(afint)
                                    delchit=chi2t_t-chi0
                                else:
                                    t_chi=(t_chi-t0)*tol/np.sqrt(delchit)+t0

                            afint=af.copy()+i*t_chi*eig0.copy()
                            chi2t_t=chi2min(afint)
                            delchit=chi2t_t-chi0

                            if delchit < 0:
                                print('delchi negative - move min and shift')
                                t0=t_chi
                                t_chi=t_chi+0.1
                                afint=af.copy()+i*t_chi*eig0.copy()
                                chi2t_t=chi2min(afint)
                                delchit=chi2t_t-chi0

                            if chi2t > 1e40:
                                if delchit < 1.:
                                    print('largest delta chi^2 allowed...')
                                    print('tchi=',t_chi)                                    
                                    print('chidel_t =',chi2t_t-chi0,chi2t_t,chi0)
                                    break

                            print('tchi=',t_chi)
                            print('chidel_t =',chi2t_t-chi0,chi2t_t,chi0)
                            if np.abs(delchit-np.power(tol,2)) < 0.01:
                                print('chidel close enough!')
                                break
                            
                            if idel > 3:
                                if np.abs(delchit-np.power(tol,2)) > 0.5 :
                                    print('chidel not close enough! Go to Brute force scan...')
                                    break

                        idelmax=10

                        if np.abs(delchit-np.power(tol,2)) > 0.1 or brute:
                            print('chidel not close enough! Brute force scan...')
                            idelmax=40
                            t_chi=0.
                            delt=0.2*tol
                            delt=0.05*tol
                            chi2t_tm=0.
                            delchimin=0.
                            for idel in range(1,idelmax+1):
                                t_chi=t_chi+delt
                                afint=af.copy()+i*t_chi*eig0.copy()
                                chi2t_t=chi2min(afint)-chi0
                                print('t_chi = ',t_chi)
                                print('delchi = ',chi2t_t)
                                print('delchimin = ',delchimin)
                                if chi2t_t < delchimin:
                                    delchimin=chi2t_t
                                tolp=tol+delchimin
                                if brute:
                                    tolp=np.sqrt(tolp)
                                if chi2t_t > (np.power(tol,2)+delchimin):

                                    mt=chi2t_t-chi2t_tm
                                    mt=mt/(np.power(t_chi,2)-np.power(t_chi-delt,2))
                                    t_chi=(tolp*tolp-chi2t_tm)/mt+np.power(t_chi-delt,2)
                                    t_chi=np.sqrt(t_chi)
                                    afint=af.copy()+i*t_chi*eig0.copy()
                                    chi2t_t=chi2min(afint)

                                    # mt=chi2t_t-chi2t_tm
                                    # mt=mt/delt
                                    # t_chi0=t_chi
                                    # t_chi=(tol-chi2t_tm)/mt+t_chi-delt
                                    # afint=af.copy()+i*t_chi*eig0.copy()
                                    # chi2t_t=chi2min(afint)

                                    print('t_chi final = ',t_chi)
                                    print('delchi final = ',chi2t_t-chi0)
                                    break

                                chi2t_tm=chi2t_t

                    chiarr=np.append(chiarr,chi2t_t-chi0)
                    chiarr=np.append(chiarr,i*t_chi)
                    chiarr=np.append(chiarr,lam[j])
                    # t_chi_arr=np.append(t_chi_arr,t_chi)
                    print('chidel_t =',chi2t_t-chi0)
                    print('chi2t-chi0=',chi2t_t-chi0)
                    print('t_chi =',t_chi)
                else:
                    t_chi=tol
                        
                    
                    chiarr=np.append(chiarr,chi2t-chi0)
                    if delp:
                        chiarr=np.append(chiarr,delpar[j])
                    else:
                        chiarr=np.append(chiarr,i*tol)
                        chiarr=np.append(chiarr,lam[j])
#                        chiarr=np.append(chiarr,jaci_d[j])


                    # chidel=chi2t-chi0
                    # chidel=chidel
                    # print('chidel = ',chidel,chi0,chi2t)
                    
                    
                chiarr=np.delete(chiarr,0)
                    # print('chis = ',chiarr)
                    # chiarrn=chiarr-chi0
                
                with open(output,'a') as outputfile:
                    if j==0 and i==-imax:
                        outputfile.write('chi0 = ')
                        outputfile.write(str(chi0))
                        outputfile.write('\n')
                        outputfile.write('Delta chi^2,t')
                        outputfile.write('\n')

                    np.savetxt(outputfile,chiarr[:],fmt="%.7E",delimiter=' ', newline=' ')
                    outputfile.write(' ')
                    outputfile.write(str(brute))
                    outputfile.write('\n')

                if pdfout:
                    evgrido()
                    gridout()
                    fit_pars.irep=fit_pars.irep+1

#    chi2t=chi2min(afi)

#    os.quit()



    