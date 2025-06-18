from pathlib import Path

from fixparpdf.chi2s import *
from fixparpdf.global_pars import *
from fixparpdf.outputs import *


class _BufferLog:

    def __init__(self, outfile):
        self._outfile = outfile.open("a")

    def write(self, msg):
        # TODO: to be changed by an actual loggerr
        msg = msg.strip()
        print(msg)
        self._outfile.write(f"{msg}\n")

    def __del__(self):
        self._outfile.close()

def corrmatcalc(hessi,afin):

    corrmat=hessi.copy()/2.
    corrmati=hessi.copy()/2.
    
    dimh=len(hessi)

    for i in range(0,dimh):
        for j in range(0,dimh):
            corrmat[i,j]=corrmat[i,j]/np.sqrt(corrmati[i,i]*corrmati[j,j])
            if np.abs(corrmat[i,j]) > 0.95:
                if i != j:
                    print('i,j,corr > 0.95, = ',i,j,corrmat[i,j],afin[i],afin[j])
           
    os.quit()
         
    # output='outputs/corr/'+label+'.dat'
    # with open(output,'w') as outputfile:
    #     for i in range(0,pdf_pars.npar_free):
    #         np.savetxt(outputfile,corrmat[i,:],fmt="%.7E",delimiter=' ', newline=' ')
    #         outputfile.write('\n')

    return corrmat

def levmar_meth2(afree):

    af=afree.copy()
    jac_calc=True
    hess_calc=True
    chi2i, jaci, hessi, _, _ =chi2min_fun(af,jac_calc,hess_calc)
    hess=hessi.copy()/2.
    jac=-jaci.copy()/2.

    lam0=1e-4
    lam0=np.max(hess.diagonal())*lam0
    lam=lam0

    nsteps=30
    ntries=500
    tol=0.02

    outputfile=open('outputs/buffer/'+inout_pars.label+'.dat','a')

    outputfile.write("LM meth 2 ")
    outputfile.write("\n")

    for nt in range(1,ntries):

        print('ntries = ',nt)
        outputfile.write("ntries = ")
        outputfile.write(str(nt))
        outputfile.write("\n")


        if nt > 1:
            if((chi2i-chi2o) < tol):
                print('chi2i - chi2o < tol : exit')
                outputfile.write("chi2i - chi2o < tol : exit ")
                outputfile.write("\n")
                # covmatout(hessi/2.) doesn't work at mo for this levmar
                break


        jac_calc=True
        hess_calc=True
        (chi2i,jaci,hessi,err,hessp)=chi2min_fun(af,jac_calc,hess_calc)
        hess=hessi.copy()/2.
        jac=-jaci.copy()/2.

        outputfile=open('outputs/buffer/'+inout_pars.label+'.dat','a')
        outputfile.write("chi2i = ")
        outputfile.write(str(chi2i))
        outputfile.write("\n")

        nu=2.
        if nt > 1:
            lammax=1./(1.+alpha)
            lam0=np.max([lammax,1e-7])
            print('lammax=',lammax,lam0)
            lam=lam*lam0

        for ns in range(1,nsteps+1):

            hess=hess+np.diag(np.ones((pdf_pars.npar_free)))*lam
            hessin=la.inv(hess)
            delpar=hessin@jac

            aft=af.copy()+delpar
            hess_calc=False
            jac_calc=False
            (chi2o,jaco,hesso,erro,hesspo)=chi2min_fun(aft,jac_calc,hess_calc)

            delchi=chi2i-chi2o
            alpha=jac@delpar/(delchi/2.+2.*jac@delpar)

            aft_al=af.copy()+delpar*alpha
            hess_calc=False
            jac_calc=False
            (chi2o,jaco,hesso,erro,hesspo)=chi2min_fun(aft_al,jac_calc,hess_calc)
            delchi=chi2i-chi2o

            rho=delchi/np.abs(lam*delpar@delpar*np.power(alpha,2)+jac@delpar*alpha)

            print('lam = ',lam)
            print('alpha =',alpha)
            print('nstep = ',ns)
            print('chi2i = ',chi2i)
            print('chi2o = ',chi2o)
            print('chi2o - chi2i = ',chi2o-chi2i)
            print('rho = ',rho)

            if rho > tol:
                print('rho =', rho,' > tol')
                outputfile.write("rho = ")
                outputfile.write(str(rho))
                outputfile.write("\n")
                af=aft_al.copy()
                print('pars out =',aft_al)
                break
                
            lam=lam+np.abs(delchi)/2./alpha

    afout=aft.copy()
    return afout

def levmar_meth3(afree):

    print('pars in =',afree)

    af=afree.copy()
    jac_calc=True
    hess_calc=True
    (chi2i,jaci,hessi,err,hessp)=chi2min_fun(af,jac_calc,hess_calc)
    hess=hessi.copy()/2.
    jac=-jaci.copy()/2.

    lam0=1e-4
    lam0=np.max(hess.diagonal())*lam0
    lam=lam0
    nu=2.
    
    nsteps=30
    ntries=5000
    tol=0.01
    tolstop=tol
    tol=0.75

    outputfile=open('outputs/buffer/'+inout_pars.label+'.dat','a')

    outputfile.write("LM meth 3 ")
    outputfile.write("\n")

    for nt in range(1,ntries):

        print('ntries = ',nt)
        outputfile.write("ntries = ")
        outputfile.write(str(nt))
        outputfile.write("\n")

        if nt > 1:
            if((chi2i-chi2o) < tolstop):
                print('chi2i - chi2o < tol : exit')
                outputfile.write("chi2i - chi2o < tol : exit ")
                outputfile.write("\n")
                jac_calc=True
                hess_calc=True
                chi2_pars.add_hessp=False
                (chi2i,jaci,hessi,err,hessp)=chi2min_fun(af,jac_calc,hess_calc)
                hess=hessi.copy()/2.
                jac=-jaci.copy()/2.
                covmatout(hess,jac)
                break


        jac_calc=True
        hess_calc=True
        dload_pars.dcov=1
        (chi2i,jaci,hessi,err,hessp)=chi2min_fun(af,jac_calc,hess_calc)
        hess=hessi.copy()/2.
        jac=-jaci.copy()/2.


        jmax=np.max(jac)
        print('jmax =',jmax)

        outputfile=open('outputs/buffer/'+inout_pars.label+'.dat','a')
        outputfile.write("chi2i = ")
        outputfile.write(str(chi2i))
        outputfile.write("\n")

        nu=2.
        if nt > 1:
            lammax=1.-np.power(2.*rho-1.,3)
            lam0=np.max([lammax,1./3.])
            print('lammax=',lammax,lam0)
            lam=lam*lam0

        for ns in range(1,nsteps+1):
            
            hess=hessi.copy()/2.
            
            hess=hess+np.diag(np.ones((pdf_pars.npar_free)))*lam
            hessin=la.inv(hess)
            delpar=hessin@jac
            
            aft=af.copy()+delpar  
            hess_calc=False
            jac_calc=False
            dload_pars.dcov=1
            (chi2o,jaco,hesso,erro,hesspo)=chi2min_fun(aft,jac_calc,hess_calc)
            
            # alpha0=jac@delpar
            # alpha=alpha0/((chi2o-chi2i)/2.+2.*alpha0)
            
            delchi=chi2i-chi2o
            rho=delchi/np.abs(lam*delpar@delpar+jac@delpar)

            # print('alpha=',rho,chi2o,chi2i,delchi,lam)
            print('lam = ',lam)
            print('nstep = ',ns)
            print('chi2i = ',chi2i)
            print('chi2o = ',chi2o)
            print('chi2o - chi2i = ',chi2o-chi2i)
            print('rho = ',rho)

            outputfile=open('outputs/buffer/'+inout_pars.label+'.dat','a')
            outputfile.write("lam = ")
            outputfile.write(str(lam))
            outputfile.write("\n")
            outputfile.write("chi2o = ")
            outputfile.write(str(chi2o))
            outputfile.write("\n")
#            outputfile.write("chi2i = ")
#            outputfile.write(str(chi2i))
#            outputfile.write("\n")


            if rho > tol:
                print('rho =', rho,' > tol')
                outputfile.write("rho = ")
                outputfile.write(str(rho))
                outputfile.write("\n")
                af=aft.copy()
                parsout()
                print('pars out =',aft)
                break

            lam=lam*nu
            nu=nu*2
    
    afout=aft.copy()        

    return afout

def levmar(afree):
    """
    Levenberg–Marquardt algorithm, takes the set of free parameters
    and runs the minimization.

    Returns the same array with the minimized parameters.
    """

    lev_comb=False
    lev_update=True
    lev_minpack=False
    if(lev_comb):
        lev_update=True
    rhocon=False
    levmsht=False
    nsteps=30
    ntries=1000
    tol=min_pars.tollm

    nstepmax=False
    af=afree.copy()

    lam=0.001
    lam_initial = lam

    del_grat=False
    pos_hess=False

    div=10.

    mult=div

    lmin=1e-10
    
    hessmax=np.zeros((pdf_pars.npar_free,pdf_pars.npar_free))

    # TODO: a context manager would be good here
    bufferfile = Path("outputs") / "buffer" / f"{inout_pars.label}.dat"
    _bufferlog = _BufferLog(bufferfile)
    _bufferlog.write(f"""Levmar options:
    {levmsht=}
    {lev_update=}
    {lev_minpack=}
    {pos_hess=}
    {lev_comb=}
    sgd={min_pars.sgd}
LM meth 1
""")
    
    if del_grat:
        nsteps=35

    if min_pars.sgd:
        nsteps=100
        ntries=20

    # Do ntries without positivity,
    nt = 0
    chi2i = 1e10 
    chi2o = 1e10
    # if positivity is to be activated, do so after we would've finished
    # if and only if we finished due to tolerance
    fit_pars.pos_const = False

    while nt < ntries:
        nt += 1

        if nt > 1:
            lam=lam/div

        if lam < lmin:
            lam=lmin
        if lam > lam_initial:
            lam = lam_initial
        
        _bufferlog.write(f'ntries = {nt}')

        if(nstepmax):
            _bufferlog.write('max steps reached: exit')
            hess=hessi.copy()/2.
            covmatout(hess,jac)
            break

        if chi2o < 1e4 and not fit_pars.pos_const: # approx. chi2 ~ 2.5
            # Cheap way of implementing positivity
            fit_pars.pos_const = fit_pars.nnpdf_pos

        if nt > 1 and (abs(chi2i - chi2o) < tol):
            _bufferlog.write(f"Tolerance reached: {chi2i=:.5} - {chi2o:.5} < {tol}")
            # Check whether we are leaving
            # Do we need to start the positivity scan?
            if fit_pars.nnpdf_pos and not fit_pars.pos_const:
                _bufferlog.write("########### Restart with positivity")
                fit_pars.pos_const = True
                # Set to 1 and go back to the top of the loop
                nt = 0
                lam = lam_initial
                continue
            _bufferlog.write("Exit!")
            jac_calc=True
            hess_calc=True
            print('running chi2min_fun')
            dload_pars.dcov=1
            chi2_pars.add_hessp=False
            (chi2i,jaci,hessi,_,hessp)=chi2min_fun(af,jac_calc,hess_calc)
            print('run chi2min_fun')
            hess=hessi.copy()/2.
            jac=-jaci.copy()/2.
            covmatout(hess,jac) 
            break

        hess_calc=True
        jac_calc=True
        dload_pars.dcov=1

        # Compute the chi2 before the minimization and the jacobian / hessian
        chi2i, jac, hessi, _, hessp = chi2min_fun(af,jac_calc,hess_calc)

        # Divide the jacobian by 2 and change sign
        jac /= -2.0

        print('chi2i = ',chi2i)
        print('jac =', jac)
        print('hess =',hessi/2.)
        
        if nt == 1:
            if not lev_update:
                hess=hessi.copy()/2.
                lammax0=np.max(hess.diagonal())
                lam=lam*lammax0
                lmin=lam/1e10

        _bufferlog.write(f"chi2i = {chi2i}")

        for ns in range(1,nsteps+1):
            
            _bufferlog.write(f"step = {ns} ({lam=})\n")

            hess = hessi/2.

            if min_pars.sgd:
                if(lev_update):
                    hess=np.diag(hess.diagonal())*lam  # updated relationship                                                   
                else:
                    hess=np.diag(np.ones((pdf_pars.npar_free)))*lam

                hessin=la.inv(hess)

                print('hess =', hess)
                
                delpar=hessin@jac

                rhoden=lam*delpar@np.diag(hess.diagonal())@delpar+delpar@jac                
                
            elif not levmsht:

                if lev_update:

                    if(lev_minpack):
                        for i in range(0,len(hess)):
                            hessmax0=hess[i,i]
                            hessmax[i,i]=np.max([hessmax[i,i],hessmax0])
                            hess=hess+lam*hessmax
                    else:
                        hess += np.diag(hess.diagonal()*lam)
                else:
                    if(pos_hess):
                        hess=hess+hessp*lam
                    else:
                        hess += np.eye(pdf_pars.npar_free)*lam

                if(lev_comb):
                    hess0=hessi.copy()/2.+np.diag(np.ones((pdf_pars.npar_free)))*lam 
                    hess0in=la.inv(hess0)
                    print('hess0 =',hess0)
                    delpar0=hess0in@jac
                    corrmat=corrmatcalc(hess0,af)
                    print('corr mat 0 =',corrmat)
                    
                try:
                    hessin = la.inv(hess)
                except la.LinAlgError as error:
                    print(error)
                    hessin = hess_zeros(hess)
                    parsout()
                
                delpar = hessin@jac
                print('delpar = ', delpar)

                #rhoden = lam*delpar@np.diag(hess.diagonal())@delpar+delpar@jac
                rhoden = lam*np.sum(delpar**2 @ hess.diagonal()) + delpar@jac
            
            elif levmsht:
                hesst=hessi.copy()/2
                hesstt=hessi.copy()/2.                                       
                for i in range(0,pdf_pars.npar_free):
                    for j in range(0,pdf_pars.npar_free):
                        if i==j: 
                            hesst[i,j]=1.+lam
                        else: 
                            hesst[i,j]=hesst[i,j]/np.sqrt(hesstt[i,i]*hesstt[j,j])    
                try:
                    hesstin=la.inv(hesst)
                except la.LinAlgError as error:
                    print(error)
                    hesstin=hess_zeros(hesst)
            
                for i in range(0,pdf_pars.npar_free):                        
                    for j in range(0,pdf_pars.npar_free):                              
                        hesstin[i,j]=hesstin[i,j]/np.sqrt(hesstt[i,i]*hesstt[j,j])       
                delpar=jac@hesstin  

            # Update parameters:
            aft = af + delpar
            if(lev_comb):
                aft0 = af + delpar0

            _bufferlog.write(f"pars after step: {aft}")
            
            hess_calc=False
            jac_calc=False
            dload_pars.dcov=1
            
            chi2o, _, _, erro, _ =  chi2min_fun(aft,jac_calc,hess_calc)
    
            if(lev_comb):
                chi2o0, *_ = chi2min_fun(aft0,jac_calc,hess_calc)
                print('chi2o0 - chi2i = ',chi2o0-chi2i)        
                _bufferlog.write(f"chi2o0 = {chi2o0}")

                print('delpar=',delpar)
                print('delpar0=',delpar0)
                print('chi2o =',chi2o)
                print('chi2o0=',chi2o0)

            print('nstep = ',ns)

            if(erro):
                print('PARCHECK FAIL: unstable region of parameter space - set chi2 = 1e9')
            else:
                print('chi2o,chi2i = ',chi2o,chi2i)
                print('chi2o - chi2i = ',chi2o-chi2i)
                _bufferlog.write(f"{chi2o=}")
                _bufferlog.write(f"{chi2i=}")
  
            if(lev_comb):
                chi2o=min(chi2o0,chi2o)
            
            delchi=chi2i-chi2o

            if rhocon:
                eps_rho=0.01*rhoden
                print('rhoden = ',rhoden)
                print('eps_rho =',eps_rho)
            else:
                eps_rho=0.
                
            if delchi > eps_rho:
                af=aft.copy()
                parsout()
                _bufferlog.write("chi2o < chi2i - next iteration")
                print('new pars = ',af)
                break           

            if ns==nsteps:
                nstepmax=True
                break
                
            if(del_grat):
                lam=lam*3.
            else:
                lam=lam*mult
                
    afout=aft

    return afout
