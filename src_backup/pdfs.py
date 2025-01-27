from global_pars import *
from chebyshevs import *
import numpy as np
import os
from scipy.integrate import quadrature
from scipy.misc import derivative as deriv

def func_pdfs_diff(eps,ipdf,x,iorder):

    # print(eps)

    if eps == 0.:
        out=pdfs_msht(ipdf,pdf_pars.parinarr[0,:],x)
        if pdf_pars.parin_newmin_reset:
            pdf_pars.parinarr_newmin[pdf_pars.parin_newmin_counter,:]=pdf_pars.parinarr[0,:]
    else:
        if pdf_pars.parin_newmin_reset:
            (pars,eps_out)=parinc_newmin(pdf_pars.parinarr[0,:],chi2_pars.ipdf_newmin-1,eps)
            pdf_pars.parinarr_newmin[pdf_pars.parin_newmin_counter,:]=pars
        else:
            pars=pdf_pars.parinarr_newmin[pdf_pars.parin_newmin_counter,:]
        out=pdfs_msht(ipdf,pars,x)

    pdf_pars.parin_newmin_counter+=1

    return out

def pdfs_diff(ipdf,x):

    eps=1e-5
    # (pars,eps_out)=parinc_newmin(pdf_pars.parinarr[0,:],chi2_pars.ipdf_newmin-1,eps)
    eps=parinc_eps(pdf_pars.parinarr[0,:],chi2_pars.ipdf_newmin-1,eps)

    # # pdfout=pdfs_msht(ipdf,pars,x)-pdfs_msht(ipdf,pdf_pars.parinarr[0,:],x)
    # pdfout=func_pdfs_diff(eps,ipdf,x)-func_pdfs_diff(0.,ipdf,x)
    # # pdfout/=chi2_pars.eps_arr_newmin[chi2_pars.ipdf_newmin]
    # pdfout/=eps

    # derivative(func, x0, dx=1.0

    pdf_pars.parin_newmin_counter=0
    iorder=5
    pdfout=deriv(func_pdfs_diff,0.,eps,args=(ipdf,x,iorder),order=iorder)
    pdf_pars.parin_newmin_reset=False


    # print(pdfout,test)
    # os.quit()

    return pdfout

def pdfs_msht(ipdf,pars,x):

    
    etaq=pars[57]


    if etaq < 0.0:
        etaqt=np.power(1.-x,-etaq)
    else:
        etaqt=1.

        

    if ipdf==-3:
        sm=pdfs_msht_basis(6,pars,x)
        sp=pdfs_msht_basis(4,pars,x)
        out=(sp-sm)/2.
    elif ipdf==-2:
        sp=pdfs_msht_basis(4,pars,x)
        sea=pdfs_msht_basis(3,pars,x)
        dbpub=(sea-sp)/2.
        if etaqt < 1e-20:
            out=0.0
        else:
            dbdub=pdfs_msht_basis(7,pars,x)        
            out=dbpub/(1+dbdub)
    elif ipdf==-1:
        sp=pdfs_msht_basis(4,pars,x)
        sea=pdfs_msht_basis(3,pars,x)
        dbpub=(sea-sp)/2.
        if etaqt < 1e-20:
            out=dbpub
        else:
            dbdub=pdfs_msht_basis(7,pars,x)
            out=dbpub/(1+dbdub)*dbdub
    elif ipdf==0:
        out=pdfs_msht_basis(5,pars,x)
    elif ipdf==1:
        sp=pdfs_msht_basis(4,pars,x)
        sea=pdfs_msht_basis(3,pars,x)
        dbpub=(sea-sp)/2.
        dv=pdfs_msht_basis(2,pars,x)
        if etaqt < 1e-20:
            db=dbpub
        else:
            dbdub=pdfs_msht_basis(7,pars,x)
            db=dbpub/(1+dbdub)*dbdub
        out=dv+db
    elif ipdf==2:
        sp=pdfs_msht_basis(4,pars,x)
        sea=pdfs_msht_basis(3,pars,x)
        uv=pdfs_msht_basis(1,pars,x)
        dbpub=(sea-sp)/2.
        if etaqt < 1e-20:
            ub=0.
        else:
            dbdub=pdfs_msht_basis(7,pars,x)
            ub=dbpub/(1+dbdub)
        out=uv+ub
    elif ipdf==3:
        sm=pdfs_msht_basis(6,pars,x)
        sp=pdfs_msht_basis(4,pars,x)
        out=(sp+sm)/2.
    elif ipdf==-4:
        out=pdfs_msht_basis(8,pars,x)/2.
    elif ipdf==4:
        out=pdfs_msht_basis(8,pars,x)/2.
    else:
        out=0.
        
    return out
        
def pdfs_msht_basis(ipdf,pars,x):

    if ipdf==1:
        # auv=pars[0:9].copy()
        auv=pars[basis_pars.i_uv_min:basis_pars.i_uv_max].copy()
        out=q_msht(x,auv)
    if ipdf==2:
        # adv=pars[9:18]
        adv=pars[basis_pars.i_dv_min:basis_pars.i_dv_max].copy()
        out=q_msht(x,adv)
    if ipdf==3:
        # asea=pars[18:27]
        asea=pars[basis_pars.i_sea_min:basis_pars.i_sea_max].copy()
        out=q_msht(x,asea)
    if ipdf==4:
        # asp=pars[27:36]
        asp=pars[basis_pars.i_sp_min:basis_pars.i_sp_max].copy()
        out=q_msht(x,asp)
    if ipdf==5:
        # ag=pars[36:46]
        ag=pars[basis_pars.i_g_min:basis_pars.i_g_max].copy()
        out=g_msht(x,ag)
    if ipdf==6:
        # asm=pars[46:56]
        asm=pars[basis_pars.i_sm_min:basis_pars.i_sm_max].copy()
        out=sm_msht(x,asm)
    if ipdf==7:
        # adbub=pars[56:64]
        adbub=pars[basis_pars.i_dbub_min:basis_pars.i_dbub_max].copy()
        out=dbub_msht(x,adbub)
    if ipdf==8:
        # fitcharm=pars[64:73]
        fitcharm=pars[basis_pars.i_ch_min:basis_pars.i_ch_max].copy()
        out=q_msht(x,fitcharm)

    return out

def sm_msht(x,ain):
    asm=ain[0]
    delsm=ain[1]
    etasm=ain[2]
    x0=ain[3]

    aqc1=ain[4]
    aqc2=ain[5]
    aqc3=ain[6]
    aqc4=ain[7]
    aqc5=ain[8]
    aqc6=ain[9]
    
    out=asm*np.power(1-x,etasm)*np.power(x,delsm)*(1-x/x0)
    out=out*(1.+aqc1*cheb_msht(1,x)+aqc2*cheb_msht(2,x)+aqc3*cheb_msht(3,x)+aqc4*cheb_msht(4,x)+aqc5*cheb_msht(5,x)+aqc6*cheb_msht(6,x))

    return out

def sp_norm_fix(asea,ain):

    aq=ain[0]
    delq=ain[1]
    etaq=ain[2]
    aqc1=ain[3]
    aqc2=ain[4]
    aqc3=ain[5]
    aqc4=ain[6]
    aqc5=ain[7]
    aqc6=ain[8] 

    norm_cheb=(1.+aqc1+aqc2+aqc3+aqc4+aqc5+aqc6)

    if basis_pars.Cheb_8:
        aqc7=ain[9]
        aqc8=ain[10]
        norm_cheb+=aqc7+aqc8
    
    out=asea/norm_cheb/3.

    return out



def Phi_msht_int():

    xmin=1e-6
    xmax=0.5


    if fit_pars.theoryidi==211 or fit_pars.theoryidi==40001000:
        pdfmax=8    
    else: 
        pdfmax=9

    if fit_pars.theoryidi==211 or fit_pars.theoryidi==40001000:
        pdfmax=3    
    else: 
        pdfmax=4

    out=[]

    # for pdfi in range (1,pdfmax):
    #     print(pdfi)
    #     outi=quadrature(arclength_msht,xmin,xmax,args=(pdfi,),rtol=1.0e-04,maxiter=5000)[0]
    #     out=np.append(outi,out)

    for pdfi in range (-pdfmax,pdfmax+1):
        print(pdfi)
        outi=quadrature(arclength_msht,xmin,xmax,args=(pdfi,),rtol=1.0e-04,maxiter=5000)[0]
        out=np.append(outi,out)

    out=np.flip(out)

    return out

def Phi_msht_diff(x,pdfi):

    b=10.
    c=-0.5

    eps=x*1e-3

    xp=x+eps
    xm=x-eps
    x2p=x+2.*eps
    x2m=x-2.*eps

    phi_plus=Phi_msht(xp,pdfi)
    phi_minus=Phi_msht(xm,pdfi)
    phi_0=Phi_msht(x,pdfi)

    # xarr=np.array([x2m,xm,x,xp,x2p])
    # pdfarr=[]
    # for i in range (0,len(xarr)):
    #     pdfarr=np.append(Phi_msht(xarr[i],pdfi),pdfarr)


    # out=diff2_5point(pdfarr,xarr,eps)
    # out=np.power(out,2)

    # print(out)

    out=(phi_plus+phi_minus-2.*phi_0)/np.power(eps,2)
    out=np.power(out,2)

    # print('test')
    # print(x,out)

    return out

def pdf_test(x):
    a=1.
    b=10.
    c=-0.5

    out=a*np.power(x,c)*np.power(1.-x,b)

    return out

def diff_2point(f,x,eps):

    out=f[1]-f[0]
    out=out/2./eps

    return out

def diff2_5point(f,x,eps):

    out=-f[0]+16.*f[1]-30.*f[2]+16.*f[3]-f[4]
    out=out/12./np.power(eps,2)

    return out

def diff_5point(f,x,eps):

    out=f[0]-8.*f[1]+8.*f[2]-f[3]
    out=out/12./eps

    return out

def arclength_msht(x,pdfi):

    eps=x*1e-3
    xp=x+eps
    xm=x-eps

    # pdf0=pdfs_msht_basis(pdfi,pdf_pars.pdfparsi,x)
    # pdfm=pdfs_msht_basis(pdfi,pdf_pars.pdfparsi,xm)
    # pdfp=pdfs_msht_basis(pdfi,pdf_pars.pdfparsi,xp)

    pdf0=pdfs_msht(pdfi,pdf_pars.pdfparsi,x)
    pdfm=pdfs_msht(pdfi,pdf_pars.pdfparsi,xm)
    pdfp=pdfs_msht(pdfi,pdf_pars.pdfparsi,xp)

    out=(pdfp-pdfm)/2./eps
    out=np.power(out,2)
    out=np.sqrt(1.+out)

    return out


def Phi_msht(x,pdfi):

    eps=x*1e-3
    xp=x+eps
    xm=x-eps
    x2p=x+2.*eps
    x2m=x-2.*eps

    pdf0=pdfs_msht_basis(pdfi,pdf_pars.pdfparsi,x)
    pdfm=pdfs_msht_basis(pdfi,pdf_pars.pdfparsi,xm)
    pdfp=pdfs_msht_basis(pdfi,pdf_pars.pdfparsi,xp)

 

    # pdfm=pdf_test(xm)
    # pdfp=pdf_test(xp)

    xarr=np.array([x2m,xm,xp,x2p])
    # pdfarr=np.log(pdf_test(xarr))
    # pdfarr=[]
    # for i in range (0,len(xarr)):
    #     pdfarr=np.append(np.log(pdfs_msht_basis(pdfi,pdf_pars.pdfparsi,xarr[i])),pdfarr)
    # pdfarr=np.flip(pdfarr)

    # out=diff_5point(pdfarr,xarr,eps)

    out=(pdfp-pdfm)/2./eps/pdf0

    out=out*x*(1.-x)

    return out

def t8_msht(x):

    sbar=pdfs_msht(-3,pdf_pars.pdfparsi,x)
    s=pdfs_msht(3,pdf_pars.pdfparsi,x)
    ubar=pdfs_msht(-2,pdf_pars.pdfparsi,x)
    u=pdfs_msht(2,pdf_pars.pdfparsi,x)
    dbar=pdfs_msht(-1,pdf_pars.pdfparsi,x)
    d=pdfs_msht(1,pdf_pars.pdfparsi,x)

    out=u+ubar+d+dbar-2.*(s+sbar)

    return out

def q_msht_lowx_norm(ain):

    aq=ain[0]
    delq=ain[1]
    etaq=ain[2]
    aqc1=ain[3]
    aqc2=ain[4]
    aqc3=ain[5]
    aqc4=ain[6]
    aqc5=ain[7]
    aqc6=ain[8]  

    out=aq*(1.+aqc1+aqc2+aqc3+aqc4+aqc5+aqc6)

    return out
    
def q_msht(x,ain):
    aq=ain[0]
    delq=ain[1]
    etaq=ain[2]
    aqc1=ain[3]
    aqc2=ain[4]
    aqc3=ain[5]
    aqc4=ain[6]
    aqc5=ain[7]
    aqc6=ain[8]

    if aq < 1e-20:
        out=aq
    else:   
        # print(aq,etaq,delq)
        out=aq*np.power(1.-x,etaq)*np.power(x,delq)*(1.+aqc1*cheb_msht(1,x)+aqc2*cheb_msht(2,x)+aqc3*cheb_msht(3,x)+aqc4*cheb_msht(4,x)+aqc5*cheb_msht(5,x)+aqc6*cheb_msht(6,x))

    if basis_pars.Cheb_8:
        aqc7=ain[9]
        aqc8=ain[10]
        out+=aq*np.power(1.-x,etaq)*np.power(x,delq)*(aqc7*cheb_msht(7,x)+aqc8*cheb_msht(8,x))

    return out

def lg1_msht(x,ain):
    agp=ain[0]
    etagp=ain[1]
    delgp=ain[2]
    agc1=ain[3]
    agc2=ain[4]
    agc3=ain[5]
    agc4=ain[6]
    agm=ain[7]
    etagm=ain[8]
    delgm=ain[9]
    
    x=np.exp(x)

    out=agp*np.power(1.-x,etagp)*np.power(x,delgp)*(1.+agc1*cheb_msht(1,x)+agc2*cheb_msht(2,x)+agc3*cheb_msht(3,x)+agc4*cheb_msht(4,x))

    if basis_pars.Cheb_8:
        agc5=ain[10]
        agc6=ain[11]
        out+=agp*np.power(1.-x,etagp)*np.power(x,delgp)*(agc5*cheb_msht(5,x)+agc6*cheb_msht(6,x))

    out=out*x
    
    return out

def lg2_msht(x,ain):
    agp=ain[0]
    etagp=ain[1]
    delgp=ain[2]
    agc1=ain[3]
    agc2=ain[4]
    agc3=ain[5]
    agc4=ain[6]
    agm=ain[7]
    etagm=ain[8]
    delgm=ain[9]

    x=np.exp(x)

    out=agm*np.power(1.-x,etagm)*np.power(x,delgm)
    out=out*x
    
    return out

def g_msht(x,ain):
    agp=ain[0]
    etagp=ain[1]
    delgp=ain[2]
    agc1=ain[3]
    agc2=ain[4]
    agc3=ain[5]
    agc4=ain[6]
    agm=ain[7]
    etagm=ain[8]
    delgm=ain[9]


    if(basis_pars.g_second_term):
        out=agp*np.power(1.-x,etagp)*np.power(x,delgp)*(1.+agc1*cheb_msht(1,x)+agc2*cheb_msht(2,x)+agc3*cheb_msht(3,x)+agc4*cheb_msht(4,x))
        out+=agm*np.power(1.-x,etagm)*np.power(x,delgm)
        if basis_pars.Cheb_8:
            agc5=ain[10]
            agc6=ain[11]
            out+=agp*np.power(1.-x,etagp)*np.power(x,delgp)*(agc5*cheb_msht(5,x)+agc6*cheb_msht(6,x))
    else:
        out=agp*np.power(1.-x,etagp)*np.power(x,delgp)*(1.+agc1*cheb_msht(1,x)+agc2*cheb_msht(2,x)+agc3*cheb_msht(3,x)+agc4*cheb_msht(4,x)+etagm*cheb_msht(5,x)+delgm*cheb_msht(6,x))
        if(basis_pars.g_cheb7):
            out=out+agp*agm*cheb_msht(7,x)
        if basis_pars.Cheb_8:
            agc7=ain[10]
            agc8=ain[11]
            out+=agp*np.power(1.-x,etagp)*np.power(x,delgp)*(agc7*cheb_msht(7,x)+agc8*cheb_msht(8,x))
    return out


def dbub_msht(xin,ain):
    aq=ain[0]
    etaq=ain[1]
    aqc1=ain[2]
    aqc2=ain[3]
    aqc3=ain[4]
    aqc4=ain[5]
    aqc5=ain[6]
    aqc6=ain[7]

    x=xin
    if etaq < 0. and xin > 0.999:
        x=0.999
    
    out=aq*np.power(1.-x,etaq)*(1.+aqc1*cheb_msht(1,x)+aqc2*cheb_msht(2,x)+aqc3*cheb_msht(3,x)+aqc4*cheb_msht(4,x)+aqc5*cheb_msht(5,x)+aqc6*cheb_msht(6,x))


    if basis_pars.Cheb_8:
        aqc7=ain[8]
        aqc8=ain[9]
        out+=aq*np.power(1.-x,etaq)*(aqc7*cheb_msht(7,x)+aqc8*cheb_msht(8,x))

    return out

def cheb_msht(i,x):
    y=1.-2.*np.sqrt(x)
    
    if i==1:
        out=y
    elif i==2:
        out=2.*np.power(y,2)-1.
    elif i==3:
        out=4.*np.power(y,3)-3.*y
    elif i==4:
        out=8.*np.power(y,4)-8.*np.power(y,2)+1.
    elif i==5:
        out=16.*np.power(y,5)-20*np.power(y,3)+5.*y
    elif i==6:
        out=32.*np.power(y,6)-48.*np.power(y,4)+18.*np.power(y,2)-1.
    elif i==7:
        out=64.*np.power(y,7)-112.*np.power(y,5)+56.*np.power(y,3)-7.*y
    elif i==8:
        out=128.*np.power(y,8)-256.*np.power(y,6)+160.*np.power(y,4)-32.*np.power(y,2)+1.
        
    return out

def smin_norm(asm):

#    xmin=1.0E-15
#    xmax=0.99
#    lxmin=np.log(xmin)
#    lxmax=np.log(xmax)

#    i1=quadrature(sm1_msht,lxmin,lxmax,args=(asm,),rtol=1.0e-04,maxiter=50)
#    int1=i1[0]
#    i2=quadrature(sm2_msht,lxmin,lxmax,args=(asm,),rtol=1.0e-04,maxiter=50)
#    int2=i2[0]

    int1=sm1_int(asm,0.)
    int2=sm2_int(asm,0.)

#    print('test',i1[0],i2[0],i1t,i2t)
    
    out=int2/int1
    return out

def sm1_int(ain,xmin):

    asm=ain[0]
    delsm=ain[1]-1.
    etasm=ain[2]
    x0=ain[3]

    aqc1=ain[4]
    aqc2=ain[5]
    aqc3=ain[6]
    aqc4=ain[7]
    aqc5=ain[8]
    aqc6=ain[9]

    i1t=I(delsm,etasm,xmin)+aqc1*Ic1(delsm,etasm,xmin)+aqc2*Ic2(delsm,etasm,xmin)+aqc3*Ic3(delsm,etasm,xmin)+aqc4*Ic4(delsm,etasm,xmin)
    i1t=i1t+aqc5*Ic5(delsm,etasm,xmin)+aqc6*Ic6(delsm,etasm,xmin)
    i1t=i1t*asm

    return i1t

def sm2_int(ain,xmin):

    asm=ain[0]
    delsm=ain[1]
    etasm=ain[2]
    x0=ain[3]

    aqc1=ain[4]
    aqc2=ain[5]
    aqc3=ain[6]
    aqc4=ain[7]
    aqc5=ain[8]
    aqc6=ain[9]

    i1t=I(delsm,etasm,xmin)+aqc1*Ic1(delsm,etasm,xmin)+aqc2*Ic2(delsm,etasm,xmin)+aqc3*Ic3(delsm,etasm,xmin)+aqc4*Ic4(delsm,etasm,xmin)
    i1t=i1t+aqc5*Ic5(delsm,etasm,xmin)+aqc6*Ic6(delsm,etasm,xmin)
    i1t=i1t*asm/x0


    return i1t

def qv_norm(iq,aq):

#    xmin=1.0E-9
#    xmax=0.99
#    lxmin=np.log(xmin)
#    lxmax=np.log(xmax)
#    i1t=quadrature(lq_msht,lxmin,lxmax,args=(aq,),rtol=1.0e-03,maxiter=50)
    
    i1=qv_int(aq,1,0.)

    
    if iq==1: #uv
        out=2./i1
    if iq==2: #dv
        out=1./i1
        
    return out
    
def qv_int(aq,iq,xmin):

    if iq==1:
        delq=aq[1]-1.
    else:
        delq=aq[1]
    etaq=aq[2]
    aqc1=aq[3]
    aqc2=aq[4]
    aqc3=aq[5]
    aqc4=aq[6]
    aqc5=aq[7]
    aqc6=aq[8]

    i1t=I(delq,etaq,xmin)+aqc1*Ic1(delq,etaq,xmin)+aqc2*Ic2(delq,etaq,xmin)+aqc3*Ic3(delq,etaq,xmin)+aqc4*Ic4(delq,etaq,xmin)
    i1t=i1t+aqc5*Ic5(delq,etaq,xmin)+aqc6*Ic6(delq,etaq,xmin)

    if basis_pars.Cheb_8:
        aqc7=aq[9]
        aqc8=aq[10]
        i1t+=aqc7*Ic7(delq,etaq,xmin)+aqc8*Ic8(delq,etaq,xmin)

    i1t=i1t*aq[0]

    return i1t

def msum_ag(pars):


#    xmin=1.0E-15
#    xmax=0.99
#    lxmin=np.log(xmin)
#    lxmax=np.log(xmax)
    
#    i1=quadrature(msum_pdf_ng,lxmin,lxmax,args=(pars,),rtol=1.0e-04,maxiter=50)
#    outng=i1[0]

#    ag=pars[36:46]
#    i2=quadrature(lg1_msht,lxmin,lxmax,args=(ag,),rtol=1.0e-04,maxiter=50)
#    outg1=i2[0]

#    i3=quadrature(lg2_msht,lxmin,lxmax,args=(ag,),rtol=1.0e-04,maxiter=50)
#    outg2=i3[0]
    
    # auv=pars[0:9]
    auv=pars[basis_pars.i_uv_min:basis_pars.i_uv_max].copy()
    # adv=pars[9:18]
    adv=pars[basis_pars.i_dv_min:basis_pars.i_dv_max].copy()
    # asea=pars[18:27]
    asea=pars[basis_pars.i_sea_min:basis_pars.i_sea_max].copy()
    # ag=pars[36:46]  
    ag=pars[basis_pars.i_g_min:basis_pars.i_g_max].copy()
    # fitcharm=pars[64:73]
    fitcharm=pars[basis_pars.i_ch_min:basis_pars.i_ch_max].copy()

    xmin=0.


    outng=qv_int(auv,2,xmin)+qv_int(adv,2,xmin)+qv_int(asea,2,xmin)+qv_int(fitcharm,2,xmin)

    

    # outng=0.66336
    # outng=0.67364

    # print(adv)
    # print(qv_int(adv,2,xmin))
    # exit()

    # pch cheb 4
    # outng=0.673202645641724
    # # pch cheb 5
    # outng=0.6764550686407598
    # # fch cheb 4   
    # outng=0.6071106839785037
    # # fch cheb 5    
    # outng=0.608941296908194

    outg1=int_g1_msht(ag,xmin)
    outg2=int_g2_msht(ag,xmin)


    # outng=auv[1]
 

    # print(qv_int(auv,2,xmin),qv_int(adv,2,xmin),qv_int(asea,2,xmin),qv_int(afitcharm,2,xmin))
    # print(outng)
    # exit()
    # os.quit()
    # msht
    #    outng=0.6634
    # nnpdf pch
    #    outng=0.6770523344876442

    #    outng=0.6732041038230732

    # outng=0.613133969446266

    # outng=0.6764846130843789

    out=1.-outng-outg2
    out=out/outg1

    return out

def int_g1_msht(ain,xmin):

    agp=ain[0]
    etagp=ain[1]
    delgp=ain[2]
    agc1=ain[3]
    agc2=ain[4]
    agc3=ain[5]
    agc4=ain[6]
    agm=ain[7]
    etagm=ain[8]
    delgm=ain[9]

    if(basis_pars.g_second_term):
        out=I(delgp,etagp,xmin)+agc1*Ic1(delgp,etagp,xmin)+agc2*Ic2(delgp,etagp,xmin)+agc3*Ic3(delgp,etagp,xmin)+agc4*Ic4(delgp,etagp,xmin)
        if basis_pars.Cheb_8:
            agc5=ain[10]
            agc6=ain[11]
            out+=agc5*Ic5(delgp,etagp,xmin)+agc6*Ic6(delgp,etagp,xmin)
    else:
        out=I(delgp,etagp,xmin)+agc1*Ic1(delgp,etagp,xmin)+agc2*Ic2(delgp,etagp,xmin)+agc3*Ic3(delgp,etagp,xmin)+agc4*Ic4(delgp,etagp,xmin)+etagm*Ic5(delgp,etagp,xmin)+delgm*Ic6(delgp,etagp,xmin)
        if(basis_pars.g_cheb7):
            out=out+agm*Ic7(delgp,etagp,xmin)
        if basis_pars.Cheb_8:
            agc7=ain[10]
            agc8=ain[11]
            out+=agc7*Ic7(delgp,etagp,xmin)+agc8*Ic8(delgp,etagp,xmin)

    out=out*agp

    return out

def int_g2_msht(ain,xmin):

    agp=ain[0]
    etagp=ain[1]
    delgp=ain[2]
    agc1=ain[3]
    agc2=ain[4]
    agc3=ain[5]
    agc4=ain[6]
    agm=ain[7]
    etagm=ain[8]
    delgm=ain[9]

    
    if(basis_pars.g_second_term):
        out=agm*I(delgm,etagm,xmin)
    else:
        out=0.


    return out



def sumrules(parin):
    
    out=parin.copy()

    # asm=out[46:56]
    asm=out[basis_pars.i_sm_min:basis_pars.i_sm_max].copy()

    if basis_pars.dvd_eq_uvd:
        # out[10]=out[1]
        out[basis_pars.i_dv_min+1]=out[basis_pars.i_uv_min+1]
    if basis_pars.asp_fix:
        # out[28]=out[19]
        out[basis_pars.i_sp_min+1]=out[basis_pars.i_sea_min+1]


    if basis_pars.t8_int:
        # asea=out[18:27].copy()
        asea=out[basis_pars.i_sea_min:basis_pars.i_sea_max].copy()
        # asp=out[27:36].copy()
        asp=out[basis_pars.i_sp_min:basis_pars.i_sp_max].copy()
        norm_sea=q_msht_lowx_norm(asea)  
        asp_new=sp_norm_fix(norm_sea,asp)
        # out[27]=asp_new
        out[basis_pars.i_sp_min]=asp_new

    # auv=out[0:9].copy()
    auv=out[basis_pars.i_uv_min:basis_pars.i_uv_max].copy()
    # adv=out[9:18].copy()
    adv=out[basis_pars.i_dv_min:basis_pars.i_dv_max].copy()

    x0=smin_norm(asm)

    # out[49]=x0
    out[basis_pars.i_sm_min+3]=x0

    out[0]=qv_norm(1,auv)
    # print('out0 =',out[0])
    # out[9]=qv_norm(2,adv)
    out[basis_pars.i_dv_min]=qv_norm(2,adv)
    
    ag=msum_ag(out)

    # out[36]=ag
    out[basis_pars.i_g_min]=ag


    # xmin=1.0E-50
    # xmax=1.0E-12
    # xmin=1.0E-9
    # xmax=0.999
    # lxmin=np.log(xmin)
    # lxmax=np.log(xmax)


    # agpar=out[basis_pars.i_g_min:basis_pars.i_g_max]
    # i2=quadrature(lg1_msht,lxmin,lxmax,args=(agpar,),rtol=1.0e-04,maxiter=50)
    # outg1=i2[0]

    # i3=quadrature(lg2_msht,lxmin,lxmax,args=(agpar,),rtol=1.0e-04,maxiter=50)
    # outg2=i3[0]

    # print(outg1,outg2,outg1+outg2)

    # outg1=int_g1_msht(agpar,xmin)
    # outg2=int_g2_msht(agpar,xmin)

    # print(outg1,outg2,outg1+outg2)

    # exit()

    # agarr=out[36:46].copy()
    # test=g_msht(1e-1,agarr)

    # print('g(1e-1,1) = ',test)
    # exit()


    return out

def initpars():

    auv=uv_init()
    adv=dv_init()
    asea=sea_init()
    asp=sp_init()
    ag=g_init()
    asm=smin_init()
    adbub=dbub_init()
    fitcharm=fitcharm_init()


    # pdfpars=np.concatenate((auv,adv,asea,asp,ag,asm,adbub))  
    pdfpars=np.concatenate((auv,adv,asea,asp,ag,asm,adbub,fitcharm))  

    return pdfpars

def g_init():

    # a=pdf_pars.parsin[36:46]
    a=pdf_pars.parsin[basis_pars.i_g_min:basis_pars.i_g_max]
    a[0]=1. # Set to one so overwritten easily by sum rule   
    
    return a

def smin_init():

    # a=pdf_pars.parsin[46:56]
    a=pdf_pars.parsin[basis_pars.i_sm_min:basis_pars.i_sm_max]
    a[3]=1. # Set to one so overwritten easily by sum rule 
    
    return a

def uv_init():

    # a=pdf_pars.parsin[0:9] 
    a=pdf_pars.parsin[basis_pars.i_uv_min:basis_pars.i_uv_max]
    a[0]=1. # Set to one so overwritten easily by sum rule  

    return a

def dv_init():

    # a=pdf_pars.parsin[9:18]
    a=pdf_pars.parsin[basis_pars.i_dv_min:basis_pars.i_dv_max]
    a[0]=1. # Set to one so overwritten easily by sum rule  
    
    return a

def sp_init():

    # a=pdf_pars.parsin[27:36]
    a=pdf_pars.parsin[basis_pars.i_sp_min:basis_pars.i_sp_max]

    return a

    
def fitcharm_init():

    # a=pdf_pars.parsin[64:73]
    a=pdf_pars.parsin[basis_pars.i_ch_min:basis_pars.i_ch_max]

    return a

def sea_init():

    # a=pdf_pars.parsin[18:27]
    a=pdf_pars.parsin[basis_pars.i_sea_min:basis_pars.i_sea_max]

    return a

def dbub_init():

    # a=pdf_pars.parsin[56:64]
    a=pdf_pars.parsin[basis_pars.i_dbub_min:basis_pars.i_dbub_max]
    
    return a

def parset(af,parin):

    afi=af.copy()
    out=parin.copy()


    for i in range(1,basis_pars.n_pars):
        if pdf_pars.par_isf[i]:
            out[i]=afi[0]
            afi=np.delete(afi,0)
    


    return out

def parcheck(pars):

    err=False
    
    delgp=pars[basis_pars.i_g_min+2]
    delgm=pars[basis_pars.i_g_min+9]
    delsea=pars[basis_pars.i_sea_min+1]
    delsp=pars[basis_pars.i_sp_min+1]
    etagp=pars[basis_pars.i_g_min+1]
    etagm=pars[basis_pars.i_g_min+8]
    etasea=pars[basis_pars.i_sea_min+2]
    etasp=pars[basis_pars.i_sp_min+2]
    delu=pars[1]
    deld=pars[basis_pars.i_dv_min+1]
    delsm=pars[basis_pars.i_sm_min+1]
    etau=pars[2]
    etad=pars[basis_pars.i_dv_min+2]
    etasm=pars[basis_pars.i_sm_min+2]
    etafitcharm=pars[basis_pars.i_ch_min+2]
    delfitcharm=pars[basis_pars.i_ch_min+1]

    # delgp=pars[38]
    # delgm=pars[45]
    # delsea=pars[19]
    # delsp=pars[28]
    # etagp=pars[37]
    # etagm=pars[44]
    # etasea=pars[20]
    # etasp=pars[29]
    # delu=pars[1]
    # deld=pars[10]
    # delsm=pars[47]
    # etau=pars[2]
    # etad=pars[11]
    # etasm=pars[48]
    # etafitcharm=pars[66]
    # delfitcharm=pars[65]


    # etadbub=pars[51]

    # if etadbub < 0:
    #     print('PARCHECK : etadbub < 0')
    #     err=True
    
    if etasm > 1000.:
        print('PARCHECK : etasm > 1000.')
        err=True

    if etasm < 0:
        print('PARCHECK : etasm < 0')
        err=True
    
    if etau < 0:
        print('PARCHECK : etau < 0')
        err=True
        
    if etad < 0:
        print('PARCHECK : etad < 0')
        err=True
    
    if delsm < 0:
        print('PARCHECK : delsm < 0')
        err=True
    
    if delu < 0:
        print('PARCHECK : delu < 0')
        err=True

    if deld < 0:
        print('PARCHECK : deld < 0')
        err=True

    if etasea < 0:
        print('PARCHECK : etasea < 0')
        err=True

    if etasp < 0:
        print('PARCHECK : etasp < 0')
        err=True

    if etafitcharm < 0:
        print('PARCHECK : etafitcharm < 0')
        err=True
    
    if etagp < 0:
        print('PARCHECK : etagp < 0')
        err=True
    
    if delgp < -1:
        print('PARCHECK : delgp < -1')
        err=True

    gptest=I(delgp,etagp,0.)
    if gptest < 1e-50:
        print('PARCHECK : delgp,etagp too high - unstable')
        err=True

    if delsea < -1:
        print('PARCHECK : delsea < -1')
        err=True

    if delsp < -1:
        print('PARCHECK : delsp < -1')
        err=True

    if delfitcharm < -1:
        print('PARCHECK : delfitcharm < -1')
        err=True

    if(basis_pars.g_second_term):
        
        if etagm < -1:
            print('PARCHECK : etagm < -1')
            err=True
            
        if delgm < -1:
            print('PARCHECK : delgm < -1')
            err=True 

    return err
        
#    print('test',delg,delgp)

def parinc_eps(parin,ipar,eps):

    npar=pdf_pars.par_free_i[ipar]

    eps_n=eps*np.abs(parin[npar])

    if np.abs(parin[npar]) < 1e-8:
        eps_n=1e-12

    return eps_n

def parinc_newmin(parin,ipar,eps):

    npar=pdf_pars.par_free_i[ipar]

    # eps_n=eps*np.abs(parin[npar])
    eps_n=eps

    # if np.abs(parin[npar]) < 1e-8:
    #     eps_n=1e-12

    out=parin.copy()
    out[npar]=out[npar]+eps_n

    ###  sum rules...

    if basis_pars.asp_fix:
        out[basis_pars.i_sp_min+1]=out[basis_pars.i_sea_min+1]
    if basis_pars.dvd_eq_uvd:
        out[basis_pars.i_dv_min+1]=out[basis_pars.i_uv_min+1]


    out[basis_pars.i_sm_min+3]=1.
    out[0]=1.
    out[basis_pars.i_dv_min]=1.
    out[basis_pars.i_g_min]=1.

    if basis_pars.t8_int:
        asea=out[basis_pars.i_sea_min:basis_pars.i_sea_max].copy()
        asp=out[basis_pars.i_sp_min:basis_pars.i_sp_max].copy()
        norm_sea=q_msht_lowx_norm(asea)  
        asp_new=sp_norm_fix(norm_sea,asp)
        out[basis_pars.i_sp_min]=asp_new

    asm=out[basis_pars.i_sm_min:basis_pars.i_sm_max].copy()
    auv=out[basis_pars.i_uv_min:basis_pars.i_uv_max].copy()

    adv=out[basis_pars.i_dv_min:basis_pars.i_dv_max].copy()
    x0=smin_norm(asm)
    
    out[basis_pars.i_sm_min+3]=x0

    out[0]=qv_norm(1,auv)

    out[basis_pars.i_dv_min]=qv_norm(2,adv)

    ag=msum_ag(out)

    out[basis_pars.i_g_min]=ag

    return (out,eps_n)

def parinc(parin,ipar,epar):

    npar=pdf_pars.par_free_i[ipar]

    eps=1e-5 # was 1e-5 before
    eps_n=eps*np.abs(parin[npar])

    if np.abs(parin[npar]) < 1e-8:
        eps_n=1e-12

    out=parin.copy()
    if epar==1:
        out[npar]=out[npar]+eps_n
    elif epar==2:
        out[npar]=out[npar]-eps_n
    elif epar==3:
        out[npar]=out[npar]-eps_n*2.
    elif epar==4:
        out[npar]=out[npar]+eps_n*2.

    ###  sum rules...

    if basis_pars.asp_fix:
        # out[28]=out[19]
        out[basis_pars.i_sp_min+1]=out[basis_pars.i_sea_min+1]
    if basis_pars.dvd_eq_uvd:
        # out[10]=out[1]
        out[basis_pars.i_dv_min+1]=out[basis_pars.i_uv_min+1]


    out[basis_pars.i_sm_min+3]=1.
    out[0]=1.
    out[basis_pars.i_dv_min]=1.
    out[basis_pars.i_g_min]=1.

    if basis_pars.t8_int:
        # asea=out[18:27].copy()
        asea=out[basis_pars.i_sea_min:basis_pars.i_sea_max].copy()
        # asp=out[27:36].copy()
        asp=out[basis_pars.i_sp_min:basis_pars.i_sp_max].copy()
        norm_sea=q_msht_lowx_norm(asea)  
        asp_new=sp_norm_fix(norm_sea,asp)
        # out[27]=asp_new
        out[basis_pars.i_sp_min]=asp_new

    # asm=out[46:56].copy()
    asm=out[basis_pars.i_sm_min:basis_pars.i_sm_max].copy()
    # auv=out[0:9].copy()
    auv=out[basis_pars.i_uv_min:basis_pars.i_uv_max].copy()

    # adv=out[9:18]
    adv=out[basis_pars.i_dv_min:basis_pars.i_dv_max].copy()
    x0=smin_norm(asm)
    
    # out[49]=x0
    out[basis_pars.i_sm_min+3]=x0

    out[0]=qv_norm(1,auv)
    # print('out =',out[0])
    # LHL NEW TO DELETE
    # out[0]=2.344063147372577
    # out[9]=qv_norm(2,adv)
    out[basis_pars.i_dv_min]=qv_norm(2,adv)

    ag=msum_ag(out)

    # out[36]=ag
    out[basis_pars.i_g_min]=ag

    return (out,eps_n)
