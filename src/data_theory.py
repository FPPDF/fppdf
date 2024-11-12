from global_pars import *
# from validphys.api import API
from validphys import core
from validphys.api import API
from validphys.pseudodata import make_replica
from validphys.convolution import predictions, linear_predictions, central_predictions
import pandas as pd
from validphys.loader import Loader
from validphys.convolution import hadron_predictions, dis_predictions
from validphys.fkparser import load_fktable
from validphys.kinematics import *
import operator
import lhapdf
from validphys.commondataparser import load_commondata



def dat_calc_rep(dscomb,cov):

    if fit_pars.nlo_cuts:
        # intersection=[{"dataset_inputs": dload_pars.dscomb, "theoryid": 212}]
        intersection=[{"dataset_inputs": dload_pars.dscomb, "theoryid": 200}]
        lcd = API.dataset_inputs_loaded_cd_with_cuts(dataset_inputs=dscomb,theoryid=fit_pars.theoryidi,use_cuts="fromintersection",cuts_intersection_spec=intersection)                          
    else:
        lcd = API.dataset_inputs_loaded_cd_with_cuts(dataset_inputs=dscomb,theoryid=fit_pars.theoryidi,use_cuts='internal')        
    
    if fit_pars.pseud:
        dattot=make_replica(lcd,replica_mcseed=fit_pars.irep,dataset_inputs_sampling_covmat=cov,sep_mult=False,genrep=True)  


    else:
        dattot=make_replica(lcd,replica_mcseed=fit_pars.irep,dataset_inputs_sampling_covmat=cov,sep_mult=False,genrep=False)
        

    return(dattot)

def del_pen(delta):

    sig=0.05
    d0=0.3
    pen=np.exp(-(delta-d0)/sig)
    deriv=-pen/sig
    deriv2=pen/sig/sig


    return (pen,deriv,deriv2)    

def del_pen_calc():

    chi0d=del_pen(pdf_pars.deld_arr[0])[0]
    chi0u=del_pen(pdf_pars.delu_arr[0])[0]

    idv=0
    iuv=0
    diffd=0.
    diffu=0.
    hessd=0.
    hessu=0.

    for ip in range(1,pdf_pars.npar_free+1):
        deltad=pdf_pars.deld_arr[ip]-pdf_pars.deld_arr[0]
        deltau=pdf_pars.delu_arr[ip]-pdf_pars.delu_arr[0]
        if np.abs(deltad) > 1e-30:
            idv=ip
            diffd=del_pen(pdf_pars.deld_arr[0])[1]
            hessd=del_pen(pdf_pars.deld_arr[0])[2]
        if np.abs(deltau) > 1e-30:
            iuv=ip
            diffu=del_pen(pdf_pars.delu_arr[0])[1]
            hessu=del_pen(pdf_pars.delu_arr[0])[2]

    return (chi0d,chi0u,diffd,diffu,hessd,hessu,idv,iuv)

def pos_calc(pdata):

    tot=0.
    totdiff=0.

    
    for j in range(0,len(pdata)):
    
        if fit_pars.theoryidi==40001000:            
            api_predictions = API.positivity_predictions_data_result(theoryid=fit_pars.theoryidi, pdf=pdf_pars.PDFlabel, posdataset=pdata[j],use_cuts="internal")
        else:
            api_predictions = API.positivity_predictions_data_result(theoryid=fit_pars.theoryidi, pdf=pdf_pars.PDFlabel, posdataset=pdata[j])
        out=api_predictions.central_value

        # print(pdata[j])
        # print(out)

        outelu=np.zeros(len(out))
        for i in range(0,len(out)):
            lam=1e6
            if  pdata[j]["dataset"] == 'POSDYU':
                lam=1e10
#                print(i,out[i],elu(-out[i],lam))
            if pdata[j]["dataset"] == 'POSDYD':
                lam=1e10
            if pdata[j]["dataset"] == 'POSDYS':
                lam=1e10
                #                print(out[i],elu(-out[i],lam))

            lam=fit_pars.lampos
                
            outelu[i]=elu(-out[i],lam)

        # print(pdata[j],np.sum(outelu))
        tot=tot+np.sum(outelu)
        
    return tot

def elu(x,lam):
    alpha=1e-7
#    alpha=0.1/lam
#    lam=1e6
    if x>=0:
        out=x
    else:
        out=alpha*(np.exp(x)-1.)
        
    out=out*lam
    return out

def theory_calc(i,dataset_testii,inpt,ctrue):

    dnam=str(API.dataset(**inpt))
    predarr=theory_calc_def(i,dataset_testii,inpt,ctrue)

    return(predarr)

def theory_calc_def(i,dataset_testii,inpt,ctrue):

    if ctrue:
        cfaci=API.cfac(**dataset_testii)

    dnam=str(API.dataset(**inpt))
    pdfin=load_nnpdf.l.check_pdf(pdf_pars.PDFlabel)
 
    if ctrue:
        if "variant" in dataset_testii:
            ds = load_nnpdf.l.check_dataset(dnam, theoryid=fit_pars.theoryidi, variant=dataset_testii["variant"], cfac=cfaci)
        else:
            ds = load_nnpdf.l.check_dataset(dnam, theoryid=fit_pars.theoryidi, cfac=cfaci)
    else:
        # ds = load_nnpdf.l.check_dataset(dnam, theoryid=fit_pars.theoryidi)
        if "variant" in dataset_testii:
            print(dataset_testii["variant"])
            ds = load_nnpdf.l.check_dataset(dnam, theoryid=fit_pars.theoryidi, variant=dataset_testii["variant"])
        else:  
            ds = load_nnpdf.l.check_dataset(dnam, theoryid=fit_pars.theoryidi)

    if fit_pars.nlo_cuts:
        # ds_nlo=load_nnpdf.l.check_dataset(dnam, theoryid=212)
        ds_nlo=load_nnpdf.l.check_dataset(dnam, theoryid=200)
        ds.cuts=ds_nlo.cuts


    preds = central_predictions(ds,pdfin).T
    assert isinstance(preds, pd.DataFrame)
    test=preds.iloc[0]
    assert isinstance(test, pd.Series)
    predarr=test.array

    dload_pars.fk_loadarr[i]=False


    return(predarr)

def plot_data_x(ds,pred):

    kin=kinematics_table_notable(ds,cuts=None)
    kin=kin.T

    assert isinstance(kin, pd.DataFrame)

    kin_x=kin.iloc[0]
    kin_q=kin.iloc[1]

    assert isinstance(kin_x, pd.Series)
    assert isinstance(kin_q, pd.Series)
    kin_x=kin_x.array
    kin_q=kin_q.array


    output='outputs/chi2ind_group/'+output
    with open(output,'w') as outputfile:

        for i in range(0,len(kin_x)): 
            print(i,kin_x[i],kin_q[i],pred[i])
            L=[str(f'{i:5}'),' ',str(f'{kin_x[i]:10}'),' ',str(f'{kin_q[i]:6}'),' ',str(f'{pred[i]:25}')]
            outputfile.writelines(L)
            outputfile.write('\n')

    exit()
