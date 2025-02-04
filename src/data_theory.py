from global_pars import *
# from validphys.api import API
from validphys import core
from validphys.api import API
from validphys.pseudodata import make_replica
from validphys.convolution import predictions, linear_predictions, central_predictions
from validphys.convolution import _predictions,central_fk_predictions
import pandas as pd
from validphys.loader import Loader
from validphys.convolution import hadron_predictions, dis_predictions
from validphys.convolution import central_hadron_predictions, central_dis_predictions
from validphys.convolution import _gv_hadron_predictions
from validphys.fkparser import load_fktable
from validphys.kinematics import *
import operator
import lhapdf
from validphys.commondataparser import load_commondata
from validphys.convolution import OP
import functools
from validphys.pdfbases import evolution
import contextlib

def dat_calc_rep(dscomb,cov):

    if fit_pars.nlo_cuts:
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

def pos_calc(pdata, vp_pdf=None):

    tot=0.
    totdiff=0.
    if vp_pdf is None:
        vp_pdf = pdf_pars.PDFlabel

    
    for j in range(0,len(pdata)):
    
        if fit_pars.theoryidi==40001000 or fit_pars.theoryidi==50001000:            
            api_predictions = API.positivity_predictions_data_result(theoryid=fit_pars.theoryidi, pdf=vp_pdf, posdataset=pdata[j],use_cuts="internal")
        else:
            api_predictions = API.positivity_predictions_data_result(theoryid=fit_pars.theoryidi, pdf=vp_pdf, posdataset=pdata[j])
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

def _asy(a, b):
    return (a - b) / (a + b)

def _com(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t):
    return (a + b + c + d + e + f + g + h + i + j) / (k + l + m + n + o + p + q + r + s + t)


def output_fktables_names(fknam):

    output='fknames.dat'
    
    with open(output,'a') as outputfile:
    
        outputfile.write(str(fknam))                                                   
        outputfile.write('\n')
        

def _predictions_2pdfs(dataset, fkfunc, pdf1, pdf2=None):
    """Combine data on all the FKTables in the database according to the
    reduction operation defined therein. Dispatch the kind of predictions (for
    all replicas, central, etc) according to the provided ``fkfunc``, which
    should have the same interface as e.g. ``fk_predictions``.
    """
 
    # print(dataset)
    # print(dataset.op)
    opfunc = OP[dataset.op]
    if dataset.cuts is None:
        raise PredictionsRequireCutsError(
            "FKTables do not always generate predictions for some datapoints "
            "which are usually cut. Loading predictions without cuts can "
            "therefore produce predictions whose shape doesn't match the uncut "
            "commondata and is not supported."
        )
    cuts = dataset.cuts.load()
    all_predictions = []
    ifk=0
    # print(dataset.fkspecs)
    # os.quit()
    # for fkarr in dataset.fkspecs[0].metadata.FK_tables:
    #     for fk in fkarr:
    #         output_fktables_names(fk)

    for fk in dataset.fkspecs:
        # print('ifk=',ifk)
        # print(fk)
        # print('')
        # print(fk.metadata.FK_tables)
        # print('')
        # print(len(fk.metadata.FK_tables(0))
        # os.quit()
        if fit_pars.load_fk:
            fk_w_cuts = load_fktable(fk).with_cuts(cuts)
        else:
            fk_w_cuts = Dummy()
            fk_w_cuts.hadronic=True
        if pdf2 is not None:
            all_predictions.append(fkfunc(dataset,fk_w_cuts, pdf1, pdf2, ifk))
        else:
            all_predictions.append(fkfunc(dataset,fk_w_cuts, pdf1, None, ifk))
        ifk+=1

    # Old fktables repeated values to make DEN and NUM sizes match in RATIO operations
    # pineappl tables instead just contain the one value used
    # The code below works for both situation while keeping `true_div` as the operation
    if dataset.op == "RATIO":
        all_predictions[-1] = all_predictions[-1].values


    # print(all_predictions)
    # print(opfunc(*all_predictions))
    # os.quit()
    return opfunc(*all_predictions)

def _predictions_2pdfs_new(dataset, fkfunc, pdf1, pdf2=None):
    """Combine data on all the FKTables in the database according to the
    reduction operation defined therein. Dispatch the kind of predictions (for
    all replicas, central, etc) according to the provided ``fkfunc``, which
    should have the same interface as e.g. ``fk_predictions``.
    """
    
    opfunc = OP[dataset.op]
    if dataset.cuts is None:
        raise PredictionsRequireCutsError(
            "FKTables do not always generate predictions for some datapoints "
            "which are usually cut. Loading predictions without cuts can "
            "therefore produce predictions whose shape doesn't match the uncut "
            "commondata and is not supported."
        )
    cuts = dataset.cuts.load()
    all_predictions = []
    all_predictions_r1 = []
    all_predictions_r2 = []
    ifk=0
    fk_w_cuts_arr=[]
    for fk in dataset.fkspecs:
        if fit_pars.load_fk:
            fk_w_cuts = load_fktable(fk).with_cuts(cuts)
            # print(fk_w_cuts)
        else:
            fk_w_cuts = Dummy()
            fk_w_cuts.hadronic=True
        # fk_w_cuts = load_fktable(fk).with_cuts(cuts)
        # Edited so dO/dpar calculated 
        if fit_pars.newmin and fk_w_cuts.hadronic:
            if dataset.op == 'NULL':
                fkapp=fkfunc(dataset,fk_w_cuts, pdf1, pdf2)+fkfunc(dataset,fk_w_cuts, pdf2, pdf1)
                all_predictions.append(fkapp)
            elif dataset.op == 'RATIO' or dataset.op == 'ASY' or dataset.op == 'COM':
                fk_w_cuts_arr.append(fk_w_cuts)
            else:
                all_predictions.append(fkfunc(dataset,fk_w_cuts, pdf1, pdf2))
        else:
            if dataset.op == 'RATIO':
                fk_w_cuts_arr.append(fk_w_cuts)
            else:   
                all_predictions.append(fkfunc(dataset,fk_w_cuts, pdf2))
        ifk+=1


    if dataset.op == 'RATIO' and pdf2 is None:
        fk_num_diff=fkfunc(dataset,fk_w_cuts_arr[0],pdf2)       
        all_predictions_r1.append(fk_num_diff)
        fk_den=fkfunc(dataset,fk_w_cuts_arr[1],pdf1)
        all_predictions_r1.append(fk_den)

        fk_den_diff=fkfunc(dataset,fk_w_cuts_arr[1],pdf2)
        fk_num=fkfunc(dataset,fk_w_cuts_arr[0],pdf1)
        all_predictions_r2.append(fk_den_diff*fk_num)
        fkapp=np.power(fk_den,2)
        all_predictions_r2.append(fkapp)

    if dataset.op == 'RATIO' and pdf2 is not None:
        fk_num_12=fkfunc(dataset,fk_w_cuts_arr[0], pdf1, pdf2,0)+fkfunc(dataset,fk_w_cuts_arr[0], pdf2, pdf1,0)
        fkapp=fk_num_12
        all_predictions_r1.append(fkapp)
        fk_den_1=fkfunc(dataset,fk_w_cuts_arr[1], pdf1,None,1)
        fkapp=fk_den_1
        all_predictions_r1.append(fkapp)

        # print(all_predictions_r1)

        fk_num_1=fkfunc(dataset,fk_w_cuts_arr[0], pdf1,None,0)
        fk_den_12=fkfunc(dataset,fk_w_cuts_arr[1], pdf1, pdf2,1)+fkfunc(dataset,fk_w_cuts_arr[1], pdf2, pdf1,1)
        # Corrected so that works when denominator is single number
        # fkapp=fk_num_1*fk_den_12
        fkapp=fk_num_1
        # print(fkapp)
        all_predictions_r2.append(fkapp)
        # fkapp=np.power(fk_den_1,2)
        fkapp=np.power(fk_den_1,2)/fk_den_12
        # print(fkapp)
        all_predictions_r2.append(fkapp)

      

    if dataset.op == 'ASY' and pdf2 is not None:
        fk_12_a=fkfunc(dataset,fk_w_cuts_arr[0], pdf1, pdf2,0)+fkfunc(dataset,fk_w_cuts_arr[0], pdf2, pdf1,0)
        fk_12_b=fkfunc(dataset,fk_w_cuts_arr[1], pdf1, pdf2,1)+fkfunc(dataset,fk_w_cuts_arr[1], pdf2, pdf1,1)
        all_predictions_r1.append(fk_12_a-fk_12_b)
        fk_1_a=fkfunc(dataset,fk_w_cuts_arr[0], pdf1,pdf1,0)
        fk_1_b=fkfunc(dataset,fk_w_cuts_arr[1], pdf1,pdf1,1)
        all_predictions_r1.append(fk_1_a+fk_1_b)

        fkapp=fk_12_a+fk_12_b
        fkapp*=(fk_1_a-fk_1_b)
        all_predictions_r2.append(fkapp)
        fkapp=np.power(fk_1_a+fk_1_b,2)
        all_predictions_r2.append(fkapp)

    if dataset.op == 'COM' and pdf2 is not None:
        fk_num_12=0.
        for i in range(0,10):
            fk_num_12+=fkfunc(dataset,fk_w_cuts_arr[i], pdf1, pdf2)+fkfunc(dataset,fk_w_cuts_arr[i], pdf2, pdf1)
        all_predictions_r1.append(fk_num_12)
        fk_den_1=0.
        for i in range(10,20):
            fk_den_1+=fkfunc(dataset,fk_w_cuts_arr[i], pdf1)
        all_predictions_r1.append(fk_den_1)

        fk_num_1=0.
        for i in range(0,10):
            fk_num_1+=fkfunc(dataset,fk_w_cuts_arr[i], pdf1)
        fk_den_12=0.
        for i in range(10,20):
            fk_den_12+=fkfunc(dataset,fk_w_cuts_arr[i], pdf1, pdf2)+fkfunc(dataset,fk_w_cuts_arr[i], pdf2, pdf1)
        fkapp=fk_num_1*fk_den_12
        all_predictions_r2.append(fkapp)
        fkapp=np.power(fk_den_1,2)
        all_predictions_r2.append(fkapp)

    # Old fktables repeated values to make DEN and NUM sizes match in RATIO operations
    # pineappl tables instead just contain the one value used
    # The code below works for both situation while keeping `true_div` as the operation
    
    if dataset.op == "RATIO":
        # all_predictions[-1] = all_predictions[-1].values
        all_predictions_r1[-1] = all_predictions_r1[-1].values
        all_predictions_r2[-1] = all_predictions_r2[-1].values
        out=opfunc(*all_predictions_r1)-opfunc(*all_predictions_r2)
    elif dataset.op == "ASY":
        all_predictions_r1[-1] = all_predictions_r1[-1].values
        all_predictions_r2[-1] = all_predictions_r2[-1].values
        out=operator.truediv(*all_predictions_r1)-operator.truediv(*all_predictions_r2)
    elif dataset.op == "COM":
        all_predictions_r1[-1] = all_predictions_r1[-1].values
        all_predictions_r2[-1] = all_predictions_r2[-1].values
        out=operator.truediv(*all_predictions_r1)-operator.truediv(*all_predictions_r2)
    else:
        out=opfunc(*all_predictions)


    return out

def df_2_array(predsin):
    
    preds=predsin.copy().T
    assert isinstance(preds, pd.DataFrame)
    test=preds.iloc[0]
    assert isinstance(test, pd.Series)
    predarr=test.array

    # out=pd.DataFrame(predarr)
    
    return predarr


def central_hadron_predictions_2pdfs_wmsht(ds,loaded_fk, pdf1, pdf2=None,fk_it=None):

    # print(fk_it)
    if fit_pars.load_fk:
        out=central_hadron_predictions_2pdfs(loaded_fk, pdf1, pdf2)
        outarr=df_2_array(out)

    dataset_testii=fit_pars.dataset_ii_global


    out=pd.DataFrame(outarr)

    return out

def central_hadron_predictions_2pdfs(loaded_fk, pdf1, pdf2=None):
    """Implementation of :py:func:`central_fk_predictions` for hadronic
    observables."""
    gv1 = functools.partial(evolution.central_grid_values, pdf=pdf1)
    if pdf2 is not None:
        gv2 = functools.partial(evolution.central_grid_values, pdf=pdf2)
    else:
        gv2=gv1

    out=_gv_hadron_predictions(loaded_fk, gv1, gv2)
    return out

def central_fk_predictions_2pdfs(ds,loaded_fk, pdf1, pdf2=None, ifk=None):
    """Same as :py:func:`fk_predictions`, but computing predictions for the
    central PDF member only."""
    if loaded_fk.hadronic:
        if pdf2 is not None:
            # out=central_hadron_predictions_2pdfs(loaded_fk, pdf1, pdf2)
            out=central_hadron_predictions_2pdfs_wmsht(ds,loaded_fk, pdf1, pdf2, ifk)
            return out
            # return central_hadron_predictions_2pdfs(loaded_fk, pdf1, pdf2)
        else:
            # out=central_hadron_predictions_2pdfs(loaded_fk, pdf1)
            out=central_hadron_predictions_2pdfs_wmsht(ds,loaded_fk, pdf1, None, ifk)
            return out
            # return central_hadron_predictions_2pdfs(loaded_fk, pdf1)
    else:
        return central_dis_predictions(loaded_fk, pdf1)

def theory_calc(i,dataset_testii,inpt,ctrue):

    dnam=str(API.dataset(**inpt))

    fit_pars.load_fk=True

    predarr=theory_calc_def(i,dataset_testii,inpt,ctrue)


    return(predarr)

def preds_calc(ds,pdfin0,pdfin=None):

    # cuts = ds.cuts.load()
    # table=load_fktable(ds.fkspecs[0]).with_cuts(cuts)
    # print(ds.op,table.hadronic)

    if pdfin is not None:
        preds = _predictions_2pdfs_new(ds,central_fk_predictions_2pdfs,pdfin0,pdfin).T
        # if table.hadronic:
        #     preds = _predictions_2pdfs_new(ds,central_fk_predictions_2pdfs,pdfin0,pdfin).T
        # else:
        #     preds=central_predictions(ds,pdfin).T
    else:
        preds = _predictions_2pdfs(ds,central_fk_predictions_2pdfs,pdfin0).T
        # preds=central_predictions(ds,pdfin0).T
    # preds=central_predictions(ds,pdfin).T
    assert isinstance(preds, pd.DataFrame)
    test=preds.iloc[0]
    assert isinstance(test, pd.Series)
    predarr=test.array

    # print(predarr)

    return predarr

def kin_out(inpt):

    cd = API.commondata(**inpt)
    lcd = load_commondata(cd)
    kin1=lcd.commondata_table["kin1"]
    kin2=lcd.commondata_table["kin2"]
    print(kin1)
    print(kin2)



def theory_calc_def(i,dataset_testii,inpt,ctrue):

    # print(i)
    # print(dload_pars.fk_loadarr[i])

    fit_pars.dataset_ii_global=dataset_testii

    if ctrue:
        cfaci=API.cfac(**dataset_testii)

    dnam=str(API.dataset(**inpt))

   

    # print(dnam)

    # print('PDF LOAD...')
    pdfin=load_nnpdf.l.check_pdf(pdf_pars.PDFlabel)
    
    if fit_pars.newmin and pdf_pars.iPDF > 0:
        pdfin0=load_nnpdf.l.check_pdf(pdf_pars.PDFlabel_cent)
    #     print(pdf_pars.iPDF,pdf_pars.PDFlabel,pdf_pars.PDFlabel_cent)
    # print('...FINISH')

    # print('ds LOAD...')    

    # if ctrue:
    #     ds = load_nnpdf.l.check_dataset(dnam, theoryid=fit_pars.theoryidi, cfac=cfaci)
    # else:
    #     ds = load_nnpdf.l.check_dataset(dnam, theoryid=fit_pars.theoryidi)


    if ctrue:
        if "variant" in dataset_testii:
            ds = load_nnpdf.l.check_dataset(dnam, theoryid=fit_pars.theoryidi, variant=dataset_testii["variant"], cfac=cfaci)
        else:
            ds = load_nnpdf.l.check_dataset(dnam, theoryid=fit_pars.theoryidi, cfac=cfaci)
    else:
        # ds = load_nnpdf.l.check_dataset(dnam, theoryid=fit_pars.theoryidi)
        if "variant" in dataset_testii:
            # print(dataset_testii["variant"])
            ds = load_nnpdf.l.check_dataset(dnam, theoryid=fit_pars.theoryidi, variant=dataset_testii["variant"])
        else:  
            ds = load_nnpdf.l.check_dataset(dnam, theoryid=fit_pars.theoryidi)

    if fit_pars.nlo_cuts:
        # ds_nlo=load_nnpdf.l.check_dataset(dnam, theoryid=212)
        ds_nlo=load_nnpdf.l.check_dataset(dnam, theoryid=200)
        ds.cuts=ds_nlo.cuts

 

    # cuts = ds.cuts.load()
    # table=load_fktable(ds.fkspecs[0]).with_cuts(cuts)
    # # print(table.hadronic)

    # preds = _predictions(ds,pdfin,central_fk_predictions_test).T

    # print(preds_test)


    # kin_out(inpt)

    # print(inpt)
    # print(lcd.commondata_table["kin1"])
    # os.quit()

    # pset_nnpdf = lhapdf.getPDFSet(pdf_pars.PDFlabel)
    # lh_nnpdf = pset_nnpdf.mkPDFs()
    # cuts = ds.cuts.load()
    # table=load_fktable(ds.fkspecs[0]).with_cuts(cuts)
    # dload_pars.xarr_test=np.append(dload_pars.xarr_test,table.xgrid)
    # dload_pars.xarr_test=np.sort(dload_pars.xarr_test)
    # print(len(dload_pars.xarr_test))
    # print('')

    # print(table.xgrid)
    # print(np.min(table.xgrid),np.max(table.xgrid))
    # print(len(table.xgrid))
    # if dload_pars.fk_loadarr[i]:
    #     cuts = ds.cuts.load()
    #     if ds.op != "NULL":
    #         tablearr=[]
    #         for fk in ds.fkspecs:
    #             tabledum=load_fktable(fk).with_cuts(cuts)
    #             dload_pars.t_had[i]=tabledum.hadronic
    #             tablearr.append(tabledum)
    #             dload_pars.fk_arr.append(tabledum)
    #             # dload_pars.ifk+=1
    #     else:
    #         table=load_fktable(ds.fkspecs[0]).with_cuts(cuts)
    #         dload_pars.t_had[i]=table.hadronic
    #         dload_pars.fk_arr.append(table)
    #         # dload_pars.fk_ind[i]=dload_pars.ifk
    #         # dload_pars.ifk+=1
    #     # if ds.op == "RATIO" or ds.op == "ASY":
    #     #     table1=load_fktable(ds.fkspecs[1]).with_cuts(cuts)
    #     #     dload_pars.fk_arr.append(table1)
    #     #     # dload_pars.fk_ind[i]=dload_pars.ifk
    #     #     dload_pars.ifk+=1
    # else:
    #     if ds.op != "NULL":
    #         tablearr=[]
    #         for fk in ds.fkspecs:
    #             tablearr.append(dload_pars.fk_arr[dload_pars.ifk])
    #             dload_pars.ifk+=1
    #     else:
    #         table=dload_pars.fk_arr[dload_pars.ifk]
    #         dload_pars.ifk+=1
    # if dload_pars.t_had[i]:
    #     if ds.op != "NULL":
    #         test1=[]
    #         idum=0
    #         for fk in ds.fkspecs:
    #             test1.append(hadron_predictions(tablearr[idum], pdfin))
    #             idum+=1
    #         if ds.op == "RATIO":        
    #             test1[-1] = test1[-1].values
    #             test1=operator.truediv(*test1)
    #         elif ds.op == "ASY":
    #             test1=_asy(*test1)
    #         elif ds.op == "COM":
    #             test1=_com(*test1)
    #         elif ds.op == "ADD":
    #             test1=operator.add(*test1)
    #     else:
    #         test1=hadron_predictions(table, pdfin)
    # else:
    #     if ds.op != "NULL":
    #         test1=[]
    #         idum=0
    #         for fk in ds.fkspecs:
    #             test1.append(dis_predictions(tablearr[idum], pdfin))
    #             idum+=1
    #         if ds.op == "RATIO":        
    #             test1[-1] = test1[-1].values
    #             test1=operator.truediv(*test1)
    #         elif ds.op == "ASY":
    #             test1=_asy(*test1)
    #         elif ds.op == "COM":
    #             test1=_com(*test1)
    #         elif ds.op == "ADD":
    #             test1=operator.add(*test1)
    #     else:
    #         test1=dis_predictions(table, pdfin)
    # test1=test1.T
    # assert isinstance(test1, pd.DataFrame)
    # testa=test1.iloc[0]
    # assert isinstance(testa, pd.Series)
    # predarr=testa.array  

    # print('...FINISH')

    # preds = central_predictions(ds,pdfin).T
    # assert isinstance(preds, pd.DataFrame)
    # test=preds.iloc[0]
    # assert isinstance(test, pd.Series)
    # predarr=test.array

    if fit_pars.newmin and pdf_pars.iPDF > 0:
        # predarr1=preds_calc(ds,pdfin0,pdfin)
        # predarr2=preds_calc(ds,pdfin,pdfin0)
        # predarr=predarr1+predarr2

        predarr=preds_calc(ds,pdfin0,pdfin)

    else:
        predarr=preds_calc(ds,pdfin)
        # predarr=preds_calc(ds,pdfin,pdf_nn)
        # predarrt=preds_calc(ds,pdf_nn,pdfin)
        # print(predarr)
        # print(predarrt)
        # print((predarr-predarrt)/predarr)
        # os.quit()

    dload_pars.fk_loadarr[i]=False


    # print(predarr)
    # print(str(dnam))
    # os.quit()

    # if str(dnam)=='BCDMSP_dwsh':
    
    # plot_data_x(i,ds,predarr)  



    # print(i,ds.op)
    # print(ds.fkspecs)
    # print(i,dload_pars.fk_loadarr[i])
    # print(predarr)
    # print(predarr-predarra)


    return(predarr)

def plot_data_x(idat,ds,pred):

    kin=kinematics_table_notable(ds,cuts=None)
    kin=kin.T

    assert isinstance(kin, pd.DataFrame)

    print(idat)
    print(kin)

    kin_x=kin.iloc[0]
    kin_q=kin.iloc[1]

    assert isinstance(kin_x, pd.Series)
    assert isinstance(kin_q, pd.Series)
    kin_x=kin_x.array
    kin_q=kin_q.array



    # output='xq_tot.dat'
    # output='outputs/chi2ind_group/'+output
    # with open(output,'a') as outputfile:

    #     for i in range(0,len(kin_x)): 
    #         print(idat,kin_x[i],kin_q[i],pred[i])
    #         L=[str(f'{idat:5}'),' ',str(f'{kin_x[i]:10}'),' ',str(f'{kin_q[i]:6}'),' ',str(f'{pred[i]:25}')]
    #         outputfile.writelines(L)
    #         outputfile.write('\n')

    
