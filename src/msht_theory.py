from global_pars import *
import lhapdf
import pineappl
import numpy as np
from validphys.api import API
from validphys.commondataparser import load_commondata
import contextlib
import os
import sys
from contextlib import contextmanager
import io

def pdf_dict_add(pdf,out):

    pdf_dict={
        "name": str(pdf),
        "pdf": pdf,
        "lhapdf": out
        }
    fit_pars.pdf_dict.append(pdf_dict)

    # for i in range(0,len(fit_pars.pdf_dict)):
    #     print('DICT = ',fit_pars.pdf_dict[i]["name"])

def pdf_dict_check(pdf):

    pdf_in_dict=False
    for i in range(0,len(fit_pars.pdf_dict)):
        if str(pdf) in fit_pars.pdf_dict[i]["name"]:
            # print(str(pdf),fit_pars.pdf_dict[i]["name"],'yes!')
            pdf_in_dict=True
            out=fit_pars.pdf_dict[i]["lhapdf"]

    if not pdf_in_dict:
        out=lhapdf.mkPDF(str(pdf), 0)
        pdf_dict_add(pdf,out)

    return out



def pdf_caller(pdf1,pdf2=None):

    if pdf2 is None:
        # out1 = lhapdf.mkPDF(str(pdf1), 0)
        out2 = None
        # pdf_dict_add(pdf1)
        # print('test')
        out1=pdf_dict_check(pdf1)

    else:
        # out1 = lhapdf.mkPDF(str(pdf1), 0)
        # out2 = lhapdf.mkPDF(str(pdf2), 0)
        out1=pdf_dict_check(pdf1)
        out2=pdf_dict_check(pdf2)

        

    return (out1,out2)

def predi_calc(fktable,pdfin,pdfin1=None):

    parton1=2212

    # print('predi_calc')


    if pdfin1 is None:
        predi=np.array(fktable.convolve_with_one(parton1, pdfin.xfxQ2))
        # predi=np.array(fktable.convolve_with_two(parton1, pdfin.xfxQ2,parton1, pdfin.xfxQ2))
    else:
        predi=np.array(fktable.convolve_with_two(parton1, pdfin.xfxQ2,parton1, pdfin1.xfxQ2))
        # predi=np.array(fktable.convolve_with_one(parton1, pdfin1.xfxQ2))

        # print(pdfin,pdfin1)
        # os.quit()



    return predi

def redirect_stdout(new_target):
    old_target, sys.stdout = sys.stdout, new_target # replace sys.stdout
    try:
        yield new_target # run some code with the replaced stdout
    finally:
        sys.stdout = old_target # restore to the previous value

# normalized ttbar

def msht_theory_ATLAS8Z3D(dnam,dataset_testii,pdf1,pdf2):

    fktables=["ATLAS_Z3d_66_80","ATLAS_Z3d_80_91","ATLAS_Z3d_91_102","ATLAS_Z3d_102_116","ATLAS_Z3d_116_150","ATLAS_Z3d_150_200"]
    mbin=[14.,11.,11.,14.,34.,50.]
 
    fktable_arr=[]
    for fklab in fktables:
        fktable = pineappl.fk_table.FkTable.read(str(fit_pars.theories_path)+"/theory_"+str(fit_pars.theoryidi)+"/fastkernel/"+fklab+".pineappl.lz4")
        fktable_arr.append(fktable)

    (pdfin,pdfin1)=pdf_caller(pdf1,pdf2)
        # if pdf2 is None:
        #     # pdfin = lhapdf.mkPDF(pdf_pars.PDFlabel, 0)
        #     pdfin = lhapdf.mkPDF(str(pdf1), 0)
        # else:
        #     pdfin = lhapdf.mkPDF(pdf_pars.PDFlabel, 0)
    parton1=2212

    pred=[]
    for i in range (0,len(fktables)):
        # if pdf2 is None:
        #     predi=np.array(fktable_arr[i].convolve_with_one(parton1, pdfin.xfxQ2))
        # else:
        #     predi=np.array(fktable_arr[i].convolve_with_two(parton1, pdfin.xfxQ2,parton1, pdfin.xfxQ2))
        predi=predi_calc(fktable_arr[i],pdfin,pdfin1)
        pred.append(predi)

    cfdir=str(fit_pars.theories_path)+"/theory_"+str(fit_pars.theoryidi)+"/cfactor/"

#   From ATLAS (Sashsa Glazov)

    binnorm=2.5e-3

    cfaci=API.cfac(**dataset_testii)

    i=0
    for fklab in fktables:
        inputcf_atnorm=cfdir+"CF_ATnorm_"+fklab+".dat"
        distcf_atnorm=np.loadtxt(inputcf_atnorm,skiprows=9)
        distcf_atnorm=distcf_atnorm[:,0].flatten()
       
        inputcf_cfnnlo=cfdir+"CF_CFNNLO_"+fklab+".dat"
        distcf_cfnnlo=np.loadtxt(inputcf_cfnnlo,skiprows=9)
        distcf_cfnnlo=distcf_cfnnlo[:,0].flatten()

        # multipy by K-factors and ATLAS norm
        if 'ATnorm' in cfaci:
            pred[i]*=distcf_atnorm

        if 'CFNNLO' in cfaci:
            pred[i]*=distcf_cfnnlo

        pred[i]*=binnorm/mbin[i]
        i+=1

#   Sum over cos(theta) bins (always 6) and only store 'non-zero' ones (as in data)

    predout=[]
    iflag=0
    sig_include=np.ones(72,dtype='bool')
    for i in (9,10,11,22,23,34,35,46,47,58,59,70,71):
        sig_include[i]=False

    for i in range (0,len(fktables)):
        jtot=(np.rint(len(pred[i])/6)).astype(int)
        for j in range(0,jtot):
            jmin=j*6
            jmax=jmin+6
            sigint=np.sum(pred[i][jmin:jmax])
            
         
            if sig_include[iflag]:
            # if sigint > 0:
                predout.append(sigint)
                sig_include[iflag]=True

            iflag+=1
   
    # predout=np.array(predout)
    # print(sig_include)
    # os.quit()

    inputcf_atsc=cfdir+"CF_ATsc_"+str(dnam)+".dat"
    distcf_atsc=np.loadtxt(inputcf_atsc,skiprows=9)
    distcf_atsc=distcf_atsc[:,0].flatten()

    # print(len(fktables))
    # print(pred)
    # print(predout)
    # print(distcf_atsc)

    if 'ATsc' in cfaci:
        predout+=distcf_atsc

    return predout

def msht_theory_CDFwasym_ind(ifk,pdf1,pdf2):

    # with contextlib.redirect_stdout(None):
    #     pdfin = lhapdf.mkPDF(pdf_pars.PDFlabel, 0)
    (pdfin,pdfin1)=pdf_caller(pdf1,pdf2)

    parton1=2212

    if ifk == 0:
        fktable_wp = pineappl.fk_table.FkTable.read(str(fit_pars.theories_path)+"/theory_"+str(fit_pars.theoryidi)+"/fastkernel/grid-CDFWp-no-optimize-pbeam.pineappl.lz4")
        # pred = np.array(fktable_wp.convolve_with_one(parton1, pdfin.xfxQ2))
        pred=predi_calc(fktable_wp,pdfin,pdfin1)
    else:
        fktable_wm = pineappl.fk_table.FkTable.read(str(fit_pars.theories_path)+"/theory_"+str(fit_pars.theoryidi)+"/fastkernel/grid-CDFWm-no-optimize-pbeam.pineappl.lz4")
        # pred = np.array(fktable_wm.convolve_with_one(parton1, pdfin.xfxQ2))
        pred=predi_calc(fktable_wm,pdfin,pdfin1)


    # print(pred_wp)
    # print(pred_wm)

    # pred=(pred_wp-pred_wm)/(pred_wp+pred_wm)

    return pred

def msht_theory_CDFwasym():

    fktable_wp = pineappl.fk_table.FkTable.read(str(fit_pars.theories_path)+"/theory_"+str(fit_pars.theoryidi)+"/fastkernel/grid-CDFWp-no-optimize-pbeam.pineappl.lz4")
    fktable_wm = pineappl.fk_table.FkTable.read(str(fit_pars.theories_path)+"/theory_"+str(fit_pars.theoryidi)+"/fastkernel/grid-CDFWm-no-optimize-pbeam.pineappl.lz4")

    # pdfin = lhapdf.mkPDF(pdf_pars.PDFlabel, 0)
    pdfin =pdf_dict_check(pdf_pars.PDFlabel)


    parton1=2212
    pred_wp = np.array(fktable_wp.convolve_with_one(parton1, pdfin.xfxQ2))
    pred_wm = np.array(fktable_wm.convolve_with_one(parton1, pdfin.xfxQ2))

    # print(pred_wp)
    # print(pred_wm)

    pred=(pred_wp-pred_wm)/(pred_wp+pred_wm)

    return pred
    

def msht_theory_CMS8ttDD(predin,pdf1,pdf2,ifk):

    # load grid with all four pt bins to give full total xs (removed for rest as normalized)

    fktable = pineappl.fk_table.FkTable.read(str(fit_pars.theories_path)+"/theory_"+str(fit_pars.theoryidi)+"/fastkernel/CMS8ttDD_LHC8_PTtYt-1.45-2.50nnlo_renorm.pineappl.lz4")
    
    # with contextlib.redirect_stdout(None):
    #     pdfin = lhapdf.mkPDF(pdf_pars.PDFlabel, 0)

    (pdfin,pdfin1)=pdf_caller(pdf1,pdf2)

    parton1=2212
    # predy3full = np.array(fktable.convolve_with_one(parton1, pdfin.xfxQ2))
    predy3full = np.array(predi_calc(fktable,pdfin,pdfin1))

    predy0=predin[0:4]
    predy1=predin[4:8]
    predy2=predin[8:12]
    predy3=predin[12:15]

    # print(predy3,predy3full)
    # print(predy0,predy1,predy2,predy3)
    norm=0.35*np.sum(predy0)+0.5*np.sum(predy1)+0.6*np.sum(predy2)+1.05*np.sum(predy3full)
    # print(np.sum(predy3),np.sum(predy3full))
    # print(norm)
    # os.quit()

    if ifk == 1:
        predin=norm*np.ones((len(predin)))

    # predin/=norm

    return predin


# normalized cross section for Z 7 TeV

def msht_theory_LHCb15WZ(predin,pdf1,pdf2):


    predz0=predin[0:17]
    predwp=predin[17:25]
    predwm=predin[25:33]
    predz08=predin[33:51]
    predwp8=predin[51:59]
    predwm8=predin[59:67]

    # So that data goes w+,w- in each bin (to match original MSHT)
    predwpm=np.vstack((predwp,predwm)).ravel('F')
    predwpm8=np.vstack((predwp8,predwm8)).ravel('F')
    predin=np.concatenate((predz0,predwpm,predz08,predwpm8))

    # print(pred)   

    # os.quit()

#   Have to renormalize theory so sig and dsig/dy to match data
    predin[0:16]/=8.
    predin[16:29]/=4.
    predin[29:33]/=2.

    predin[33:49]/=8.
    predin[49:63]/=4.
    predin[63:]/=2.

    # print(predin[63:])
    # print('')
    # print(predin[17:29])
    # print(predin[29:33])
    # print(len(predin[29:34]))
    # os.quit()

    return predin

def msht_theory_HMDYatATLAS8(predin,pdf1,pdf2):

    cfdir=str(fit_pars.theories_path)+"/theory_"+str(fit_pars.theoryidi)+"/cfactor/"

    inputPI=cfdir+"CF_HMDYatATLAS8_PI.dat"
    distPI=np.loadtxt(inputPI)
    predin+=distPI

    return predin

def msht_theory_CMSDDZ0(predin,dataset_testii,pdf1,pdf2,ifk):


    fktable = pineappl.fk_table.FkTable.read(str(fit_pars.theories_path)+"/theory_"+str(fit_pars.theoryidi)+"/fastkernel/grid-total-Z0_rap1-_dmdy_60m120.pineappl.lz4")
    
    # with contextlib.redirect_stdout(None):
    #     pdfin = lhapdf.mkPDF(pdf_pars.PDFlabel, 0)

    (pdfin,pdfin1)=pdf_caller(pdf1,pdf2)

    parton1=2212
    # pred_Zpeak = np.array(fktable.convolve_with_one(parton1, pdfin.xfxQ2))
    pred_Zpeak = np.array(predi_calc(fktable,pdfin,pdfin1))

    cfaci=API.cfac(**dataset_testii)
    cfdir=str(fit_pars.theories_path)+"/theory_"+str(fit_pars.theoryidi)+"/cfactor/"

    inputPI=cfdir+"DYatCMSDDZ0_PI.dat"
    distPI=np.loadtxt(inputPI)
    predin+=distPI

    if 'CFNNLO' in cfaci:
        inputcf_cfnnlo=cfdir+"CF_CFNNLO_"+"grid-total-Z0_rap1-_dmdy_60m120"+".dat"
        distcf_cfnnlo=np.loadtxt(inputcf_cfnnlo,skiprows=9)
        distcf_cfnnlo=distcf_cfnnlo[:,0].flatten()
        pred_Zpeak*=distcf_cfnnlo

    xstot_Zpeak=np.sum(pred_Zpeak)*0.1*2.
    # pred_Zpeak /=xstot_Zpeak
    # print(pred_Zpeak)

    # print(xstot_Zpeak)
    # os.quit()

    # predin/=xstot_Zpeak
    
    if ifk == 1:
        predin=xstot_Zpeak*np.ones((len(predin)))


    return predin

def msht_theory_CMSZ0(dataset_testii,pdf1,pdf2,ifk):

    fktable = pineappl.fk_table.FkTable.read(str(fit_pars.theories_path)+"/theory_"+str(fit_pars.theoryidi)+"/fastkernel/grid-total-Z0_CMS-pT20.pineappl.lz4")
    
    # with contextlib.redirect_stdout(None):
    #     pdfin = lhapdf.mkPDF(pdf_pars.PDFlabel, 0)

    (pdfin,pdfin1)=pdf_caller(pdf1,pdf2)

    parton1=2212
    # pred = np.array(fktable.convolve_with_one(parton1, pdfin.xfxQ2))
    pred=predi_calc(fktable,pdfin,pdfin1)

    cfaci=API.cfac(**dataset_testii)
    cfdir=str(fit_pars.theories_path)+"/theory_"+str(fit_pars.theoryidi)+"/cfactor/"

    if 'CFNNLO' in cfaci:
        inputcf_cfnnlo=cfdir+"CF_CFNNLO_"+"grid-total-Z0_CMS-pT20"+".dat"
        distcf_cfnnlo=np.loadtxt(inputcf_cfnnlo,skiprows=9)
        distcf_cfnnlo=distcf_cfnnlo[:,0].flatten()
        pred*=distcf_cfnnlo

    xstot=np.sum(pred)*0.1/0.983

    if ifk == 0:
        predin=pred
    else:
        predin=xstot*np.ones((len(pred)))
    # print(pred)
    # print(xstot)

    return predin

# Calcualte theory  with ppbar value (current default nnpdf code does not allow)

def msht_theory_D0wasym_ind(ifk,pdf1,pdf2):

    # with contextlib.redirect_stdout(None):
    #     pdfin = lhapdf.mkPDF(pdf_pars.PDFlabel, 0)

    (pdfin,pdfin1)=pdf_caller(pdf1,pdf2)

    parton1=2212

    if ifk == 0:
        fktable_wp = pineappl.fk_table.FkTable.read(str(fit_pars.theories_path)+"/theory_"+str(fit_pars.theoryidi)+"/fastkernel/grid-D0wpf1-no-optimize_pbeam.pineappl.lz4")
        # pred= np.array(fktable_wp.convolve_with_one(parton1, pdfin.xfxQ2))
        pred=np.array(predi_calc(fktable_wp,pdfin,pdfin1))
    else:
        fktable_wm = pineappl.fk_table.FkTable.read(str(fit_pars.theories_path)+"/theory_"+str(fit_pars.theoryidi)+"/fastkernel/grid-D0wmf1-no-optimize_pbeam.pineappl.lz4")
        # pred= np.array(fktable_wm.convolve_with_one(parton1, pdfin.xfxQ2))
        pred=np.array(predi_calc(fktable_wm,pdfin,pdfin1))


    # print(pred_wp)
    # print(pred_wm)

    # pred=(pred_wp-pred_wm)/(pred_wp+pred_wm)

    # print(pred)

    return pred

def msht_theory_D0wasym():

    # with contextlib.redirect_stdout(None):
    #     pdfin = lhapdf.mkPDF(pdf_pars.PDFlabel, 0)

    pdfin =pdf_dict_check(pdf_pars.PDFlabel)


    fktable_wp = pineappl.fk_table.FkTable.read(str(fit_pars.theories_path)+"/theory_"+str(fit_pars.theoryidi)+"/fastkernel/grid-D0wpf1-no-optimize_pbeam.pineappl.lz4")
    fktable_wm = pineappl.fk_table.FkTable.read(str(fit_pars.theories_path)+"/theory_"+str(fit_pars.theoryidi)+"/fastkernel/grid-D0wmf1-no-optimize_pbeam.pineappl.lz4")

    parton1=2212
    pred_wp = np.array(fktable_wp.convolve_with_one(parton1, pdfin.xfxQ2))
    pred_wm = np.array(fktable_wm.convolve_with_one(parton1, pdfin.xfxQ2))

    # print(pred_wp)
    # print(pred_wm)

    pred=(pred_wp-pred_wm)/(pred_wp+pred_wm)

    # print(pred)

    return pred

# Replace first theory point with ppbar value (current default nnpdf code does not allow)

def msht_theory_ttbar(pred_in,pdf1,pdf2):

    predin=pred_in.copy()

    fktable = pineappl.fk_table.FkTable.read(str(fit_pars.theories_path)+"/theory_"+str(fit_pars.theoryidi)+"/fastkernel/grid-ttbar-tevnew_ppbar.pineappl.lz4")
    
    # print('ttbart')
    # with contextlib.redirect_stdout(None):
    #     pdfin = lhapdf.mkPDF(pdf_pars.PDFlabel, 0)

    # pdfin = lhapdf.mkPDF(pdf_pars.PDFlabel, 0)
    pdfin =pdf_dict_check(pdf_pars.PDFlabel)
  

    # print('ttbart1')

    parton1=-2212
    pred_ppbar = np.array(fktable.convolve_with_one(parton1, pdfin.xfxQ2))
    conv=0.04/4.
    pred_ppbar = pred_ppbar*conv

    pi=np.pi
    mt=172.5
 
    # pdfin = lhapdf.mkPDF('MSHT20nnlo_as118', 0)
    pdfin =pdf_dict_check('MSHT20nnlo_as118')

    alpha_s=pdfin.alphasQ2(mt*mt)
    
    knnlottbartev = 1. + 6.948*alpha_s/pi + 91.702*(alpha_s/pi)**2
    knlottbartev  = 1. + 6.948*alpha_s/pi
    cf_ttbar_tev=knnlottbartev/knlottbartev

    pred_ppbar = pred_ppbar*cf_ttbar_tev

    predin[0]=pred_ppbar

    

    return(predin)

def msht_kfs():

    for i in range(0,len(fit_pars.dataset_40)):
        dset=str(fit_pars.dataset_40[i]["dataset"])
        # cfac=str(fit_pars.dataset_40[i]["cfac"])
        cfacns=fit_pars.dataset_40[i]["cfac"]
        # if cfac == "['CFNNLO']":
        #     kfs_new(dset)
        if dset != 'MSHT-ATLAS_Z0_8TEV_3D_M-Y':
            for cfs in cfacns:
                if cfs == 'CFNNLO':
                    kfs_new(dset)
        # if dset == 'MSHT-TTbar-TOT_X-SEC':
        #     # kfs_MSHT_TTbar_TOT_X_SEC()
        #     kfs_new(dset)
        # if dset == 'MSHT-ATLASWZRAP36PB_ETA' and cfac == "['CFNNLO']":
        #     # kfs_MSHT_ATLASWZRAP36PB_ETA(i,dset)
        #     kfs_new(dset)
        # if dset == 'MSHT-CMS_WPWM_7TEV_ASY' and cfac == "['CFNNLO']":
        #     # kfs_MSHT_CMS_WPWM_7TEV_ASY(i)
        #     kfs_new(dset)
            

    
    return

def cfpath(fktable,cf):
    cf_dir=str(fit_pars.theories_path)+"/theory_"+str(fit_pars.theoryidi)+'/cfactor/CF_'
    path=cf_dir+cf+'_'+fktable+'.dat'

 
    return path

def kfs_new(dnam):

    ds = load_nnpdf.l.check_dataset(dnam, theoryid=fit_pars.theoryidi)

    # print(ds.fkspecs[0].metadata.FK_tables)


    for fkarr in ds.fkspecs[0].metadata.FK_tables:
        for fk in fkarr:
            # print(fk)
            inputcf_knlo=cfpath(str(fk),'Knlo')
 
            distcf_knlo=np.loadtxt(inputcf_knlo,skiprows=9)
            if distcf_knlo.ndim == 2:
                distcf_knlo=distcf_knlo[:,0].flatten()
                inputcf_knnlo=cfpath(str(fk),'Knnlo')
                distcf_knnlo=np.loadtxt(inputcf_knnlo,skiprows=9)
                distcf_knnlo=distcf_knnlo[:,0].flatten()
            else:
                distcf_knlo=np.array([distcf_knlo[0]])
                inputcf_knnlo=cfpath(str(fk),'Knnlo')
                distcf_knnlo=np.loadtxt(inputcf_knnlo,skiprows=9)
                distcf_knnlo=np.array([distcf_knnlo[0]])
            
        

            cfnnlo=distcf_knnlo/distcf_knlo
            outputdir=cfpath(str(fk),'CFNNLO')
            write_cf_arrayin(outputdir,cfnnlo,dnam)
            # print(cfnnlo)
    # os.quit()

    

def kfs_MSHT_ATLASWZRAP36PB_ETA(i,dnam):
    

    pi=np.pi        

    with contextlib.redirect_stdout(None):
        pdfin = lhapdf.mkPDF('MSHT20nnlo_as118', 0)
    alpha_s=pdfin.alphasQ2(8315.0)

    inptt = {                                                                                                                 
        "dataset_input": fit_pars.dataset_40[i],                                                                                      
        "use_cuts": "internal",                                                                                               
        "theoryid": fit_pars.theoryidi,                                                                         
    }  

    cd = API.commondata(**inptt)
    lcd = load_commondata(cd)
    kin=lcd.commondata_table["kin1"]
    kin_z=kin[0:8]
    kin_wm=kin[8:19]
    kin_wp=kin[19:30]

    cf_z=[]
    cf_wm=[]
    cf_wp=[]
    
    for y in kin_z:
        knnloz=1.+dz(y)*alpha_s/pi+ez(y)*np.power(alpha_s/pi,2)
        knloz=1.+dz(y)*alpha_s/pi
        cf_z=np.append(cf_z,knnloz/knloz)


    for y in kin_wm:
        knnlowm=1.+dm(y)*alpha_s/pi+em(y)*np.power(alpha_s/pi,2)
        knlowm=1.+dm(y)*alpha_s/pi
        cf_wm=np.append(cf_wm,knnlowm/knlowm)

    for y in kin_wp:
        knnlowp=1.+dp(y)*alpha_s/pi+ep(y)*np.power(alpha_s/pi,2)
        knlowp=1.+dp(y)*alpha_s/pi
        cf_wp=np.append(cf_wp,knnlowp/knlowp)


    # print(cf_z)
    # print(cf_wm)
    # print(cf_wp)

    output_dir=str(fit_pars.theories_path)+"/theory_"+str(fit_pars.theoryidi)+'/cfactor/'
    output_z=output_dir+'CF_CFNNLO_atlas-Z0-rapidity.dat'
    output_wm=output_dir+'CF_CFNNLO_atlas-Wminus-rapidity.dat'
    output_wp=output_dir+'CF_CFNNLO_atlas-Wplus-rapidity.dat'

    write_cf_arrayin(output_z,cf_z,'MSHT-ATLASWZRAP36PB_ETA')
    write_cf_arrayin(output_wm,cf_wm,'MSHT-ATLASWZRAP36PB_ETA')
    write_cf_arrayin(output_wp,cf_wp,'MSHT-ATLASWZRAP36PB_ETA')

    return

def write_cf_arrayin(output,cf_arr,label):

    write_cf_header(output,label)  
    with open(output,'a') as outputfile:
        for cf in cf_arr:
            outputfile.write(str(cf))
            outputfile.write(' 0.')
            outputfile.write('\n')

def dp(y):

    alphas0=0.11932  
    out=(knp(y)-1.)*np.pi/alphas0

    return out


def ep(y):

    alphas0=0.11932    
    out=(knnp(y)-knp(y))*np.power(np.pi/alphas0,2)

    return out

def dm(y):

    alphas0=0.11932  
    out=(knm(y)-1.)*np.pi/alphas0

    return out


def em(y):

    alphas0=0.11932    
    out=(knnm(y)-knm(y))*np.power(np.pi/alphas0,2)

    return out

def dz(y):

    alphas0=0.11707   
    out=(knz(y)-1.)*np.pi/alphas0

    return out

def ez(y):

    alphas0=0.11707     
    out=(knnz(y)-knz(y))*np.power(np.pi/alphas0,2)

    return out
 
def expand_quartic(a,y):

    out=a[0]+a[1]*y+a[2]*np.power(y,2)+a[3]*np.power(y,3)+a[4]*np.power(y,4)

    return out

def expand_cubic(a,y):

    out=a[0]+a[1]*y+a[2]*np.power(y,2)+a[3]*np.power(y,3)

    return out



def expand_poly5(a,y):

    out=a[0]+a[1]*np.power(y,a[2])+a[3]*np.power(y,a[4])

    return out

def knp(y):

    a=[0.2352374E+01,-0.1245037E+01,-0.4987484E-04, 0.3298243E-02, 0.1845026E+01]
    b=[0.1371318E+01, 0.3486271E+00,-0.1324011E+01,-0.5809094E+00,-0.5763467E+00]     

    if y < 1.7 :
      out=expand_poly5(a,y)
    else:
      out=expand_poly5(b,y)

    if y < 0.1:
        out=expand_poly5(a,0.1)

    return out

def knnp(y):

    a=[1.0909306,7.7507557E-3,-4.6027597E-4,7.30173886E-4  ]
 
    out=expand_cubic(a,y)

    return out

def knm(y):

    a=[1.1079977,-5.4418515E-4,6.136942E-3,-2.1376735E-3 ]
 
    out=expand_cubic(a,y)

    return out

def knnm(y):

    a=[0.1091280E+01,-0.9649422E-02,0.1622416E-01,-0.5229499E-02 ]
 
    out=expand_cubic(a,y)

    return out

def knz(y):

    a=[0.1076251E+01, 0.1833730E-01,-0.3368323E-01, 0.3309733E-01,-0.9615538E-02]
    b=[0.1103950E+01,-0.9414490E-01,0.1021254E+00,-0.3824836E-01, 0.4938694E-02]     

    if y < 1.7 :
      out=expand_quartic(a,y)
    else:
      out=expand_quartic(b,y)

    return out

def knnz(y):

    a=[1.074914,7.72668E-3,4.466234E-3,-1.391265E-3 ]
 
    out=expand_cubic(a,y)

    return out
      

def kfs_MSHT_TTbar_TOT_X_SEC():

    pi=np.pi

    mt=172.5
    with contextlib.redirect_stdout(None):
        pdfin = lhapdf.mkPDF('MSHT20nnlo_as118', 0)
    alpha_s=pdfin.alphasQ2(mt*mt)

    
    knnlottbarlhc8 = 1. + 13.984*alpha_s/pi + 152.36*np.power(alpha_s/pi,2)
    knlottbarlhc8  = 1. + 13.984*alpha_s/pi
    cf_ttbar_lhc8=knnlottbarlhc8/knlottbarlhc8 

    knnlottbartev = 1. + 6.948*alpha_s/pi + 91.702*(alpha_s/pi)**2
    knlottbartev  = 1. + 6.948*alpha_s/pi
    cf_ttbar_tev=knnlottbartev/knlottbartev

    knnlottbarlhc = 1. + 13.855*alpha_s/pi + 154.59*(alpha_s/pi)**2
    knlottbarlhc  = 1. + 13.855*alpha_s/pi
    cf_ttbar_lhc=knnlottbarlhc/knlottbarlhc

    # print(cf_ttbar_lhc,cf_ttbar_lhc8,cf_ttbar_tev)
    # write_cf_header('test.dat','test')

    output_dir=str(fit_pars.theories_path)+"/theory_"+str(fit_pars.theoryidi)+'/cfactor/'
    output_lhc=output_dir+'CF_NRM_grid-total-TTbar_ATLAS-7TeV.dat'
    output_lhc8=output_dir+'CF_NRM_grid-total-TTbar_rap34-1-_ATLAS-8TeV.dat'
    output_tev=output_dir+'CF_NRM_grid-ttbar-tevnew_ppbar.dat'

    write_cf_header(output_lhc,'MSHT-TTbar-TOT_X-SEC')  
    with open(output_lhc,'a') as outputfile:
        outputfile.write(str(cf_ttbar_lhc))
        outputfile.write(' 0.')

    write_cf_header(output_lhc8,'MSHT-TTbar-TOT_X-SEC')  
    with open(output_lhc8,'a') as outputfile:
        outputfile.write(str(cf_ttbar_lhc8))
        outputfile.write(' 0.')

    write_cf_header(output_tev,'MSHT-TTbar-TOT_X-SEC')  
    with open(output_tev,'a') as outputfile:
        outputfile.write(str(cf_ttbar_tev))
        outputfile.write(' 0.')

    return

def write_cf_header(output,dataset):

    with open(output,'w') as outputfile:
        outputfile.write('********************************************************************************')
        outputfile.write('\n')
        outputfile.write('SetName: ')
        outputfile.write(dataset)
        outputfile.write('\n')
        outputfile.write('Author: Lucian Harland-Lang (MSHT)')
        outputfile.write('\n')
        outputfile.write('Date: ')
        outputfile.write('\n')
        outputfile.write('CodesUsed: None')
        outputfile.write('\n')
        outputfile.write('TheoryInput: None')
        outputfile.write('\n')
        outputfile.write('PDFset: None')
        outputfile.write('\n')
        outputfile.write('Warnings: None')
        outputfile.write('\n')
        outputfile.write('********************************************************************************')
        outputfile.write('\n')
