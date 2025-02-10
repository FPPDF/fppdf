from pathlib import Path
from global_pars import *
from pdfs import *
import lhapdf
import os
import shutil as sh


def initlha(name,lhdir):

    
    # dirin=pdf_pars.lhapdfdir+'NNPDF40_nnlo_pch_as_01180/'+'NNPDF40_nnlo_pch_as_01180.info'
    dirin='input/MSHT20nnlo_as118_mem1.info'

    lhdir = Path(lhdir)
    dirlha = lhdir / name 
    dirlha.mkdir(exist_ok = True, parents=True)
        
    sh.copy(dirin, dirlha / f"{name}.info")

def writelha(name,lhdir,parin):
    print(f"Writing {name}")

    lhdir = Path(lhdir)
    output = lhdir / name / f"{name}_0000.dat"

    with output.open("w") as outputfile:
        outputfile.write('PdfType: replica')                                                   
        outputfile.write('\n')
        outputfile.write('Format: lhagrid1')
        outputfile.write('\n')
        outputfile.write('---')

 #        PDFlabelmsht='MSHT20nnlo_as118'
        PDFlabelmsht='NNPDF40_nnlo_pch_as_01180'
        inputf=Path(pdf_pars.lhapdfdir) / PDFlabelmsht / f"{PDFlabelmsht}_0000.dat"
        filein=open(inputf,"r")
        content=filein.readlines()
        
        ctest=filein.read().splitlines()

        # nnpdf
#        xarr=np.loadtxt(inputf,skiprows=4,max_rows=1)
        # msht
        xarr=np.loadtxt(inputf,skiprows=3,max_rows=1)
        



        nx=1000
        # nx=250

        xmin=1e-9
        xmax=1.0
        lxmin=np.log(xmin)
        lxmax=np.log(xmax)

        xarr=np.zeros(1)

        for i in range(0,nx+1):
            lx=lxmin+(lxmax-lxmin)*i/nx
            x=np.exp(lx)
            xarr=np.append(xarr,x)

        # xarr=np.delete(xarr,0)

        xarr=dload_pars.xarr_tot

#       Add x=1 point as otherwise new code crashes!
        if fit_pars.theoryidi==40001000 or fit_pars.theoryidi==50001000 :
            xarr=np.append(xarr,1.)
            # dload_pars.xarr_tot=xarr
        
        nx=len(xarr)

        
        
        outputfile.write('\n')
        # nnpdf
#         outputfile.write(content[4])        
        np.savetxt(outputfile,xarr,fmt="%.14E",delimiter=' ', newline=' ')
        outputfile.write("\n")
        # msht
        # outputfile.write(content[3])
        if fit_pars.theoryidi==211 or fit_pars.theoryidi==212 or fit_pars.theoryidi==40001000 or fit_pars.theoryidi==50001000:
            outputfile.write('1.000000E+00 1.5100000E+00')
        elif fit_pars.theoryidi==200:
            outputfile.write('1.650000E+00 2.0000000E+00')
        outputfile.write("\n")
        outputfile.write('-5 -4 -3 -2 -1 21 1 2 3 4 5')
        outputfile.write('\n')

        
        if(pdf_pars.lhin):
            pset = lhapdf.getPDFSet(pdf_pars.PDFlabel_lhin)
            pdfs = pset.mkPDFs()

        pdfarr=np.zeros(11)

#        x0=0.99161022901535034
#        x0=9.9161020e-01

#        pdfout_d=pdfs_msht(1,parin,x0)
#        pdfout_dbar=pdfs_msht(-1,parin,x0)    

#        dv=pdfout_d-pdfout_dbar
#        print(dv,pdfout_d,pdfout_dbar)
#        os.quit()

        for ix in range(0,nx):
            for iq in range(0,2):
                xin=xarr[ix]


                if fit_pars.theoryidi==211 or fit_pars.theoryidi==40001000 or fit_pars.theoryidi==50001000:
                    qin=1.00
                elif fit_pars.theoryidi==200:
                    qin=1.65
                elif fit_pars.theoryidi==212:
                    qin=1.00
                
                for i in range(-5,0):
                    if(pdf_pars.lhin):
                        # pdfout=pdfs[0].xfxQ(i,xin,qin)
                        pdfout=pdfs[fit_pars.lhrep].xfxQ(i,xin,qin)
                    else:
                        if iq==0:
                            if fit_pars.newmin and chi2_pars.ipdf_newmin > 0:
                                pdfout=pdfs_diff(i,xin)
                            else:
                                pdfout=pdfs_msht(i,parin,xin)
                        else:
                            pdfout=1.
                    pdfarr[i+5]=pdfout
                    
                for i in range(1,6):
                    if(pdf_pars.lhin):
                        pdfout=pdfs[fit_pars.lhrep].xfxQ(i,xin,qin)
                        # pdfout=pdfs[0].xfxQ(i,xin,qin)
                    else:
                        if iq==0:
                            if fit_pars.newmin and chi2_pars.ipdf_newmin > 0:
                                pdfout=pdfs_diff(i,xin)
                            else:
                                pdfout=pdfs_msht(i,parin,xin)
                        else:
                            pdfout=1.
                    pdfarr[i+5]=pdfout

                if(pdf_pars.lhin):
                    pdfout=pdfs[fit_pars.lhrep].xfxQ(21,xin,qin)
                else:
                    if iq==0:
                        if fit_pars.newmin and chi2_pars.ipdf_newmin > 0:
                            pdfout=pdfs_diff(0,xin)
                        else:
                            pdfout=pdfs_msht(0,parin,xin)
                    else:
                        pdfout=1.
                pdfarr[5]=pdfout
            
#                np.savetxt(outputfile,pdfarr,fmt="%.7E",delimiter=' ', newline=' ')
                np.savetxt(outputfile,pdfarr,fmt="%.14E",delimiter=' ', newline=' ')
                outputfile.write('\n')

        outputfile.write('---')

def writelha_end(name,lhdir,parin):
    
    output=lhdir+name+'/'+name+'_0000.dat'
    if fit_pars.irep < 10:
        output=lhdir+name+'/'+name+'_000'+str(fit_pars.irep)+'.dat'
    elif fit_pars.irep < 100:
        output=lhdir+name+'/'+name+'_00'+str(fit_pars.irep)+'.dat'
    else:
        output=lhdir+name+'/'+name+'_0'+str(fit_pars.irep)+'.dat'

    with open(output,'w') as outputfile:
    
        outputfile.write('PdfType: replica')                                                   
        outputfile.write('\n')
        outputfile.write('Format: lhagrid1')
        outputfile.write('\n')
        outputfile.write('---')

 #        PDFlabelmsht='MSHT20nnlo_as118'
        PDFlabelmsht='NNPDF40_nnlo_pch_as_01180'
        inputf=pdf_pars.lhapdfdir+PDFlabelmsht+'/'+PDFlabelmsht+'_0000.dat'
        filein=open(inputf,"r")
        content=filein.readlines()
        
        ctest=filein.read().splitlines()

        # nnpdf
        xarr=np.loadtxt(inputf,skiprows=3,max_rows=1)
        # msht
#        xarr=np.loadtxt(inputf,skiprows=3,max_rows=1)
        nx=len(xarr)
    

        outputfile.write('\n')
        # nnpdf
        outputfile.write(content[3])
        # msht
#        outputfile.write(content[3])
        outputfile.write('1.000000E+00 1.5100000E+00')
        outputfile.write("\n")
        outputfile.write('-5 -4 -3 -2 -1 21 1 2 3 4 5')
        outputfile.write('\n')

        
        if(pdf_pars.lhin):
            pset = lhapdf.getPDFSet(pdf_pars.PDFlabel_lhin)
            pdfs = pset.mkPDFs()

        pdfarr=np.zeros(11)

        
        for ix in range(0,nx):
            
            for iq in range(0,2):
                xin=xarr[ix]
#                qin=qarr[iq]
                qin=1.00
                
                for i in range(-5,0):
                    if(pdf_pars.lhin):
                        # pdfout=pdfs[0].xfxQ(i,xin,qin)
                        pdfout=pdfs[fit_pars.lhrep].xfxQ(i,xin,qin)
                    else:
                        if iq==0:
                            pdfout=pdfs_msht(i,parin,xin)
                        else:
                            pdfout=1.
                    pdfarr[i+5]=pdfout
                    
                for i in range(1,6):
                    if(pdf_pars.lhin):
                        # pdfout=pdfs[0].xfxQ(i,xin,qin)
                        pdfout=pdfs[fit_pars.lhrep].xfxQ(i,xin,qin)
                    else:
                        if iq==0:
                            pdfout=pdfs_msht(i,parin,xin)
                        else:
                            pdfout=1.
                    pdfarr[i+5]=pdfout

                if(pdf_pars.lhin):
                    # pdfout=pdfs[0].xfxQ(21,xin,qin)
                    pdfout=pdfs[fit_pars.lhrep].xfxQ(21,xin,qin)
                else:
                    if iq==0:
                        pdfout=pdfs_msht(0,parin,xin)
                    else:
                        pdfout=1.
                pdfarr[5]=pdfout
            
                np.savetxt(outputfile,pdfarr,fmt="%.7E",delimiter=' ', newline=' ')
                outputfile.write('\n')

        outputfile.write('---')
