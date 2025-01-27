from global_pars import *
from pdfs import *
import lhapdf
import os
import shutil as sh

def initlha(name,lhdir):

    
#    dirin=pdf_pars.lhapdfdir+'test/'+'NNPDF40_nnlo_pch_as_01180.info'
    # dirin=pdf_pars.lhapdfdir+'MSHT20nnlo_as118/'+'MSHT20nnlo_as118_mem1.info'
    dirin='input/lha.info'
    
    dirlha=lhdir+name+'/'

    try: 
        os.mkdir(dirlha) 
    except OSError as error: 
        print(error)
        
    sh.copy(dirin,dirlha+name+'.info')

def writelha(name,lhdir,parin):

    output=lhdir+name+'/'+name+'_0000.dat'


    with open(output,'w') as outputfile:
    
        outputfile.write('PdfType: replica')                                                   
        outputfile.write('\n')
        outputfile.write('Format: lhagrid1')
        outputfile.write('\n')
        outputfile.write('---')

        xarr=dload_pars.xarr_tot
#       Add x=1 point as otherwise new code crashes!
        xarr=np.append(xarr,1.)

        nx=len(xarr)

        outputfile.write('\n')      
        np.savetxt(outputfile,xarr,fmt="%.14E",delimiter=' ', newline=' ')
        outputfile.write("\n")
 
        if fit_pars.theoryidi==211 or fit_pars.theoryidi==212 or fit_pars.theoryidi==40001000:
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

        for ix in range(0,nx):
            
            for iq in range(0,2):
                xin=xarr[ix]

                if fit_pars.theoryidi==211 or fit_pars.theoryidi==40001000:
                    qin=1.00
                elif fit_pars.theoryidi==200:
                    qin=1.65
                elif fit_pars.theoryidi==212:
                    qin=1.00
                
                for i in range(-5,0):
                    if(pdf_pars.lhin):
                        pdfout=pdfs[0].xfxQ(i,xin,qin)
                    else:
                        if iq==0:
                            pdfout=pdfs_msht(i,parin,xin)
                        else:
                            pdfout=1.
                    pdfarr[i+5]=pdfout
                    
                for i in range(1,6):
                    if(pdf_pars.lhin):
                        pdfout=pdfs[0].xfxQ(i,xin,qin)
                    else:
                        if iq==0:
                            pdfout=pdfs_msht(i,parin,xin)
                        else:
                            pdfout=1.
                    pdfarr[i+5]=pdfout

                if(pdf_pars.lhin):
                    pdfout=pdfs[0].xfxQ(21,xin,qin)
                else:
                    if iq==0:
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

        # using NNPDF x grid for final LHAPDF output for now

 #        PDFlabelmsht='MSHT20nnlo_as118'
        # PDFlabelmsht='NNPDF40_nnlo_pch_as_01180'
        # inputf=pdf_pars.lhapdfdir+PDFlabelmsht+'/'+PDFlabelmsht+'_0000.dat'

        inputf='input/xarr/lhax.dat'
        filein=open(inputf,"r")
        content=filein.readlines()
        
        ctest=filein.read().splitlines()

        xarr=np.loadtxt(inputf,max_rows=1)
        # nnpdf
        # xarr=np.loadtxt(inputf,skiprows=3,max_rows=1)
        # msht
#        xarr=np.loadtxt(inputf,skiprows=3,max_rows=1)
        nx=len(xarr)
    

        outputfile.write('\n')
        # nnpdf
        outputfile.write(content[0])
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
                        pdfout=pdfs[0].xfxQ(i,xin,qin)
                    else:
                        if iq==0:
                            pdfout=pdfs_msht(i,parin,xin)
                        else:
                            pdfout=1.
                    pdfarr[i+5]=pdfout
                    
                for i in range(1,6):
                    if(pdf_pars.lhin):
                        pdfout=pdfs[0].xfxQ(i,xin,qin)
                    else:
                        if iq==0:
                            pdfout=pdfs_msht(i,parin,xin)
                        else:
                            pdfout=1.
                    pdfarr[i+5]=pdfout

                if(pdf_pars.lhin):
                    pdfout=pdfs[0].xfxQ(21,xin,qin)
                else:
                    if iq==0:
                        pdfout=pdfs_msht(0,parin,xin)
                    else:
                        pdfout=1.
                pdfarr[5]=pdfout
            
                np.savetxt(outputfile,pdfarr,fmt="%.7E",delimiter=' ', newline=' ')
                outputfile.write('\n')

        outputfile.write('---')

def dellha(name):
    dirlha=pdf_pars.lhapdfdir+name+'/'
    sh.rmtree(dirlha)
