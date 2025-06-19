from pathlib import Path
import shutil

import lhapdf
import numpy as np

from fixparpdf.global_pars import fit_pars, pdf_pars
from fixparpdf.pdfs import pdfs_msht


def initlha(name, lhdir):
    """Initialize a LHAPDF directory"""

    # dirin=pdf_pars.lhapdfdir+'NNPDF40_nnlo_pch_as_01180/'+'NNPDF40_nnlo_pch_as_01180.info'
    dirin = "input/MSHT20nnlo_as118_mem1.info"

    lhdir = Path(lhdir)
    dirlha = lhdir / name
    dirlha.mkdir(exist_ok=True, parents=True)

    shutil.copy(dirin, dirlha / f"{name}.info")


def writelha_end(name, lhdir, parin):
    """Write the PDF to the LHAPDF (lhdir) directory given some input parameters"""
    # TODO: take a PDF class instead
    output = lhdir + name + "/" + name + "_0000.dat"
    if fit_pars.irep < 10:
        output = lhdir + name + "/" + name + "_000" + str(fit_pars.irep) + ".dat"
    elif fit_pars.irep < 100:
        output = lhdir + name + "/" + name + "_00" + str(fit_pars.irep) + ".dat"
    else:
        output = lhdir + name + "/" + name + "_0" + str(fit_pars.irep) + ".dat"

    with open(output, "w") as outputfile:

        outputfile.write("PdfType: replica")
        outputfile.write("\n")
        outputfile.write("Format: lhagrid1")
        outputfile.write("\n")
        outputfile.write("---")

        #        PDFlabelmsht='MSHT20nnlo_as118'
        PDFlabelmsht = "NNPDF40_nnlo_pch_as_01180"
        inputf = pdf_pars.lhapdfdir + PDFlabelmsht + "/" + PDFlabelmsht + "_0000.dat"
        filein = open(inputf, "r")
        content = filein.readlines()

        ctest = filein.read().splitlines()

        # nnpdf
        xarr = np.loadtxt(inputf, skiprows=3, max_rows=1)
        # msht
        #        xarr=np.loadtxt(inputf,skiprows=3,max_rows=1)
        nx = len(xarr)

        outputfile.write("\n")
        # nnpdf
        outputfile.write(content[3])
        # msht
        #        outputfile.write(content[3])
        outputfile.write("1.000000E+00 1.5100000E+00")
        outputfile.write("\n")
        outputfile.write("-5 -4 -3 -2 -1 21 1 2 3 4 5")
        outputfile.write("\n")

        if pdf_pars.lhin:
            pset = lhapdf.getPDFSet(pdf_pars.PDFlabel_lhin)
            pdfs = pset.mkPDFs()

        pdfarr = np.zeros(11)

        for ix in range(0, nx):

            for iq in range(0, 2):
                xin = xarr[ix]
                #                qin=qarr[iq]
                qin = 1.00

                for i in range(-5, 0):
                    if pdf_pars.lhin:
                        # pdfout=pdfs[0].xfxQ(i,xin,qin)
                        pdfout = pdfs[fit_pars.lhrep].xfxQ(i, xin, qin)
                    else:
                        if iq == 0:
                            pdfout = pdfs_msht(i, parin, xin)
                        else:
                            pdfout = 1.0
                    pdfarr[i + 5] = pdfout

                for i in range(1, 6):
                    if pdf_pars.lhin:
                        # pdfout=pdfs[0].xfxQ(i,xin,qin)
                        pdfout = pdfs[fit_pars.lhrep].xfxQ(i, xin, qin)
                    else:
                        if iq == 0:
                            pdfout = pdfs_msht(i, parin, xin)
                        else:
                            pdfout = 1.0
                    pdfarr[i + 5] = pdfout

                if pdf_pars.lhin:
                    # pdfout=pdfs[0].xfxQ(21,xin,qin)
                    pdfout = pdfs[fit_pars.lhrep].xfxQ(21, xin, qin)
                else:
                    if iq == 0:
                        pdfout = pdfs_msht(0, parin, xin)
                    else:
                        pdfout = 1.0
                pdfarr[5] = pdfout

                np.savetxt(outputfile, pdfarr, fmt="%.7E", delimiter=" ", newline=" ")
                outputfile.write("\n")

        outputfile.write("---")
    print(f"PDF written to {output}")
