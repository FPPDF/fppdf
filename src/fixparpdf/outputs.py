from pathlib import Path

import lhapdf
import numpy as np
from reportengine.utils import yaml_safe

from fixparpdf.global_pars import basis_pars, fit_pars, inout_pars, pdf_pars, shared_global_data
from fixparpdf.lhapdf_funs import initlha, writelha_end
from fixparpdf.pdfs import pdfs_msht

OUTPUT_F = Path("outputs")
BUFFER_F = OUTPUT_F / "buffer"
PARS_F = OUTPUT_F / "pars"
EVGRIDS_F = OUTPUT_F / "evgrids"
GRIDS_F = OUTPUT_F / "evgrids"
PLOTS_F = OUTPUT_F / "plots"
RES_F = OUTPUT_F / "res"
COV_F = OUTPUT_F / "cov"
for PF in [BUFFER_F, PARS_F, EVGRIDS_F, PLOTS_F, RES_F, COV_F, GRIDS_F]:
    PF.mkdir(exist_ok=True, parents=True)


def covmatout(hessi, jaci):

    #    hessin=hessi.copy()
    #    hessin=la.inv(hessin)
    pars = pdf_pars.pdfparsi.copy()
    output = COV_F / f"{inout_pars.label}.dat"

    print('call covmatout')
    print(output)

    with open(output, 'w') as outputfile:

        outputfile.write('npar=')
        outputfile.write('\n')
        outputfile.write(str(pdf_pars.npar_free))
        outputfile.write('\n')

        for i in range(0, 73):
            outputfile.write(str(pars[i]))
            outputfile.write(" ")
            outputfile.write(str(pdf_pars.par_isf[i]))
            outputfile.write(" ")
            outputfile.write("\n")

        outputfile.write('\n')

        for i in range(0, pdf_pars.npar_free):
            np.savetxt(outputfile, hessi[i, :], fmt="%.7E", delimiter=' ', newline=' ')
            outputfile.write('\n')

        #        outputfile.write('\n')

        #        for i in range(0,pdf_pars.npar_free):
        #            np.savetxt(outputfile,hessin[i,:],fmt="%.7E",delimiter=' ', newline=' ')
        #            outputfile.write('\n')

        outputfile.write('\n')

        for i in range(0, pdf_pars.npar_free):
            outputfile.write(str(jaci[i]))
            outputfile.write("\n")


def evgrido(pdf_name=None):
    """Create the exportgrid files for evolution
    If a pdf_name is not given, use the label of the fit.
    Note that the output for each member is written down to a separated "replica" folder
    in order to use the evolution routines from NNPDF without changes.
    """
    if pdf_name is None:
        if inout_pars.pd_output:
            pdf_name = inout_pars.pd_output_lab
        else:
            pdf_name = inout_pars.label

    dirgrid = EVGRIDS_F / pdf_name / "nnfit" / f"replica_{str(fit_pars.irep+1)}"
    dirgrid.mkdir(exist_ok=True, parents=True)
    output = dirgrid / f"{pdf_name}.exportgrid"

    xgrid = np.array(
        [
            1.00000000000000e-09,
            1.29708482343957e-09,
            1.68242903474527e-09,
            2.18225315420583e-09,
            2.83056741739819e-09,
            3.67148597892941e-09,
            4.76222862935315e-09,
            6.17701427376180e-09,
            8.01211109898438e-09,
            1.03923870607245e-08,
            1.34798064073805e-08,
            1.74844503691778e-08,
            2.26788118881103e-08,
            2.94163370300835e-08,
            3.81554746595878e-08,
            4.94908707232129e-08,
            6.41938295708371e-08,
            8.32647951986859e-08,
            1.08001422993829e-07,
            1.40086873081130e-07,
            1.81704331793772e-07,
            2.35685551545377e-07,
            3.05703512595323e-07,
            3.96522309841747e-07,
            5.14321257236570e-07,
            6.67115245136676e-07,
            8.65299922973143e-07,
            1.12235875241487e-06,
            1.45577995547683e-06,
            1.88824560514613e-06,
            2.44917352454946e-06,
            3.17671650028717e-06,
            4.12035415232797e-06,
            5.34425265752090e-06,
            6.93161897806315e-06,
            8.99034258238145e-06,
            1.16603030112258e-05,
            1.51228312288769e-05,
            1.96129529349212e-05,
            2.54352207134502e-05,
            3.29841683435992e-05,
            4.27707053972016e-05,
            5.54561248105849e-05,
            7.18958313632514e-05,
            9.31954227979614e-05,
            1.20782367731330e-04,
            1.56497209466554e-04,
            2.02708936328495e-04,
            2.62459799331951e-04,
            3.39645244168985e-04,
            4.39234443000422e-04,
            5.67535660104533e-04,
            7.32507615725537e-04,
            9.44112105452451e-04,
            1.21469317686978e-03,
            1.55935306118224e-03,
            1.99627451141338e-03,
            2.54691493736552e-03,
            3.23597510213126e-03,
            4.09103436509565e-03,
            5.14175977083962e-03,
            6.41865096062317e-03,
            7.95137940306351e-03,
            9.76689999624100e-03,
            1.18876139251364e-02,
            1.43298947643919e-02,
            1.71032279460271e-02,
            2.02100733925079e-02,
            2.36463971369542e-02,
            2.74026915728357e-02,
            3.14652506132444e-02,
            3.58174829282429e-02,
            4.04411060163317e-02,
            4.53171343973807e-02,
            5.04266347950069e-02,
            5.57512610084339e-02,
            6.12736019390519e-02,
            6.69773829498255e-02,
            7.28475589986517e-02,
            7.88703322292727e-02,
            8.50331197801452e-02,
            9.13244910278679e-02,
            9.77340879783772e-02,
            1.04252538208639e-01,
            1.10871366547237e-01,
            1.17582909372878e-01,
            1.24380233801599e-01,
            1.31257062945031e-01,
            1.38207707707289e-01,
            1.45227005135651e-01,
            1.52310263065985e-01,
            1.59453210652156e-01,
            1.66651954293987e-01,
            1.73902938455578e-01,
            1.81202910873333e-01,
            1.88548891679097e-01,
            1.95938145999193e-01,
            2.03368159629765e-01,
            2.10836617429103e-01,
            2.18341384106561e-01,
            2.25880487124065e-01,
            2.33452101459503e-01,
            2.41054536011681e-01,
            2.48686221452762e-01,
            2.56345699358723e-01,
            2.64031612468684e-01,
            2.71742695942783e-01,
            2.79477769504149e-01,
            2.87235730364833e-01,
            2.95015546847664e-01,
            3.02816252626866e-01,
            3.10636941519503e-01,
            3.18476762768082e-01,
            3.26334916761672e-01,
            3.34210651149156e-01,
            3.42103257303627e-01,
            3.50012067101685e-01,
            3.57936449985571e-01,
            3.65875810279643e-01,
            3.73829584735962e-01,
            3.81797240286494e-01,
            3.89778271981947e-01,
            3.97772201099286e-01,
            4.05778573402340e-01,
            4.13796957540671e-01,
            4.21826943574548e-01,
            4.29868141614175e-01,
            4.37920180563205e-01,
            4.45982706956990e-01,
            4.54055383887562e-01,
            4.62137890007651e-01,
            4.70229918607142e-01,
            4.78331176755675e-01,
            4.86441384506059e-01,
            4.94560274153348e-01,
            5.02687589545177e-01,
            5.10823085439086e-01,
            5.18966526903235e-01,
            5.27117688756998e-01,
            5.35276355048428e-01,
            5.43442318565661e-01,
            5.51615380379768e-01,
            5.59795349416641e-01,
            5.67982042055800e-01,
            5.76175281754088e-01,
            5.84374898692498e-01,
            5.92580729444440e-01,
            6.00792616663950e-01,
            6.09010408792398e-01,
            6.17233959782450e-01,
            6.25463128838069e-01,
            6.33697780169485e-01,
            6.41937782762089e-01,
            6.50183010158361e-01,
            6.58433340251944e-01,
            6.66688655093089e-01,
            6.74948840704708e-01,
            6.83213786908386e-01,
            6.91483387159697e-01,
            6.99757538392251e-01,
            7.08036140869916e-01,
            7.16319098046733e-01,
            7.24606316434025e-01,
            7.32897705474271e-01,
            7.41193177421404e-01,
            7.49492647227008e-01,
            7.57796032432224e-01,
            7.66103253064927e-01,
            7.74414231541921e-01,
            7.82728892575836e-01,
            7.91047163086478e-01,
            7.99368972116378e-01,
            8.07694250750291e-01,
            8.16022932038457e-01,
            8.24354950923382e-01,
            8.32690244169987e-01,
            8.41028750298844e-01,
            8.49370409522600e-01,
            8.57715163684985e-01,
            8.66062956202683e-01,
            8.74413732009721e-01,
            8.82767437504206e-01,
            8.91124020497459e-01,
            8.99483430165226e-01,
            9.07845617001021e-01,
            9.16210532771399e-01,
            9.24578130473112e-01,
            9.32948364292029e-01,
            9.41321189563734e-01,
            9.49696562735755e-01,
            9.58074441331298e-01,
            9.66454783914439e-01,
            9.74837550056705e-01,
            9.83222700304978e-01,
            9.91610196150662e-01,
            1.00000000000000e00,
        ]
    ).reshape(-1, 1)

    test = np.zeros([196, 14])

    if pdf_pars.lhin:
        pset = lhapdf.getPDFSet(pdf_pars.PDFlabel_lhin)
        pdfs = pset.mkPDFs()

    qin = shared_global_data["data"].q20

    for i, x in enumerate(xgrid.flat):

        if pdf_pars.lhin:
            test[i, 0] = 0.0
            test[i, 1] = 0.0
            test[i, 2] = pdfs[0].xfxQ(-4, x, qin)
            test[i, 3] = pdfs[0].xfxQ(-3, x, qin)
            test[i, 4] = pdfs[0].xfxQ(-2, x, qin)
            test[i, 5] = pdfs[0].xfxQ(-1, x, qin)
            test[i, 6] = pdfs[0].xfxQ(0, x, qin)
            test[i, 7] = pdfs[0].xfxQ(1, x, qin)
            test[i, 8] = pdfs[0].xfxQ(2, x, qin)
            test[i, 9] = pdfs[0].xfxQ(3, x, qin)
            test[i, 10] = pdfs[0].xfxQ(4, x, qin)
            test[i, 11] = 0.0
            test[i, 12] = 0.0
            test[i, 13] = 0.0
        else:
            test[i, 0] = 0.0
            test[i, 1] = 0.0
            test[i, 2] = pdfs_msht(-4, pdf_pars.pdfparsi, x)
            test[i, 3] = pdfs_msht(-3, pdf_pars.pdfparsi, x)
            test[i, 4] = pdfs_msht(-2, pdf_pars.pdfparsi, x)
            test[i, 5] = pdfs_msht(-1, pdf_pars.pdfparsi, x)
            test[i, 6] = pdfs_msht(0, pdf_pars.pdfparsi, x)
            test[i, 7] = pdfs_msht(1, pdf_pars.pdfparsi, x)
            test[i, 8] = pdfs_msht(2, pdf_pars.pdfparsi, x)
            test[i, 9] = pdfs_msht(3, pdf_pars.pdfparsi, x)
            test[i, 10] = pdfs_msht(4, pdf_pars.pdfparsi, x)
            test[i, 11] = 0.0
            test[i, 12] = 0.0
            test[i, 13] = 0.0

    # Why is this one separated?
    xlast = xgrid[195][0]
    if pdf_pars.lhin:
        test[195, 0] = 0.0
        test[195, 1] = 0.0
        test[195, 2] = pdfs[0].xfxQ(-4, xlast, qin)
        test[195, 3] = pdfs[0].xfxQ(-3, xlast, qin)
        test[195, 4] = pdfs[0].xfxQ(-2, xlast, qin)
        test[195, 5] = pdfs[0].xfxQ(-1, xlast, qin)
        test[195, 6] = pdfs[0].xfxQ(0, xlast, qin)
        test[195, 7] = pdfs[0].xfxQ(1, xlast, qin)
        test[195, 8] = pdfs[0].xfxQ(2, xlast, qin)
        test[195, 9] = pdfs[0].xfxQ(3, xlast, qin)
        test[195, 10] = pdfs[0].xfxQ(4, xlast, qin)
        test[195, 11] = 0.0
        test[195, 12] = 0.0
        test[195, 13] = 0.0
    else:
        test[195, 0] = 0.0
        test[195, 1] = 0.0
        test[195, 2] = pdfs_msht(-4, pdf_pars.pdfparsi, xlast)
        test[195, 3] = pdfs_msht(-3, pdf_pars.pdfparsi, xlast)
        test[195, 4] = pdfs_msht(-2, pdf_pars.pdfparsi, xlast)
        test[195, 5] = pdfs_msht(-1, pdf_pars.pdfparsi, xlast)
        test[195, 6] = pdfs_msht(0, pdf_pars.pdfparsi, xlast)
        test[195, 7] = pdfs_msht(1, pdf_pars.pdfparsi, xlast)
        test[195, 8] = pdfs_msht(2, pdf_pars.pdfparsi, xlast)
        test[195, 9] = pdfs_msht(3, pdf_pars.pdfparsi, xlast)
        test[195, 10] = pdfs_msht(4, pdf_pars.pdfparsi, xlast)
        test[195, 11] = 0.0
        test[195, 12] = 0.0
        test[195, 13] = 0.0

    data = {
        "replica": 1,  # fake
        "q20": qin**2,
        "xgrid": xgrid.T.tolist()[0],
        "labels": [
            "TBAR",
            "BBAR",
            "CBAR",
            "SBAR",
            "UBAR",
            "DBAR",
            "GLUON",
            "D",
            "U",
            "S",
            "C",
            "B",
            "T",
            "PHT",
        ],
        "pdfgrid": test.tolist(),
    }

    with open(output, 'w') as outputfile:
        yaml_safe.dump(data, outputfile)


def resout_nofit(pospeni, chi2t0i, chi2expi, n):
    outputfile = open('outputs/res/' + inout_pars.label + '.dat', 'w')
    outputfile.write("chi2(t0) in = ")
    outputfile.write(str(chi2t0i))
    outputfile.write("\n")
    outputfile.write("chi2(exp) in= ")
    outputfile.write(str(chi2expi))
    outputfile.write("\n")
    outputfile.write("ndat = ")
    outputfile.write(str(n))
    outputfile.write("\n")
    outputfile.write("Pos penalty in = ")
    outputfile.write(str(pospeni))
    outputfile.write("\n")


def resout(pospeni, pospenf, chi2t0i, chi2expi, chi2t0f, chi2expf, n):
    outputfile = open('outputs/res/' + inout_pars.label + '.dat', 'w')
    outputfile.write("chi2(t0) in = ")
    outputfile.write(str(chi2t0i))
    outputfile.write("\n")
    outputfile.write("chi2(exp) in= ")
    outputfile.write(str(chi2expi))
    outputfile.write("\n")
    outputfile.write("chi2(t0) out = ")
    outputfile.write(str(chi2t0f))
    outputfile.write("\n")
    outputfile.write("chi2(exp) out = ")
    outputfile.write(str(chi2expf))
    outputfile.write("\n")
    outputfile.write("ndat = ")
    outputfile.write(str(n))
    outputfile.write("\n")
    outputfile.write("Pos penalty in = ")
    outputfile.write(str(pospeni))
    outputfile.write("\n")
    outputfile.write("Pos penalty out = ")
    outputfile.write(str(pospenf))
    outputfile.write("\n")


def parsout(output_filename=None):
    """Write down the parameters of the PDF fit.
    If an output filename is not given, it will default to the label.
    """
    if output_filename is None:
        output_filename = inout_pars.label

    outputfile = (PARS_F / output_filename).with_suffix(".dat").open("w")

    pars = pdf_pars.pdfparsi.copy()
    # auv=pars[0:9]
    auv = pars[basis_pars.i_uv_min : basis_pars.i_uv_max].copy()
    auv = np.delete(auv, 0)
    # adv=pars[9:18]
    adv = pars[basis_pars.i_dv_min : basis_pars.i_dv_max].copy()
    adv = np.delete(adv, 0)
    # asea=pars[18:27]
    asea = pars[basis_pars.i_sea_min : basis_pars.i_sea_max].copy()
    # asp=pars[27:36]
    asp = pars[basis_pars.i_sp_min : basis_pars.i_sp_max].copy()
    # ag=pars[36:46]
    ag = pars[basis_pars.i_g_min : basis_pars.i_g_max].copy()
    ag = np.delete(ag, 0)
    # asm=pars[46:56]
    asm = pars[basis_pars.i_sm_min : basis_pars.i_sm_max].copy()
    asm = np.delete(asm, 3)
    # adbub=pars[56:64]
    adbub = pars[basis_pars.i_dbub_min : basis_pars.i_dbub_max].copy()
    # charm=pars[64:73]
    charm = pars[basis_pars.i_ch_min : basis_pars.i_ch_max].copy()

    outputfile.write("uv parameters (delu,etau,cu1-6)")
    outputfile.write("\n")
    for i in range(0, len(auv)):
        outputfile.write(str(auv[i]))
        outputfile.write(" ")
        outputfile.write(str(0))
        outputfile.write(" ")
        outputfile.write("\n")

    outputfile.write("dv parameters (deld,etad,cd1-6)")
    outputfile.write("\n")
    for i in range(0, len(adv)):
        outputfile.write(str(adv[i]))
        outputfile.write(" ")
        outputfile.write(str(0))
        outputfile.write(" ")
        outputfile.write("\n")

    outputfile.write("sea parameters (AS,delS,etaS,cS1-6)")
    outputfile.write("\n")
    for i in range(0, len(asea)):
        outputfile.write(str(asea[i]))
        outputfile.write(" ")
        outputfile.write(str(0))
        outputfile.write(" ")
        outputfile.write("\n")

    outputfile.write("s+ parameters (Asp,delsp,etasp,csp1-6)")
    outputfile.write("\n")
    for i in range(0, len(asp)):
        outputfile.write(str(asp[i]))
        outputfile.write(" ")
        outputfile.write(str(0))
        outputfile.write(" ")
        outputfile.write("\n")

    outputfile.write("Gluon parameters (etagp,delgp,cg1-4,etagm,delgm)")
    outputfile.write("\n")
    for i in range(0, len(ag)):
        outputfile.write(str(ag[i]))
        outputfile.write(" ")
        outputfile.write(str(0))
        outputfile.write(" ")
        outputfile.write("\n")

    outputfile.write("s- parameters (Asm,delsm,etasm, cs1-4)")
    outputfile.write("\n")
    for i in range(0, len(asm)):
        outputfile.write(str(asm[i]))
        outputfile.write(" ")
        outputfile.write(str(0))
        outputfile.write(" ")
        outputfile.write("\n")

    outputfile.write("db/ub parameters (Arho,etarho,crho1-6)")
    outputfile.write("\n")
    for i in range(0, len(adbub)):
        outputfile.write(str(adbub[i]))
        outputfile.write(" ")
        outputfile.write(str(0))
        outputfile.write(" ")
        outputfile.write("\n")

    outputfile.write("charm parameters (Ac,etac,cc1-6)")
    outputfile.write("\n")
    for i in range(0, len(charm)):
        outputfile.write(str(charm[i]))
        outputfile.write(" ")
        outputfile.write(str(0))
        outputfile.write(" ")
        outputfile.write("\n")


def plotout():

    output = PLOTS_F / f"{inout_pars.label}.dat"

    msht = 'MSHT20nnlo_as118'
    nnpdf = 'NNPDF40_nnlo_pch_as_01180'
    pset_msht = lhapdf.getPDFSet(msht)
    lh_msht = pset_msht.mkPDFs()

    pset_nnpdf = lhapdf.getPDFSet(nnpdf)
    lh_nnpdf = pset_nnpdf.mkPDFs()

    nx = 100
    xmin = 1e-6
    xmax = 0.99
    lxmin = np.log(xmin)
    lxmax = np.log(xmax)
    qin = 1.0

    pdfarr = np.zeros(22)

    with open(output, 'w') as outputfile:

        outputfile.write('x,fit(-3:3),msht20nnlo(-3:3),nnpdf40pchnnlo(-3:3)')
        outputfile.write('\n')

        for ix in range(0, nx):
            lx = lxmin + (lxmax - lxmin) * ix / nx
            xin = np.exp(lx)

            pdfarr[0] = xin

            for i in range(-3, 0):
                pdfout = pdfs_msht(i, pdf_pars.pdfparsi, xin)
                pdfarr[i + 4] = pdfout

                pdfarr[i + 11] = lh_msht[0].xfxQ(i, xin, qin)
                pdfarr[i + 18] = lh_nnpdf[0].xfxQ(i, xin, qin)

            for i in range(1, 4):
                pdfout = pdfs_msht(i, pdf_pars.pdfparsi, xin)
                pdfarr[i + 4] = pdfout

                pdfarr[i + 11] = lh_msht[0].xfxQ(i, xin, qin)
                pdfarr[i + 18] = lh_nnpdf[0].xfxQ(i, xin, qin)

            pdfout = pdfs_msht(0, pdf_pars.pdfparsi, xin)
            pdfarr[4] = pdfout
            pdfarr[11] = lh_msht[0].xfxQ(21, xin, qin)
            pdfarr[18] = lh_nnpdf[0].xfxQ(i, xin, qin)

            np.savetxt(outputfile, pdfarr, fmt="%.7E", delimiter=' ', newline=' ')
            outputfile.write('\n')


def gridout(pdf_name=None):
    """Write down the grid in the grids output folder for the given pdf_name
    If a pdf_name is not given, use the label of the fit."""
    if pdf_name is None:
        pdf_name = inout_pars.label

    initlha(pdf_name, GRIDS_F)
    pdf_pars.PDFlabel = pdf_name
    writelha_end(pdf_name, GRIDS_F, pdf_pars.pdfparsi)
