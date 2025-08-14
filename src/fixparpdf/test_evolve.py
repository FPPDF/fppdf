# to be turned into evolution script

import numpy as np
import sys
import shutil

from pathlib import Path
from validphys.loader import FallbackLoader
from evolven3fit.evolve import evolve_fit
from argparse import ArgumentParser, ArgumentTypeError
from fixparpdf.utils import _existing_path

import eko
import lhapdf

def create_lhapdf(fit_folder):
    fit_folder = Path(fit_folder)
    path_lhapdf = fit_folder / "lhapdf"  
    path_lhapdf.mkdir(exist_ok=True)
    for i, replica_file in enumerate(fit_folder.glob(f"nnfit/replica_*/{fit_folder.name}.dat")):
        dest_path_replica  = path_lhapdf / f"{fit_folder.stem}_{(i):04d}.dat"
        shutil.copy(replica_file, dest_path_replica)
    # you still need to copy the .info file

def main():
    parser = ArgumentParser()

    parser.add_argument("fit_folder", type=_existing_path, help="folder containong the fit")
    parser.add_argument("theoryID", type=str, help="theory ID")
    args = parser.parse_args()

    fit_folder = args.fit_folder
    thID = args.theoryID

    # Use the FallBackLoader to download any missing PDFs/ekos/theories
    l = FallbackLoader()
    eko_path = l.check_eko(thID)
    #fit_folder = Path("/data/theorie/tgiani/fixpar_nnpdf/outputs/evgrids/test_tg_shortmin_evscan_out")

    evolve_fit(fit_folder, True, eko_path, hessian_fit=True)
    create_lhapdf(fit_folder)

if __name__ == "__main__":
    main()
