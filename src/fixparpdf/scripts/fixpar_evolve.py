# to be turned into evolution script

from argparse import ArgumentParser
from pathlib import Path
import shutil

from evolven3fit.evolve import evolve_fit
from reportengine.utils import yaml_safe
from validphys.loader import FallbackLoader

from fixparpdf.utils import existing_path, init_global_pars
from fixparpdf.outputs import EVGRIDS_F


def create_lhapdf(fit_folder, path_lhapdf):
    fit_folder = Path(fit_folder)
    path_lhapdf.mkdir(exist_ok=True)

    members = 0
    for i, replica_file in enumerate(fit_folder.glob(f"nnfit/replica_*/{fit_folder.name}.dat")):
        dest_path_replica = path_lhapdf / f"{fit_folder.stem}_{(i):04d}.dat"
        shutil.copy(replica_file, dest_path_replica)
        members += 1

    info_data = yaml_safe.load((fit_folder / "nnfit" / f"{fit_folder.name}.info").open("r"))
    info_data["NumMembers"] = members
    yaml_safe.dump(info_data, (path_lhapdf / f"{fit_folder.name}.info").open("w"))
    print(f"Congratulations, you will find you PDF at: {path_lhapdf.absolute().as_posix()}")


def main():
    parser = ArgumentParser()

    parser.add_argument("fit_runcard", type=existing_path, help="Runcard")
    parser.add_argument("--lhapdf-path", type=Path)
    args = parser.parse_args()

    # TODO: ensure that all the runcards go through global params in utils
    config = yaml_safe.load(args.fit_runcard.open("r"))
    init_global_pars(config)

    from fixparpdf.global_pars import inout_pars, chi2_pars

    if chi2_pars.dynamic_tol:
        fit_folder = EVGRIDS_F / f"{inout_pars.label}_dyntol"
    else:
        fit_folder = EVGRIDS_F / inout_pars.label

    if not fit_folder.exists():
        raise FileNotFoundError(f"Fit folder {fit_folder} not found")

    thID = int(config["theoryid"])

    # Use the FallBackLoader to download any missing PDFs/ekos/theories
    l = FallbackLoader()
    eko_path = l.check_eko(thID)
    evolve_fit(fit_folder, True, eko_path, hessian_fit=True)

    lhapath = args.lhapdf_path
    if lhapath is None:
        lhapath = fit_folder / fit_folder.name
    create_lhapdf(fit_folder, lhapath)


if __name__ == "__main__":
    main()
