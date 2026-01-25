# to be turned into evolution script

from argparse import ArgumentParser
from pathlib import Path

from evolven3fit.evolve import ExportGrid, evolve_exportgrids_into_lhapdf
from reportengine.utils import yaml_safe
from validphys.loader import FallbackLoader

from fppdf.outputs import EVGRIDS_F, FITFOLDER_NAME, MEMBER_NAME
from fppdf.utils import existing_path, init_global_pars


def main():
    parser = ArgumentParser()

    parser.add_argument("fit_runcard", type=existing_path, help="Runcard")
    parser.add_argument("--lhapdf-path", type=Path)
    args = parser.parse_args()

    # TODO: ensure that all the runcards go through global params in utils
    config = yaml_safe.load(args.fit_runcard.open("r"))
    init_global_pars(config)

    from fppdf.global_pars import chi2_pars, inout_pars

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

    # Prepare the target LHAPDF file
    lhapath = args.lhapdf_path
    if lhapath is None:
        lhapath = fit_folder / fit_folder.name

    exportgrids = []
    output_lhas = []
    for exportgrid_file in fit_folder.glob(
        f"{FITFOLDER_NAME}/{MEMBER_NAME}_*/{fit_folder.name}.exportgrid"
    ):
        data = yaml_safe.load(exportgrid_file.read_text(encoding="UTF-8"))
        exportgrids.append(ExportGrid(**data, hessian=True))
        # Take the member folder as the member number
        member = int(exportgrid_file.parent.stem.split("_")[-1]) - 1
        output_lhas.append(lhapath / f"{fit_folder.stem}_{(member):04d}.dat")

    info_path = fit_folder / FITFOLDER_NAME / f"{fit_folder.name}.info"
    print("Starting evolution...")
    evolve_exportgrids_into_lhapdf(eko_path, exportgrids, output_lhas, info_path)

    info_data = yaml_safe.load(info_path.open("r"))
    info_data["NumMembers"] = len(exportgrids)
    with (lhapath / f"{fit_folder.name}.info").open("w") as info_out:
        yaml_safe.dump(info_data, info_out)

    print(f"Congratulations, you will find you PDF at: {lhapath.absolute().as_posix()}")


if __name__ == "__main__":
    main()
