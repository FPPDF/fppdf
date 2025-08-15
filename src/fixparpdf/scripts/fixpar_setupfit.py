#!/usr/bin/env python
"""
    Only necessary for the generation of the theory covariance matrix in the relevant subfolder,
    this is an extension of nnpdf's setupfit script for fixed parametrization.

    The extension is done by:
    1) Removing the n3fit actions and checks that are not necessary here
    2) Modifying the output paths of the script to point to what fixpar is expecting

"""
from pathlib import Path
import shutil
import tempfile

# Import directly from the script to make sure any possible changes are propagated
from n3fit.scripts.vp_setupfit import ConfigError, SetupFitApp, SetupFitConfig


class FixparSetupConfig(SetupFitConfig):

    @classmethod
    def from_yaml(cls, o, *args, **kwargs):
        # Let n3fit do the heavy lifting with the parsing
        file_content = super().from_yaml(o, *args, **kwargs).input_params

        # Check the call to this script is sensible
        thconfig = file_content.get('theorycovmatconfig', {})
        if not thconfig:
            raise ConfigError(
                "This scripts needs to be called only for the construction of the theory covariance matrix"
            )

        # And just add some extra defaults that internal functions might need
        thid = file_content["theoryid"]
        file_content.setdefault("theory", {"theoryid": thid})

        # And remove the checks we are not interested in
        file_content["actions_"] = ["datacuts::theory::theorycovmatconfig nnfit_theory_covmat"]

        return cls(file_content, *args, **kwargs)


class FixparSetupApp(SetupFitApp):
    """Ensures that the theory covariance matrix is saved under
    outputs / thcovmat / <fit label>.csv

    Note that validphys will first write the table under a temporary
    directory and this class will then move the tables to the right place.
    """

    config_class = FixparSetupConfig

    def get_commandline_arguments(self, cmdline=None):
        """Save all results in a temporary directory to be moved at the end"""
        # Skip the one level we want to override, but get its parents method
        args = super(SetupFitApp, self).get_commandline_arguments(cmdline)
        args["output"] = Path(tempfile.mkdtemp())
        return args

    def run(self):
        super().run()

        output_label = self.get_config().input_params["inout_parameters"].get("label")
        output_covmat = Path("outputs") / "thcovmat" / f"{output_label}.csv"
        output_covmat.parent.mkdir(exist_ok=True, parents=True)

        # We are interested in the full theory covmat not on each component separately
        covmat_file = (
            self.environment.table_folder
            / "datacuts_theory_theorycovmatconfig_theory_covmat_custom.csv"
        )
        if not covmat_file.exists():
            raise FileNotFoundError("No theory covmat found!!")
        shutil.copy(covmat_file, output_covmat)

        print(
            f"Theory covariance matrix can be found at {output_covmat}, now you can run your fit normally."
        )


def main():
    FixparSetupApp().main()
