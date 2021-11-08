# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import click

from bespin import BinnedStatistics, Dimension, Diagnostic, Filter
import bespin.io.obs_ioda as obsio


@click.command(name='bin')
@click.argument('obs_files',
                type=click.Path(exists=True, dir_okay=False),
                nargs=-1,
                required=True,)
@click.option('-o', '--output',
              type=click.Path(writable=True, dir_okay=False),
              required=True,
              help="Output filename.",)
@click.option('-c', '--config',
              type=click.Path(exists=True, dir_okay=False),
              help=(
                  "(NOT YET IMPLEMENTED)"
                  " An optional path to a yaml configuration file used for"
                  " generating the filter, variables, diagnostics, and bins"
                  " parameters."),)
@click.option('-f', '--filter', 'filters',
              metavar="FILTER",
              multiple=True,
              help=(
                "A string representation of a pre-binning filter. See the"
                " online documentation for details on the available filters."
                " This option can be repeated for as many filters as required."
                " Filters are run in the order given."))
@click.option('-v', '--var', 'variables',
              metavar="VAR",
              multiple=True,
              help=(
                "An observation variable on which to run the binning. This"
                " option can be repeated for as many variables as required."
                "  [Default: <every member of ObsValue/ group>]"))
@click.option('-d', '--diag', 'diagnostics',
              metavar="DIAGS",
              required=True,
              multiple=True,
              help=(
                "A diagnostic/statistic specification. See the online"
                " documentation for details on the format for this option."
                " This option can be repeated for as many diagnostics as"
                " required."))
@click.option('-b', '--bin', 'bins',
              metavar="BINS",
              multiple=True,
              help=(
                "A dimension to use for the binning. See the online"
                " documentation for details on the format for this option."
                " This option can be repeated for as many dimensions as"
                " required. If no binning dimensions are given, a global"
                " binning is performed."
              ))
@click.option('-O', '--overwrite',
              is_flag=True,
              help=("Overwrite an existing output file. By default the"
                    " program will exit if the file already exists."))
def bin_(obs_files, config, filters, variables, diagnostics, bins,
         output, overwrite):
    """Perform binning of one or more obs diagnostic files.

    One or more OBS_FILES observation diagnostic files will be binned.
    Options for the binning and filtering can be specified either by a
    combination of available options, or by passing a yaml file configuration
    with the --config option.
    """
    # make sure ioda is present
    if not obsio.ioda_found:
        raise ModuleNotFoundError(
            "Cannot bin obs statistics if IODA library is not present.")

    # check validity of input file names
    if len(obs_files) > len(set(obs_files)):
        raise ValueError("Duplicate input filenames were given.")

    # parse input options
    if config is not None:
        raise NotImplementedError('"config" option not yet implemented.')
    bins = [Dimension.from_str(b) for b in bins]
    diagnostics = [Diagnostic.from_str(d) for d in diagnostics]
    filters = [Filter.from_str(f) for f in filters]
    # TODO Make sure diagnostics are valid
    # 1) not using derived
    # 2) in right order

    # open each file, bin it, and merge all the results
    data: BinnedStatistics = None
    for obs_file in obs_files:
        print(f'processing {obs_file}')
        unbinned_data = obsio.read(obs_file)

        # if no variables were listed, by default use everything within one
        # of the obs groups. The groups listed in "search_groups" are scanned
        # in order until a set of variables is found.
        # TODO verify the group names once we have clean IODAv2 files
        search_groups = ('hofx0', 'hofx', 'ObsValue')
        if not len(variables):
            for group in search_groups:
                variables = [
                    v.split('/')[1] for v in unbinned_data.variables
                    if v.startswith(f'{group}/')]
                if len(variables):
                    break
            print(f'Automatically using variables: {variables}')

        # perform the binning
        d = BinnedStatistics.bin(
            name='bin_name',  # TODO, grab a user supplied name, or invent one
            bins=bins,
            diagnostics=diagnostics,
            variables=variables,
            unbinned_data=unbinned_data,
            filters=filters,
        )

        # merge this binned data with previous files, if applicable.
        data = d if data is None else data.merge(d)

    # write out the results
    data.write(output, overwrite=overwrite)
