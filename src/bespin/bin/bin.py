# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import click

import bespin


@click.command(name='bin')
@click.argument('obs_files', nargs=-1, required=True)
@click.option('-o', '--output', metavar="FILE", type=str, required=True, help=(
    "Output filename."))
def bin_(obs_files, output):
    """Perform binning of one or more obs diagnostic files.

    One or more OBS_FILES observation diagnostic files will be binned.

    !!!!! NOT YET IMPLEMENTED !!!!!
    """

    raise NotImplementedError()
