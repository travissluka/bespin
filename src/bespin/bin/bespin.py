# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import click

from .bin import bin_
from .concat import concat
from .merge import merge


@click.group()
@click.version_option()
def cli():
    """Binned Experiment Statistics Package for INtegrated diagnostics.

    Joint Center for Satellite Data Assimilation (JCSDA) ©2021

    Tools for binning JEDI observation diagnostics and manipulating
    those resulting binned statistics files.
    """
    pass


cli.add_command(bin_)
cli.add_command(concat)
cli.add_command(merge)
