# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import click

import bespin


@click.command()
@click.argument('input_files', nargs=-1, required=True)
@click.option('-o', '--output', metavar='FILE', type=str, required=True, help=(
    "Output filename."))
@click.option('-d', '--dimension', type=str, default='time')
def concat(input_files, output, dimension):
    """Concatenate multiple binned statistics files.

    Two or more INPUT_FILES wil be concatenated.
    A single dimension of choice will either be created or appended.
    (Compare this to the "merge" command which will keep the resulting
    dimensions the same).

    !!!!! NOT YET IMPLEMENTED !!!!!
    """
    raise NotImplementedError()
