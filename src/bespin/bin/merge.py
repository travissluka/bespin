# (C) Copyright 2021-2021 UCAR
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import click

from bespin import BinnedStatistics


@click.command()
@click.argument('input_files', nargs=-1, required=True)
@click.option('-o', '--output', metavar='FILE', type=str, required=True, help=(
    "Output filename."))
@click.option('-O', '--overwrite', is_flag=True, help=(
    "Overwrite an existing output file. By default the program will exit"
    " if the file already exists."))
def merge(input_files, output, overwrite):
    """Merge multiple binned statistics files.

    Two or more INPUT_FILES will be merged.
    The number of resulting dimensions will be the same and will effectively be
    the same as if all input obs files were binned at the same time. (Compare
    this to "bespin concat" which will create or append a single dimension).
    """
    # check validity of input file names
    if len(input_files) < 2:
        raise ValueError("Two or more files must be given to merge.")

    if len(input_files) > len(set(input_files)):
        raise ValueError("Duplicate input filenames were given.")

    # Open and merge each file one at a time
    # TODO get crazy and have threaded merging?
    itr = iter(input_files)
    merged = BinnedStatistics.read(next(itr))
    for file in itr:
        merged = merged.merge(BinnedStatistics.read(file))

    # write output file
    try:
        merged.write(output, overwrite=overwrite)
    except FileExistsError as e:
        raise FileExistsError(
            'Output file exists. Remove file or run with the "-O" option'
        ) from e
