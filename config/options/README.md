# Options

This directory includes a default options file for the use of the algorithm.

## Validation

Options should be validated after creation and before use. This can be done with [cue](https://cuelang.org/): `cue vet options_basic.json options.cue`

## Export

To export a json file from a cue template use: `cue export option_file.cue -o option_file.json`
