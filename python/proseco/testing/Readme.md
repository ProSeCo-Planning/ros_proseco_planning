## Description

This folder contains a python scripts and config files to run tests for the ProSeCo planning library.

## How to use

There are two config files which define the way tests for ProSeCo are run

1. The config `ros_proseco_planning/config/evaluator/ci_config.json` specifies which scenarios are run
   as well as the compute options and ray settings for the evaluation.
2. The file `test_thresholds.json` defines a hard and a soft threshold for each scenario.
   - `soft`: sets a soft threshold for the average success rate.
     Failing this threshold for a scenario will not fail the job but print a warning to the CI console.
   - `hard`: sets the hard threshold. A scenario with a lower success rate will result in a failed pipeline.

You can run the tests manually with the commands

```bash
python testing/tester.py
python testing/test_random_start_pos.py
```
