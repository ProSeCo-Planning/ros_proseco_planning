import sys
import argparse
from typing import Optional, Tuple, List, Any
from proseco.evaluator.head import Head


def parse_arguments() -> argparse.Namespace:
    """Load command line flags for the evaluation.

    Takes the -c argument and the -y flag.
    Returns both as parsed arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """

    parser = argparse.ArgumentParser()
    description_c = "bulk configuration file for the evaluation"
    description_y = "starts the evaluation without questions"
    description_n = "flag for starting the ray dashboard GUI"
    description_d = "specifies whether the MCTS output is printed to the console"
    description_s = (
        "specifies whether to create a results.json file in the root evaluation folder"
    )
    parser.add_argument("-c", "--config", help=description_c, type=str)
    parser.add_argument("-y", "--yes", help=description_y, action="store_true")
    parser.add_argument(
        "-sn", "--no_summary", help=description_s, default=False, action="store_true"
    )
    parser.add_argument(
        "-dn", "--no_dashboard", help=description_n, default=False, action="store_true"
    )
    parser.add_argument(
        "-d", "--debug", help=description_d, default=False, action="store_true"
    )
    parser.add_argument("--address", help="Override the ray cluster address")
    args = parser.parse_args()
    return args


def get_user_confirm(head: Head, arg: argparse.Namespace) -> None:
    """Ask for confirmation before commencing the evaluation.

    Informs the user on the amount of executions the ProSeCo Planner will undergo and gives the opportunity to abort.

    Parameters
    ----------
    head : Head
        Head node for the evaluation.
    arg : argparse.Namespace
        Parsed command line arguments for the evaluator.
    """

    n = head.number_evaluations
    head.logger.info(
        f"The specified configuration file is expected to generate {n} evaluations."
    )
    if not arg:
        i = input("Do you wish to continue? [Y,n]\n")
        i = i.lower()
        if (i == "") or ("y" in i):
            pass
        elif "n" in i:
            print("Aborting ...")
            sys.exit(0)
        else:
            print("Input not recognized, aborting ...")
            sys.exit(0)
    else:
        pass


def evaluate(
    args: argparse.Namespace = None,
    head: Optional[Head] = None,
    outside_eval: bool = False,
) -> Tuple[List[List[Any]], bool]:
    """Evaluation function

    Main method, can be called directly from evaluator.py script or from outside
    by being provided a head instance and an args instance. Returns a list
    with the following structure (i, j being index of result number and file
    number inside the result):
    [[<self.ip>_i, [<filename>_j, <json-dic>], ..., ['uuid.json', {<uuid>}], ... ]

    Parameters
    ----------
    args : argparse.Namespace = None
        Parsed command line arguments for the evaluator.
    head : Optional[Head] = None
        Evaluator head node instance.
    outside_eval : bool = False
        Needs to be set to True if the evaluation is not run from this script.

    Returns
    -------
    results
          Returns the evaluation results in a list.
    """

    if not args:
        args = parse_arguments()
    if not head:
        head = Head(args)
    assert head is not None and args is not None, "Head not properly set"
    if outside_eval:
        head.bulk_config = head.get_file(file_type="evaluator", file=args.config)
    else:
        get_user_confirm(head, args.yes)
    results = head.start()

    return results


if __name__ == "__main__":
    results = evaluate()
