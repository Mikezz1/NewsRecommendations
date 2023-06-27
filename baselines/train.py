from recbole.quick_start import run_recbole
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/pop.yaml",
        type=str,
        help="config",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run_recbole(
        config_file_list=[f"configs/{args.config}.yaml"],
    )
    # conda activate recenvc
    # /Users/mikhailoleynik/miniforge3/envs/recenvc/bin/python3 train.py
