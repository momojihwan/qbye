import utils
import sys
import argparse



def get_parser():
    """Get demo model parser"""

    parser = argparse.ArgumentParser(description='Demo parser')
    parser.add_argument(
        '--config',
        type=str,
        default='',
        help="Model configuration")
    
    parser.add_argument(
        "--model",
        type=str,
        default="./checkpoints/CTC/",
        help="ASR model"
    )

    parser.add_argument(
        "--kws_model",
        type=str,
        default="./checkpoints/KWS/",
        help="Keyword spotting model"
    )

def main(cmd=None):
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)




if __name__ == "__main__":
    main()