import argparse
from models.CTC.inference import inference_ctc
'''
python train.py \ 
    --asr_confir ../espnet/egs/librispeech/asr1/conf/tuning/CTC/asr_conformer_ctc.yaml \
    --asr_model ./checkpoints/CTC/conformer.pth \
    --audio_path 
'''


def get_parser():
    parser = argparse.ArgumentParser(description='CTC based KWS parser')

    parser.add_argument("--asr_config", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    # parser.add_argument("--anchor_path", type=str, default=None)

    return parser

def main(cmd=None):
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference_ctc(**kwargs)


if __name__ == "__main__":
    main()