import argparse

def _setup_args():
    p = argparse.ArgumentParser(
        description='trains a model with quantization aware training'
    )
    p.add_argument(
        '-m',
        '--model-name',
        help='name of model. Must be defined in models.py',
        metavar='name'
    )
    p.add_argument(
        '-l',
        '--list-models',
        help='lists available models',
        action='store_true'
    )
    p.add_argument(
        '--epochs',
        help='number of epochs for training. Default is 1',
        type=int,
        default=1,
        metavar='epochs'
    )
    p.add_argument(
        '--checkpoint-dir',
        help=('directory to save checkpoint information. '
              'Default is "./chkpt/checkpoints"'),
        default='./chkpt/checkpoints',
        type=str,
        metavar='dir'
    )
    p.add_argument(
        '--freeze',
        help='freezes the model',
        metavar='model name'
    )

    return p.parse_args(), p.print_help


def parse_args():

    cmd_args, print_help = _setup_args()

    from models import models

    if cmd_args.list_models:
        model_names = '\n'.join(m for m in models.keys())
        print(f'Models\n------------\n{model_names}')
        exit(0)

    if cmd_args.model_name is None:
        print_help()
        exit(0)

    if cmd_args.model_name not in models:
        print(f'unknown model name: {cmd_args.model_name}')
        print_help()
        exit(1)

    args = {
        'model_fn' : models[cmd_args.model_name],
        'epochs' : cmd_args.epochs,
        'checkpoint_dir' : cmd_args.checkpoint_dir,
        'frozen_filename': cmd_args.freeze
    }

    return args
