import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from model.summarization import SummaryTrainer
from model import MInterface
from model.ScanCoster import ScanCostTrainer


def load_callbacks():
    callbacks = []
    # callbacks.append(plc.EarlyStopping(
    #     monitor='val_acc',
    #     mode='max',
    #     patience=10,
    #     min_delta=0.001
    # ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_loss',
        filename='best-{epoch:02d}-{val_acc:.3f}',
        save_top_k=1,
        mode='min',
        save_last=True
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks

def main(args):
    pl.seed_everything(args.seed)
    # data_module = DInterface(**vars(args))

    if args.load_path is None:
        # model = MInterface(**vars(args))
        # model = SummaryTrainer(**vars(args))
        model = ScanCostTrainer(**vars(args))
    else:
        # model = MInterface(**vars(args))
        # model = SummaryTrainer(**vars(args))
        model = ScanCostTrainer(**vars(args))
        # args.ckpt_path = args.load_path

    # # If you want to change the logger's saving folder
    # logger = WandbLogger(save_dir=args.log_dir, project="debug")
    # args.logger = logger
    args.callbacks = load_callbacks()

    # trainer = Trainer.from_argparse_args(args, accelerator='gpu', gpus=1, log_every_n_steps=1)
    trainer = Trainer.from_argparse_args(args, accelerator='gpu', gpus=1, fast_dev_run=True)
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # Restart Control
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Training Info
    parser.add_argument('--block_size', type=int, default='20', help='Block Size of a FS block.')
    parser.add_argument('--dataset', type=str, default='dmv-tiny', help='Dataset.')
    parser.add_argument('--data_dir', default='ref/data', type=str)
    parser.add_argument('--model_name', default='transformer', type=str)
    parser.add_argument('--loss', default='mse', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='lightning_logs', type=str)
    
    # Model Hyperparameters
    parser.add_argument('--hid', default=64, type=int)
    parser.add_argument('--block_num', default=8, type=int)
    parser.add_argument('--in_channel', default=3, type=int)
    parser.add_argument('--layer_num', default=5, type=int)
    # Transformer.
    parser.add_argument(
        '--heads',
        type=int,
        default=2,
        help='Transformer: num heads.  A non-zero value turns on Transformer'\
        ' (otherwise MADE/ResMADE).'
    )
    parser.add_argument('--blocks',
                        type=int,
                        default=1,
                        help='Transformer: num blocks.')
    parser.add_argument('--dmodel',
                        type=int,
                        default=2,
                        help='Transformer: d_model.')
    parser.add_argument('--dff', type=int, default=64, help='Transformer: d_ff.')
    parser.add_argument('--transformer-act',
                        type=str,
                        default='gelu',
                        help='Transformer activation.')
    parser.add_argument(
                        '--column-masking',
                        action='store_true',
                        help='Column masking training, which permits wildcard skipping'\
                        ' at querying time.')

    # Other
    parser.add_argument('--aug_prob', default=0.5, type=float)

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)

    ## Deprecated, old version
    # parser = Trainer.add_argparse_args(
    #     parser.add_argument_group(title="pl.Trainer args"))

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=1000)

    args = parser.parse_args() # type: ignore


    main(args)