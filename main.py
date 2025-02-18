import os
import time
import torch
import pytorch_lightning as pl
from model.layers.main_layer import LightningModule
from model.args import parse_args
from data_pre.tokenizer import MolTranBertTokenizer
from data_pre.data_loader import PropertyPredictionDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only, seed

def main():
    margs = parse_args()
    print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    pos_emb_type = 'rot'
    print('pos_emb_type is {}'.format(pos_emb_type))

    run_name_fields = [
        margs.dataset_name,
        margs.measure_name,
        pos_emb_type,
        margs.fold,
        margs.mode,
        "lr",
        margs.lr_start,
        "batch",
        margs.batch_size,
        "drop",
        margs.dropout,
        margs.dims,
    ]

    run_name = "_".join(map(str, run_name_fields))

    print(run_name)
    datamodule = PropertyPredictionDataModule(margs)

    margs.dataset_names = "valid test".split()

    checkpoints_folder = margs.checkpoints_folder
    checkpoint_root = os.path.join(checkpoints_folder, margs.measure_name)
    margs.checkpoint_root = checkpoint_root
    os.makedirs(checkpoints_folder, exist_ok=True)

    checkpoint_dir = os.path.join(checkpoint_root, "models")
    results_dir = os.path.join(checkpoint_root, "results")

    margs.results_dir = results_dir
    margs.checkpoint_dir = checkpoint_dir
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoints_folder, margs.measure_name)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(period=1, save_last=True, dirpath=checkpoint_dir, filename='checkpoint', verbose=True)
    
    # 定义 ModelCheckpoint 回调
    r2_checkpoint_callback =  pl.callbacks.ModelCheckpoint(
        monitor="CCS_test_r2",
        mode="max",
        save_top_k=5,
        filename="best-model-{epoch:02d}-{CCS_test_r2:.4f}"
    )

    print(margs)
    # 输出logger 设置
    logger = TensorBoardLogger(
        save_dir=checkpoint_root,
        version=run_name,
        name="lightning_logs",
        default_hp_metric=True,
    )
    # tokenlizer 设置
    tokenizer = MolTranBertTokenizer('bert_vocab.txt')
    seed.seed_everything(margs.seed)

    # 检查点 args参数设置：
    if margs.seed_path == '':
        print("# training from scratch")
        model = LightningModule(margs, tokenizer)
    else:
        print("# loaded pre-trained model from {args.seed_path}")
        model = LightningModule(margs, tokenizer).load_from_checkpoint(margs.seed_path, strict=False, config=margs, tokenizer=tokenizer, vocab=len(tokenizer.vocab))

    # 没有参数则读取对应的 上次训练的结果
    last_checkpoint_file = os.path.join(checkpoint_dir, "last.ckpt")
    resume_from_checkpoint = None
    if os.path.isfile(last_checkpoint_file):
        print(f"resuming training from : {last_checkpoint_file}")
        resume_from_checkpoint = last_checkpoint_file
    else:
        print(f"training from scratch")

    # 模型训练器
    trainer = pl.Trainer(
        max_epochs=margs.max_epochs,
        default_root_dir=checkpoint_root,
        gpus=1,
        logger=logger,
        resume_from_checkpoint=resume_from_checkpoint,
        checkpoint_callback=checkpoint_callback,
        callbacks = [r2_checkpoint_callback],
        num_sanity_val_steps=0,
    )

    tic = time.perf_counter()
    trainer.fit(model, datamodule)
    toc = time.perf_counter()
    print('Time was {}'.format(toc - tic))


if __name__ == '__main__':
    main()