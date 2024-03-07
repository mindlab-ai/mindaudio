"""train_criteo."""

import os

from dataset import create_dataset
from mindspore import ParameterTuple, Tensor, context
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.context import ParallelMode
from mindspore.nn.optim import Adam
from mindspore.train import Model
from mindspore.train.callback import (
    CheckpointConfig,
    LossMonitor,
    ModelCheckpoint,
    TimeMonitor,
)
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindaudio.loss.ctc_loss import NetWithCTCLoss
from mindaudio.models.deepspeech2 import DeepSpeechModel
from mindaudio.scheduler.scheduler_factory import step_lr
from mindaudio.utils.hparams import parse_args
from mindaudio.utils.train_one_step import TrainOneStepWithLossScaleCell


def train(args):
    ds_train = create_dataset(
        audio_conf=args.SpectConfig,
        manifest_filepath=args.TrainingConfig.train_manifest,
        labels=args.labels,
        normalize=True,
        train_mode=True,
        batch_size=args.TrainingConfig.batch_size,
        rank=rank_id,
        group_size=group_size,
    )

    steps_size = ds_train.get_dataset_size()
    lr = step_lr(
        lr_init=args.OptimConfig.learning_rate,
        total_epochs=args.TrainingConfig.epochs,
        steps_per_epoch=steps_size,
    )

    deepspeech_net = DeepSpeechModel(
        batch_size=args.TrainingConfig.batch_size,
        rnn_hidden_size=args.ModelConfig.hidden_size,
        nb_layers=args.ModelConfig.hidden_layers,
        labels=args.labels,
        rnn_type=args.ModelConfig.rnn_type,
        audio_conf=args.SpectConfig,
        bidirectional=True,
    )

    loss_net = NetWithCTCLoss(deepspeech_net, ascend=(args.device_target == "Ascend"))
    weights = ParameterTuple(deepspeech_net.trainable_params())

    optimizer = Adam(
        weights,
        learning_rate=args.OptimConfig.learning_rate,
        eps=args.OptimConfig.epsilon,
        loss_scale=args.OptimConfig.loss_scale,
    )
    train_net = TrainOneStepWithLossScaleCell(loss_net, optimizer, Tensor(1024))
    train_net.set_train(True)
    if args.Pretrained_model != "":
        param_dict = load_checkpoint(args.Pretrained_model)
        load_param_into_net(train_net, param_dict)
        print("Successfully loading the pre-trained model")

    model = Model(train_net)
    callback_list = [TimeMonitor(steps_size), LossMonitor()]

    if args.is_distributed:
        args.CheckpointConfig.ckpt_path = os.path.join(
            args.CheckpointConfig.ckpt_path, "ckpt_" + str(get_rank()) + "/"
        )

    config_ck = CheckpointConfig(
        save_checkpoint_steps=5,
        keep_checkpoint_max=args.CheckpointConfig.keep_checkpoint_max,
    )
    ckpt_cb = ModelCheckpoint(
        prefix=args.CheckpointConfig.ckpt_file_name_prefix,
        directory=args.CheckpointConfig.ckpt_path,
        config=config_ck,
    )

    callback_list.append(ckpt_cb)
    print(callback_list)
    model.train(
        args.TrainingConfig.epochs,
        ds_train,
        callbacks=callback_list,
        dataset_sink_mode=data_sink,
    )


if __name__ == "__main__":
    rank_id = 0
    group_size = 1
    args = parse_args()
    data_sink = args.device_target != "CPU"
    context.set_context(
        mode=args.mode, device_target=args.device_target, save_graphs=False
    )
    if args.device_target == "GPU":
        context.set_context(enable_graph_kernel=True)
    if args.is_distributed:
        init()
        rank_id = get_rank()
        group_size = get_group_size()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            device_num=get_group_size(),
            parallel_mode=ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
        )
    else:
        if args.device_target == "Ascend":
            device_id = int(args.device_id)
            context.set_context(device_id=device_id)
    train(args=args)
