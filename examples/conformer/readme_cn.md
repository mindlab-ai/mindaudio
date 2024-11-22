# 使用conformer进行语音识别

> [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)

## 介绍

conformer是将一种transformer和cnn结合起来，对音频序列进行局部和全局依赖都进行建模的模型。目前基于transformer和卷积神经网络cnn的模型在ASR上已经达到了较好的效果，Transformer能够捕获长序列的依赖和基于内容的全局交互信息，CNN则能够有效利用局部特征，因此针对语音识别问题提出了卷积增强的transformer模型，称为conformer，模型性能优于transformer和cnn。目前提供版本支持在NPU和GPU上使用[conformer](https://arxiv.org/pdf/2102.06657v1.pdf)模型在aishell-1数据集上进行训练/测试和推理。

## 版本要求
| mindspore | ascend driver |  firmware   | cann toolkit/kernel |
|:---------:|:-------------:|:-----------:|:-------------------:|
|   2.3.1   |   24.1.RC2    | 7.3.0.1.231 |    8.0.RC2.beta1    |

## 模型结构

Conformer整体结构包括：SpecAug、ConvolutionSubsampling、Linear、Dropout、ConformerBlocks×N，可见如下结构图。

- ConformerBlock结构（N个该结构）：Feed Forward Module、Multi-Head Self Attention Module、Convolution Module、Feed Forward Module、Layernorm。其中每个Module都是前接一个Layernorm后接一个Dropout，且都有残差链连接，残差数据为输入数据本身。

- 马卡龙结构：可以看到ConformerBlock神似马卡龙结构，即两个一样的Feed Forward Module中间夹了Multi-Head Self Attention Module和Convolution。

  ![image-20230310165349460](https://raw.githubusercontent.com/mindspore-lab/mindaudio/main/tests/result/conformer.png)



## 使用步骤

### 1. 数据集准备

以aishell数据集为例，mindaudio提供下载、生成统计信息的脚本（包含wav文件地址信息以及对应中文信息），执行此脚本会生成train.csv、dev.csv、test.csv三个文件。

```shell
# data_path为存放数据的地址
python mindaudio/data/aishell.py --data_path "/data" --download False
```

如需下载数据， --download True

### 2. 数据预处理

#### 文字部分

根据aishell提供的aishell_transcript_v0.8.txt，生成逐字的编码文件，每个字对应一个id，输出包含编码信息的文件：lang_char.txt。

```shell
cd mindaudio/utils
python text2token.py -s 1 -n 1 "data_path/data_aishell/transcript/aishell_transcript_v0.8.txt" | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${/data_path/lang_char.txt}
```

#### 音频部分

本模型使用了全局cmvn，为提高模型训练效率，在训练前会对数据的特征进行统计，生成包含统计信息的文件：global_cmvn.json。

```shell
cd examples/conformer
python compute_cmvn_stats.py --num_workers 16 --train_config conformer.yaml --in_scp data_path/train.csv --out_cmvn data_path/global_cmvn
```

注意：--num_workers可根据训练设备的核数进行调整

### 3. 开始训练（默认使用Ascend 910）

#### 单卡
```shell
cd examples/conformer
# Standalone training
python train.py --config_path ./conformer.yaml
```

注意:

#### Ascend上进行多卡训练

需配置is_distributed参数为True

```shell
# Distribute_training
mpirun -n 8 python train.py --config_path ./conformer.yaml  --is_distributed True
```

如果脚本是由root用户执行的，必须在mpirun中添加——allow-run-as-root参数，如下所示:

```shell
mpirun --allow-run-as-root -n 8 python train.py --config_path ./conformer.yaml
```


启动训练前，可更改环境变量设置，更改线程数以提高运行速度。如下所示:

```shell
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```



### 4.评估

我们提供ctc greedy search、ctc prefix beam search、attention decoder、attention rescoring四种解码方式，可在yaml配置文件中对解码方式进行修改。

执行脚本后将生成包含预测结果的文件为result.txt

```shell
# by default using ctc greedy search decoder
python predict.py --config_path ./conformer.yaml

# using ctc prefix beam search decoder
python predict.py --config_path ./conformer.yaml --decode_mode ctc_prefix_beam_search

# using attention rescoring decoder
python predict.py --config_path ./conformer.yaml --decode_mode attention_rescoring
```

## 模型表现

在 ascend 910* mindspore2.3.1图模式上的测试性能:

| model name |    decoding method     | cards | batch size | jit level | graph compile | ms/step | cer  |                                                  recipe                                                  |                                                       weight                                                       |
|:----------:|:----------------------:|:-----:|:----------:|:---------:|:-------------:|:-------:|:----:|:--------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------:|
| conformer  |   ctc greedy search    |   8   |   bucket   |    O0     |     103s      |  727.5  | 5.62 | [conformer.yaml](https://github.com/mindspore-lab/mindaudio/blob/main/examples/conformer/conformer.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindaudio/conformer/conformer_avg_30-692d57b3-910v2.ckpt) |
| conformer  | ctc prefix beam search |   8   |   bucket   |    O0     |     103s      |  727.5  | 5.62 | [conformer.yaml](https://github.com/mindspore-lab/mindaudio/blob/main/examples/conformer/conformer.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindaudio/conformer/conformer_avg_30-692d57b3-910v2.ckpt) |
| conformer  |  attention rescoring   |   8   |   bucket   |    O0     |     103s      |  727.5  | 5.12 | [conformer.yaml](https://github.com/mindspore-lab/mindaudio/blob/main/examples/conformer/conformer.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindaudio/conformer/conformer_avg_30-692d57b3-910v2.ckpt) |
