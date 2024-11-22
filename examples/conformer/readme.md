# Using Conformer for Speech Recognition

> [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)

## Introduction

Conformer is a model that combines transformers and CNNs to model both local and global dependencies in audio sequences. Currently, models based on transformers and convolutional neural networks (CNNs) have achieved good results in automatic speech recognition (ASR). Transformers can capture long-sequence dependencies and global interactions based on content, while CNNs can effectively utilize local features. Therefore, a convolution-enhanced transformer model called Conformer has been proposed for speech recognition, showing performance superior to both transformers and CNNs. The current version supports using the Conformer model for training/testing and inference on the AISHELL-1 dataset on ascend NPU and GPU.

## Requirements
| mindspore | ascend driver |  firmware   | cann toolkit/kernel |
|:---------:|:-------------:|:-----------:|:-------------------:|
|   2.3.1   |   24.1.RC2    | 7.3.0.1.231 |    8.0.RC2.beta1    |

### Model Structure

The overall structure of Conformer includes SpecAug, ConvolutionSubsampling, Linear, Dropout, and ConformerBlocks×N, as shown in the structure diagram below.

- ConformerBlock Structure (N of this structure): Feed Forward Module, Multi-Head Self Attention Module, Convolution Module, Feed Forward Module, Layernorm. Each module is preceded by a Layernorm and followed by a Dropout, with residual connections linking the input data directly.

- Macaron Structure: The ConformerBlock resembles a macaron structure, with a Multi-Head Self Attention Module and Convolution Module sandwiched between two identical Feed Forward Modules.

  ![image-20230310165349460](https://raw.githubusercontent.com/mindspore-lab/mindaudio/main/tests/result/conformer.png)




## Usage Steps

### 1. Dataset Preparation

Take the AISHELL dataset as an example. MindAudio provides scripts to download and generate statistical information (including the addresses of wav files and corresponding Chinese information). Executing this script will generate three files: train.csv, dev.csv, and test.csv.

```shell
# data_path is the path where the data is stored
python mindaudio/data/aishell.py --data_path "/data" --download False
```

To download data, set --download parameter to be True.

### 2. Data Preprocessing

#### Text Part

Based on the aishell_transcript_v0.8.txt provided by AISHELL, generate a character-by-character encoding file where each character corresponds to an ID, outputting a file containing encoding information: lang_char.txt.

```shell
cd mindaudio/utils
python text2token.py -s 1 -n 1 "data_path/data_aishell/transcript/aishell_transcript_v0.8.txt" | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${/data_path/lang_char.txt}
```

#### Audio Part

This model uses global CMVN. To improve training efficiency, statistical features of the data are computed before training, generating a file with the statistical information: global_cmvn.json.

```shell
cd examples/conformer
python compute_cmvn_stats.py --num_workers 16 --train_config conformer.yaml --in_scp data_path/train.csv --out_cmvn data_path/global_cmvn
```

Note: --num_workers can be adjusted according to the number of cores on the training device.

### 3. Training

#### Single-Card Training (by default using Ascend 910)
```shell
cd examples/conformer
# Standalone training
python train.py --config_path ./conformer.yaml
```

Note: Use Ascend device by default.

#### Multi-Card Training on Ascend

This example uses 8 ascend NPUs.
```shell
# Distribute training
mpirun -n 8 python train.py --config_path ./conformer.yaml
```
Note:
When using multi-card training, ensure that is_distributed in the YAML file is set to True. This can be configured by modifying the YAML file or adding parameters on the command line.

```shell
# Distribute_training
mpirun -n 8 python train.py --config_path ./conformer.yaml  --is_distributed True
```
If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.


Before starting training, you can set environment variable to adjust the number of threads for faster execution as shown below:

```shell
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```



### 4. Evaluation

Four decoding methods are provided: CTC greedy search, CTC prefix beam search, attention decoder, and attention rescoring. The decoding method can be modified in the YAML configuration file.

Executing the script will generate a file containing the prediction results: result.txt.
```shell
# by default using ctc greedy search decoder
python predict.py --config_path ./conformer.yaml

# using ctc prefix beam search decoder
python predict.py --config_path ./conformer.yaml --decode_mode ctc_prefix_beam_search

# using attention rescoring decoder
python predict.py --config_path ./conformer.yaml --decode_mode attention_rescoring
```


## Performance

Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode:

| model name |    decoding method     | cards | batch size | jit level | graph compile | ms/step | cer  |                                                  recipe                                                  |                                                       weight                                                       |
|:----------:|:----------------------:|:-----:|:----------:|:---------:|:-------------:|:-------:|:----:|:--------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------:|
| conformer  |   ctc greedy search    |   8   |   bucket   |    O0     |     103s      |  727.5  | 5.62 | [conformer.yaml](https://github.com/mindspore-lab/mindaudio/blob/main/examples/conformer/conformer.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindaudio/conformer/conformer_avg_30-692d57b3-910v2.ckpt) |
| conformer  | ctc prefix beam search |   8   |   bucket   |    O0     |     103s      |  727.5  | 5.62 | [conformer.yaml](https://github.com/mindspore-lab/mindaudio/blob/main/examples/conformer/conformer.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindaudio/conformer/conformer_avg_30-692d57b3-910v2.ckpt) |
| conformer  |  attention rescoring   |   8   |   bucket   |    O0     |     103s      |  727.5  | 5.12 | [conformer.yaml](https://github.com/mindspore-lab/mindaudio/blob/main/examples/conformer/conformer.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindaudio/conformer/conformer_avg_30-692d57b3-910v2.ckpt) |
