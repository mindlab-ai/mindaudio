# Using DeepSpeech2 for Speech Recognition
> [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/abs/1512.02595)

## Introduction

DeepSpeech2 is a speech recognition model trained using CTC loss. It replaces the entire manually designed component pipeline with neural networks and can handle a variety of speech, including noisy environments, accents, and different languages. The currently provided version supports using the [DeepSpeech2](http://arxiv.org/pdf/1512.02595v1.pdf) model for training/testing and inference on the librispeech dataset on NPU.


## Requirements
| mindspore | ascend driver |  firmware   | cann toolkit/kernel |
|:---------:|:-------------:|:-----------:|:-------------------:|
|   2.3.1   |   24.1.RC2    | 7.3.0.1.231 |    8.0.RC2.beta1    |

## Model Architecture

The current reproduced model includes:

- Two convolutional layers:
  - Number of channels: 32, kernel size: 41, 11, stride: 2, 2
  - Number of channels: 32, kernel size: 41, 11, stride: 2, 1
- Five bidirectional LSTM layers (size 1024)
- A projection layer [size equal to the number of characters plus 1 (for the CTC blank symbol), 28]

## Data Processing

- Audio:
  1. Feature extraction: log power spectrum.
  2. Data augmentation: not used yet.

- Text:
  - Text encoding uses labels for English alphabet conversion; users can replace this with a tokenization model.

## Usage Steps

### 1. Preparing the Dataset
If the dataset is not downloaded, you can use the provided script to download and prepare the data with one command, as shown below:

```shell
# Download and create json
python mindaudio/data/librispeech.py --root_path "your_data_path"
```

If you have already downloaded the compressed files, operate with the following command:

```shell
# Create json
python mindaudio/data/librispeech.py --root_path "your_data_path" --data_ready True
```

LibriSpeech stores files in flac audio format. To use them in MindAudio, all flac files need to be converted to wav files. Users can use [ffmpeg](https://gist.github.com/seungwonpark/4f273739beef2691cd53b5c39629d830) or [sox](https://sourceforge.net/projects/sox/) for the conversion.

After processing, the dataset directory structure is as follows:

```
    ├─ LibriSpeech_dataset
    │  ├── train
    │  │   ├─libri_test_clean_manifest.json
    │  │   ├─ wav
    │  │   └─ txt
    │  ├── val
    │  │   ├─libri_test_clean_manifest.json
    │  │   ├─ wav
    │  │   └─ txt
    │  ├── test_clean
    │  │   ├─libri_test_clean_manifest.json
    │  │   ├─ wav
    │  │   └─ txt
    │  └── test_other
    │  │   ├─libri_test_clean_manifest.json
    │  │   ├─ wav
    │  │   └─ txt
```

The four **.json files store the absolute paths of the corresponding data. For subsequent model training and validation, update the xx_manifest in the yaml configuration file to the location of the corresponding libri_xx_manifest.json file.

### 2. Training
#### Single-Card Training
Due to the large dataset, this training method is not recommended.
```shell
# Standalone training
python train.py -c "./deepspeech2.yaml"
```
Note: The default is to use Ascend machines.

#### Multi-Card Training on Ascend
This example uses 8 NPUs. If you want to change the number of NPUs, you can modify the number of cards after -n in the command below.
```shell
# Distributed training
mpirun -n 8 python train.py -c "./deepspeech2.yaml"
```
Note: If the script is executed by the root user, you must add the --allow-run-as-root parameter in mpirun, as shown below:
```shell
mpirun --allow-run-as-root -n 8 python train.py -c "./deepspeech2.yaml"
```

### 3. Evaluating the Model
Update the path to the trained weights in the Pretrained_model section of the deepspeech2.yaml configuration file and execute the following command:
```shell
# Validate a trained model
python eval.py -c "./deepspeech2.yaml"
```

## Performance

Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode:

| model name  | cards | batch size | jit level | graph compile | ms/step |  cer  |  wer  | recipe                                                                                             |                                          weight                                          |
|:-----------:|:-----:|:----------:|:---------:|:-------------:|:-------:|:-----:|:-----:|:---------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------:|
| deepspeech2 |   8   |     64     |    O0     |     404s      |  9078   | 3.461 | 10.24 | [yaml](https://github.com/mindspore-lab/mindaudio/blob/main/examples/deepspeech2/deepspeech2.yaml) | [weights](https://download.mindspore.cn/toolkits/mindaudio/deepspeech2/deepspeech2.ckpt) |

- cer and wer tested in Librispeech `test clean` datasets.
