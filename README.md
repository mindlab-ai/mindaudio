<div align="center">


# MindAudio

[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/mindspore-lab/mindaudio/ut_test.yaml)
![GitHub issues](https://img.shields.io/github/issues/mindspore-lab/mindaudio)
![GitHub pull requests](https://img.shields.io/github/issues-pr/mindspore-lab/mindaudio)
![GitHub](https://img.shields.io/github/license/mindspore-lab/mindaudio)](<img alt="GitHub" src="https://img.shields.io/github/license/mindspore-lab/mindaudio">)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

[Introduction](#introduction) |
[Installation](#installation) |
[Get Started](#get-started-with-audio-data-analysis)

English | [中文](README_CN.md)

</div>

## Introduction

MindAudio is a toolbox of audio models and algorithms based on [MindSpore](https://www.mindspore.cn/). It provides a series of API for common audio data processing,data enhancement,feature extraction, so that users can preprocess data conveniently. Also provides examples to show how to build audio deep learning models with mindaudio.

The following is the corresponding `mindaudio` versions and supported `mindspore` versions.

|  mindaudio  |   mindspore   |
|:-----------:|:-------------:|
|    main     |    master     |
|     0.4     |  2.3.0/2.3.1  |
|     0.3     |    2.2.10     |


### data processing

```python
# read audio
>>> import mindaudio.data.io as io
>>> audio_data, sr = io.read(data_file)
# feature extraction
>>> import mindaudio.data.features as features
>>> feats = features.fbanks(audio_data)
```

## Installation

### Install with PyPI

The released version of MindAudio can be installed via `PyPI` as follows:

```shell
pip install mindaudio
```

### Install from Source

The latest version of MindAudio can be installed as follows:

```shell
git clone https://github.com/mindspore-lab/mindaudio.git
cd mindaudio
pip install -r requirements/requirements.txt
python setup.py install
```

## Get started with audio data analysis

###

MindAudio provides a series of commonly used audio data processing apis, which can be easily invoked for data analysis and feature extraction.

```python
>>> import mindaudio.data.io as io
>>> import mindaudio.data.spectrum as spectrum
>>> import numpy as np
>>> import matplotlib.pyplot as plt
# read audio
>>> audio_data, sr = io.read("./tests/samples/ASR/BAC009S0002W0122.wav")
# feature extraction
>>> n_fft = 512
>>> matrix = spectrum.stft(audio_data, n_fft=n_fft)
>>> magnitude, _ = spectrum.magphase(matrix, 1)
# display
>>> x = [i for i in range(0, 256*750, 256)]
>>> f = [i/n_fft * sr for i in range(0, int(n_fft/2+1))]
>>> plt.pcolormesh(x,f,magnitude, shading='gouraud', vmin=0, vmax=np.percentile(magnitude, 98))
>>> plt.title('STFT Magnitude')
>>> plt.ylabel('Frequency [Hz]')
>>> plt.xlabel('Time [sec]')
>>> plt.show()
```

Result presentation:

![image-20230310165349460](https://raw.githubusercontent.com/mindspore-lab/mindaudio/main/tests/result/stft_magnitude.png)


## Contributing

We appreciate all contributions to improve MindSpore Audio. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.

## License

This project is released under the [Apache License 2.0](LICENSE).

## Citation

If you find this project useful in your research, please consider citing:

```latex
@misc{MindSpore Audio 2022,
    title={{MindSpore Audio}:MindSpore Audio Toolbox and Benchmark},
    author={MindSpore Audio Contributors},
    howpublished = {\url{https://github.com/mindspore-lab/mindaudio}},
    year={2022}
}
```
