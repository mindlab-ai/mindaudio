<div align="center">


# MindAudio

[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/mindspore-lab/mindaudio/ut_test.yaml)
![GitHub issues](https://img.shields.io/github/issues/mindspore-lab/mindaudio)
![GitHub pull requests](https://img.shields.io/github/issues-pr/mindspore-lab/mindaudio)
![GitHub](https://img.shields.io/github/license/mindspore-lab/mindaudio)](<img alt="GitHub" src="https://img.shields.io/github/license/mindspore-lab/mindaudio">)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

[介绍](#介绍) |
[安装](#安装) |
[快速入门](#快速入门音频数据分析)

[English](README.md) | 中文
</div>

## 介绍
MindAudio 是基于 [MindSpore](https://www.mindspore.cn/) 的音频模型和算法工具箱。它提供了一系列用于常见音频数据处理、数据增强、特征提取的 API，方便用户对数据进行预处理。此外，它还提供了一些示例，展示如何利用 mindaudio 建立音频深度学习模型。

下表显示了相应的 `mindaudio` 版本和支持的 `mindspore` 版本。

|  mindaudio  |   mindspore   |
|:-----------:|:-------------:|
|    main     |    master     |
|     0.4     |  2.3.0/2.3.1  |
|     0.3     |    2.2.10     |

### 数据处理


```python
# read audio
>>> import mindaudio.data.io as io
>>> audio_data, sr = io.read(data_file)
# feature extraction
>>> import mindaudio.data.features as features
>>> feats = features.fbanks(audio_data)
```

## 安装

### Pypi安装

MindAudio的发布版本可以通过`PyPI`安装:

```shell
pip install mindaudio
```

### 源码安装
最新版本的 MindAudio 可以通过如下方式安装：

```shell
git clone https://github.com/mindspore-lab/mindaudio.git
cd mindaudio
pip install -r requirements/requirements.txt
python setup.py install
```

## 快速入门音频数据分析

###

MindAudio 提供了一系列常用的音频数据处理 APIs，可以轻松调用这些 APIs 进行数据分析和特征提取。

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

结果如图:

![image-20230310165349460](https://raw.githubusercontent.com/mindspore-lab/mindaudio/main/tests/result/stft_magnitude.png)


## 贡献方式
我们感谢开发者用户的所有贡献，一起让 MindAudio 变得更好。
贡献指南请参考[CONTRIBUTING.md](CONTRIBUTING.md) 。

## 许可证

MindAudio 遵循[Apache License 2.0](LICENSE)开源协议.

## 引用

如果你觉得 MindAudio 对你的项目有帮助，请考虑引用：

```latex
@misc{MindSpore Audio 2022,
    title={{MindSpore Audio}:MindSpore Audio Toolbox and Benchmark},
    author={MindSpore Audio Contributors},
    howpublished = {\url{https://github.com/mindspore-lab/mindaudio}},
    year={2022}
}
```
