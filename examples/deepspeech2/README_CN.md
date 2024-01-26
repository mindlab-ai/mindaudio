# 使用DeepSpeech2进行语音识别



## 介绍

DeepSpeech2是一种采用CTC损失训练的语音识别模型。它用神经网络取代了整个手工设计组件的管道，可以处理各种各样的语音，包括嘈杂的环境、口音和不同的语言。目前提供版本支持在NPU和GPU上使用[DeepSpeech2](http://arxiv.org/pdf/1512.02595v1.pdf)模型在librispeech数据集上进行训练/测试和推理。

### 模型结构

目前的复现的模型包括:

- 两个卷积层:
  - 通道数为 32，内核大小为  41, 11 ，步长为  2, 2
  - 通道数为 32，内核大小为  41, 11 ，步长为  2, 1
- 五个双向 LSTM 层（大小为 1024）
- 一个投影层【大小为字符数加 1（为CTC空白符号)，28】

### 数据处理

- 音频：

  1.特征提取：采用log功率谱。

  2.数据增强：暂未使用。

- 文字：

​		文字编码使用labels进行英文字母转换，用户可使用分词模型进行替换。

## 使用步骤

### 1. 数据集准备
如为未下载数据集，可使用提供的脚本进行一键下载以及数据准备，如下所示：

```shell
# Download and creat json
python mindaudio/data/librispeech.py --root_path "your_data_path"
```

如已下载好压缩文件，请按如下命令操作：

```shell
# creat json
python mindaudio/data/librispeech.py --root_path "your_data_path"  --data_ready True
```

LibriSpeech存储flac音频格式的文件。要在MindAudio中使用它们，须将所有flac文件转换为wav文件，用户可以使用[ffmpeg](https://gist.github.com/seungwonpark/4f273739beef2691cd53b5c39629d830)或[sox](https://sourceforge.net/projects/sox/)进行转换。

处理后，数据集目录结构如下所示:

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

4个**.json文件存储了相应数据的绝对路径，在后续进行模型训练以及验证中，请将yaml配置文件中的xx_manifest改为对应libri_xx_manifest.json的存放地址。

### 2. 训练
#### 单卡训练
由于数据集较大，不推荐使用此种训练方式
```shell
# Standalone training
python train.py -c "./deepspeech2.yaml"
```

注意:默认使用Ascend机器

#### 在Ascend上进行多卡训练

此样例使用 8张NPU，如果你想改变NPU的数量，可以更改下列命令中 -n 后的卡数。
```shell
# Distribute_training
mpirun -n 8 python train.py -c "./deepspeech2.yaml"
```
注意:如果脚本是由root用户执行的，必须在mpirun中添加——allow-run-as-root参数，如下所示:
```shell
mpirun --allow-run-as-root -n 8 python train.py -c "./deepspeech2.yaml"
```


### 3.评估模型

将训好的权重地址更新在deepspeech2.yaml配置文件Pretrained_model中，执行以下命令
```shell
# Validate a trained model
python eval.py -c "./deepspeech2.yaml"
```



## **模型表现**

| 模型        | 机器     | LM   | Test Clean CER | Test Clean WER | 参数                                                                                               | 权重                                                         |
| ----------- | -------- | ---- | -------------- | -------------- |--------------------------------------------------------------------------------------------------| ------------------------------------------------------------ |
| DeepSpeech2 | D910x8-G | No   | 3.461          | 10.24          | [yaml](https://github.com/mindsporelab/mindaudio/blob/main/example/deepspeech2/deepspeech2.yaml) | [weights](https://download.mindspore.cn/toolkits/mindaudio/deepspeech2/deepspeech2.ckpt) |
