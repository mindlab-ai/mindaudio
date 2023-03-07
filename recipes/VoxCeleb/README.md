# Speaker recognition experiments with VoxCeleb.
This folder contains scripts for running speaker identification and verification experiments with the VoxCeleb dataset(http://www.robots.ox.ac.uk/~vgg/data/voxceleb/).

# Speaker verification using ECAPA-TDNN embeddings
Run the following command to train speaker embeddings using [ECAPA-TDNN](https://arxiv.org/abs/2005.07143):

`python train_speaker_embeddings.py`

The speaker-id accuracy should be around 98-99% for both voxceleb1 and voceleb2.

After training the speaker embeddings, it is possible to perform speaker verification using cosine similarity.  You can run it with the following command:

`python speaker_verification_cosine.py`

This system achieves:
- EER = 1.50% (voxceleb1 + voxceleb2) with s-norm
- EER = 1.70% (voxceleb1 + voxceleb2) without s-norm

These results are all obtained with the official verification split of voxceleb1 (veri\_test2.txt)

Below you can find the results from model trained on VoxCeleb 2 dev set and tested on VoxSRC derivatives. Note that however, the models are trained on Ascend910 with 8 cards.

# VoxCeleb2 preparation
Voxceleb2 audio files are released in m4a format. All the files must be converted in wav files before
feeding them is SpeechBrain. Please, follow these steps to prepare the dataset correctly:

1. Download both Voxceleb1 and Voxceleb2.
You can find download instructions here: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/
Note that for the speaker verification experiments with Voxceleb2 the official split of voxceleb1 is used to compute EER.

2. Convert .m4a to wav
Voxceleb2 stores files with the m4a audio format. To use them within SpeechBrain you have to convert all the m4a files into wav files.
You can do the conversion using ffmpeg(https://gist.github.com/seungwonpark/4f273739beef2691cd53b5c39629d830). This operation might take several hours and should be only once.

2. Put all the wav files in a folder called wav. You should have something like `voxceleb2/wav/id*/*.wav` (e.g, `voxceleb2/wav/id00012/21Uxsk56VDQ/00001.wav`)

3. copy the `voxceleb1/vox1_test_wav.zip` file into the voxceleb2 folder.

4. Unpack voxceleb1 test files(verification split).

Go to the voxceleb2 folder and run `unzip vox1_test_wav.zip`.

5. Copy the verification split(`voxceleb1/meta/veri_test2.txt`) into voxceleb2(`voxceleb2/meta/veri_test2.txt`)

6. Now everything is ready and you can run voxceleb2 experiments:
- training embeddings:

`python train_speaker_embeddings.py`

Note: To prepare the voxceleb1 + voxceleb2 dataset you have to copy and unpack vox1_dev_wav.zip for the voxceleb1 dataset.

# Performance summary

[Speaker verification results with Voxceleb 1 + Voxceleb 2]
| System          | Dataset    | EER  |
|-----------------|------------|------|
| ECAPA-TDNN      | Voxceleb 1,2 | 1.50% |

```


