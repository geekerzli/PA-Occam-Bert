# PA-Occam-Bert

## Overview

This repo contains the code of **PA-Occam-Bert**: New State-of-the-Art for DAWNBench SQuAD.

PA-Occam-Bert improves inference latency of SQuAD2.0 to **7.579ms** with **F1 score 75.89** by applying faster transformer encoder in BERT. Visit [NVIDIA/Faster Transformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer) for more information

## Environment

- CMake >= 3.8
- Nvidia driver 418.67
- Python 2.7
- Tensorflow-gpu 1.13
- cuda10.1
- cudnn7

## Results Reproduction 

- Fine-tune **BERT-Base** on SQuAD 2.0 task. 

- Save model checkpoints and **evalutate F1** with [evaluate-v.2.0.py](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/).

- **Build Faster Transformer** 

  -  Build with tensorflow mode. Checkout  [Faster Transformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer) for detailed instructions.

  ```shell
  $ mkdir -p build
  $ cd build
  $ cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_TF=ON -DTF_PATH=/your/path/to/tensorflow-gpu .. 
  $ make
  ```

  Note: xx is the compute capability of your GPU. For example, 60 (P40) or 61 (P4) or 70 (V100) or 75(T4).

- copy **libtf_fastertransformer.so** and **libtf_fastertransformer.a** to PA-Occam-Bert/lib

  ```shell
  $ cp your_local_fastertransformer/lib/btf_fastertransformer.so PA-Occam-Bert/lib
  $ cp your_local_fastertransformer/lib/btf_fastertransformer.a PA-Occam-Bert/lib
  ```

- Generate optimized **GEMM** algorithm file. For batch_size=1, sequence length=128, head_num=12, size_per_head=64，use the script below

  ```shell
  $ ./bin/gemm_fp32 1 128 12 64
  ```

- **Inferencing**

  ```shell
  $ mkdir YOUR_BERT_MODEL_PATH
  $ mkdir SQUAD_DATA_PATH
  $ export BERT_BASE_DIR = YOUR_BERT_MODEL_PATH
  $ export SQUAD_DIR = SUQAD_DATA_PATH
  ```

  Note: *YOUR_BERT_MODEL_PATH* should includes vocab.txt, bert_config.json, bert_model.ckpt which can be download from [Bert](https://github.com/google-research/bert). *SUQAD_DATA_PATH* should includes [train-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O $SQUAD_DIR/train-v1.1.json) and [dev-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O $SQUAD_DIR/dev-v1.1.json) (wget to download)

  ```shell
  $ cd build-faster-trans
  $ python run_squad.py \
  $   --vocab_file = $BERT_BASE_DIR/vocab.txt \
  $   --bert_config = $BERT_BASE_DIR/bert_config.json \
  $   --init_checkpoint = $BERT_BASE_DIR/bert_model.ckpt \
  $   --do_train = False \
  $   --train_file = $SQUAD_DIR/train-v2.0.json \
  $   --do_predict = True \
  $   --predict_file = $SQUAD_DIR/dev-v2.0.json \
  $   --max_seq_length = 128 \
  $   --doc_stride = 128 \
  $   --output_dir = ./squad_out \
  $   --version_2_with_negative = True \
  $   --floatx="float32" \
  $   --use_fasterTF = True
  ```

## Reference

```
@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```
