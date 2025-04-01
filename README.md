<div align="center">
<p align="center">
  <h2>Video to Audio and Piano</h2>
  <a href="https://arxiv.org/abs/2503.22200">Paper</a> | <a href="https://acappemin.github.io/Video-to-Audio-and-Piano.github.io">Webpage</a> | <a href="https://huggingface.co/spaces/lshzhm/Video-to-Audio-and-Piano">Huggingface Demo</a> | <a href="https://huggingface.co/lshzhm/Video-to-Audio-and-Piano/tree/main">Models</a>
</p>
</div>

## Enhance Generation Quality of Flow Matching V2A Model via Multi-Step CoT-Like Guidance and Combined Preference Optimization

[Haomin Zhang](https://scholar.google.com/citations?user=cxj9ZbAAAAAJ), [Sizhe Shan](), [Haoyu Wang](), [Zihao Chen](), [Xiulong Liu](), [Chaofan Ding](), [Xinhan Di]()

AI Lab Giant Network, Zhejiang University, University of Washington

## Results

<img width="100%" alt="Image" src="https://github.com/user-attachments/assets/8ba666a6-42c7-4d46-80fb-00a44802d701"/>

**1. Results of Video-to-Audio Synthesis**

https://github.com/user-attachments/assets/d6761371-8fc2-427c-8b2b-6d2ac22a2db2

https://github.com/user-attachments/assets/50b33e54-8ba1-4fab-89d3-5a5cc4c22c9a

**2. Results of Video-to-Piano Synthesis**

https://github.com/user-attachments/assets/b6218b94-1d58-4dc5-873a-c3e8eef6cd67

https://github.com/user-attachments/assets/ebdd1d95-2d9e-4add-b61a-d181f0ae38d0


## Installation

**1. Create a conda environment**

```bash
conda create -n v2ap python=3.10
conda activate v2ap
```

**2. Install requirements**

```bash
pip install -r requirements.txt
```


**Pretrained models**

The models are available at https://huggingface.co/lshzhm/Video-to-Audio-and-Piano/tree/main.


## Inference

**1. Video-to-Audio inference**

```bash
python src/inference_v2a.py
```

**2. Video-to-Piano inference**

```bash
python src/inference_v2p.py
```

## Dateset is in progress


## Acknowledgement

- [Audeo](https://github.com/shlizee/Audeo) for video to midi prediction
- [E2TTS](https://github.com/lucidrains/e2-tts-pytorch) for CFM structure and base E2 implementation
- [FLAN-T5](https://huggingface.co/google/flan-t5-large) for FLAN-T5 text encode
- [CLIP](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) for CLIP image encode
- [AudioLDM Eval](https://github.com/haoheliu/audioldm_eval) for audio evaluation

