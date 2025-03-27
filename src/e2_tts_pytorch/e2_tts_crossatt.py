"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
dt - dimension text
"""

from __future__ import annotations

from pathlib import Path
from random import random
from functools import partial
from itertools import zip_longest
from collections import namedtuple

from typing import Literal, Callable

import jaxtyping
from beartype import beartype

import torch
import torch.nn.functional as F
from torch import nn, tensor, Tensor, from_numpy
from torch.nn import Module, ModuleList, Sequential, Linear
from torch.nn.utils.rnn import pad_sequence

import torchaudio
from torchaudio.functional import DB_to_amplitude
from torchdiffeq import odeint

import einx
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, pack, unpack

from x_transformers import (
    Attention,
    FeedForward,
    RMSNorm,
    AdaptiveRMSNorm,
)

from x_transformers.x_transformers import RotaryEmbedding

import sys
sys.path.insert(0, "/zhanghaomin/codes3/vocos-main/")
from vocos import Vocos

from transformers import AutoTokenizer
from transformers import T5EncoderModel
from transformers import EncodecModel, AutoProcessor

import os
import math
import traceback
import numpy as np
from moviepy.editor import AudioFileClip, VideoFileClip
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
#import open_clip
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import time

import warnings
warnings.filterwarnings("ignore")

def normalize_wav(waveform):
    waveform = waveform - torch.mean(waveform)
    waveform = waveform / (torch.max(torch.abs(waveform[0, :])) + 1e-8)
    return waveform * 0.5

def read_frames_with_moviepy(video_path, max_frame_nums=None):
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        print("########", duration)
        CUT = False
        if duration > 30.0:
            duration = 30.0
            CUT = True
        fps = clip.fps
        frames = []
        for frame in clip.iter_frames():
            frames.append(frame)
            if CUT and len(frames) >= fps*duration:
                break
    except:
        print("Error read_frames_with_moviepy", video_path)
        traceback.print_exc()
        return None, None, None
    if max_frame_nums is not None:
        frames_idx = np.linspace(0, len(frames) - 1, max_frame_nums, dtype=int)
        return np.array(frames)[frames_idx, ...], duration, fps
    else:
        return np.array(frames), duration, fps

pad_sequence = partial(pad_sequence, batch_first = True)

# constants

class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        return self.abstract_dtype[Tensor, shapes]

Float = TorchTyping(jaxtyping.Float)
Int   = TorchTyping(jaxtyping.Int)
Bool  = TorchTyping(jaxtyping.Bool)

# named tuples

LossBreakdown = namedtuple('LossBreakdown', ['flow', 'velocity_consistency'])

E2TTSReturn = namedtuple('E2TTS', ['loss', 'cond', 'pred_flow', 'pred_data', 'loss_breakdown'])

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def pack_one_with_inverse(x, pattern):
    packed, packed_shape = pack([x], pattern)

    def inverse(x, inverse_pattern = None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse

class Identity(Module):
    def forward(self, x, **kwargs):
        return x

# tensor helpers

def project(x, y):
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')

    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = F.normalize(y, dim = -1)

    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthogonal = x - parallel

    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)

# simple utf-8 tokenizer, since paper went character based

def list_str_to_tensor(
    text: list[str],
    padding_value = -1
) -> Int['b nt']:

    list_tensors = [tensor([*bytes(t, 'UTF-8')]) for t in text]
    padded_tensor = pad_sequence(list_tensors, padding_value = -1)
    return padded_tensor

# simple english phoneme-based tokenizer

from g2p_en import G2p
import jieba
from pypinyin import lazy_pinyin, Style

def get_g2p_en_encode():
    g2p = G2p()

    # used by @lucasnewman successfully here
    # https://github.com/lucasnewman/e2-tts-pytorch/blob/ljspeech-test/e2_tts_pytorch/e2_tts.py

    phoneme_to_index = g2p.p2idx
    num_phonemes = len(phoneme_to_index)

    extended_chars = [' ', ',', '.', '-', '!', '?', '\'', '"', '...', '..', '. .', '. . .', '. . . .', '. . . . .', '. ...', '... .', '.. ..']
    num_extended_chars = len(extended_chars)

    extended_chars_dict = {p: (num_phonemes + i) for i, p in enumerate(extended_chars)}
    phoneme_to_index = {**phoneme_to_index, **extended_chars_dict}

    def encode(
        text: list[str],
        padding_value = -1
    ) -> Int['b nt']:

        phonemes = [g2p(t) for t in text]
        list_tensors = [tensor([phoneme_to_index[p] for p in one_phoneme]) for one_phoneme in phonemes]
        padded_tensor = pad_sequence(list_tensors, padding_value = -1)
        return padded_tensor

    return encode, (num_phonemes + num_extended_chars)

def all_en(word):
    res = word.replace("'", "").encode('utf-8').isalpha()
    return res

def all_ch(word):
    res = True
    for w in word:
        if not '\u4e00' <= w <= '\u9fff':
            res = False
    return res

def get_g2p_zh_encode():
    puncs = ['，', '。', '？', '、']
    pinyins = ['a', 'a1', 'ai1', 'ai2', 'ai3', 'ai4', 'an1', 'an3', 'an4', 'ang1', 'ang2', 'ang4', 'ao1', 'ao2', 'ao3', 'ao4', 'ba', 'ba1', 'ba2', 'ba3', 'ba4', 'bai1', 'bai2', 'bai3', 'bai4', 'ban1', 'ban2', 'ban3', 'ban4', 'bang1', 'bang2', 'bang3', 'bang4', 'bao1', 'bao2', 'bao3', 'bao4', 'bei', 'bei1', 'bei2', 'bei3', 'bei4', 'ben1', 'ben2', 'ben3', 'ben4', 'beng1', 'beng2', 'beng4', 'bi1', 'bi2', 'bi3', 'bi4', 'bian1', 'bian2', 'bian3', 'bian4', 'biao1', 'biao2', 'biao3', 'bie1', 'bie2', 'bie3', 'bie4', 'bin1', 'bin4', 'bing1', 'bing2', 'bing3', 'bing4', 'bo', 'bo1', 'bo2', 'bo3', 'bo4', 'bu2', 'bu3', 'bu4', 'ca1', 'cai1', 'cai2', 'cai3', 'cai4', 'can1', 'can2', 'can3', 'can4', 'cang1', 'cang2', 'cao1', 'cao2', 'cao3', 'ce4', 'cen1', 'cen2', 'ceng1', 'ceng2', 'ceng4', 'cha1', 'cha2', 'cha3', 'cha4', 'chai1', 'chai2', 'chan1', 'chan2', 'chan3', 'chan4', 'chang1', 'chang2', 'chang3', 'chang4', 'chao1', 'chao2', 'chao3', 'che1', 'che2', 'che3', 'che4', 'chen1', 'chen2', 'chen3', 'chen4', 'cheng1', 'cheng2', 'cheng3', 'cheng4', 'chi1', 'chi2', 'chi3', 'chi4', 'chong1', 'chong2', 'chong3', 'chong4', 'chou1', 'chou2', 'chou3', 'chou4', 'chu1', 'chu2', 'chu3', 'chu4', 'chua1', 'chuai1', 'chuai2', 'chuai3', 'chuai4', 'chuan1', 'chuan2', 'chuan3', 'chuan4', 'chuang1', 'chuang2', 'chuang3', 'chuang4', 'chui1', 'chui2', 'chun1', 'chun2', 'chun3', 'chuo1', 'chuo4', 'ci1', 'ci2', 'ci3', 'ci4', 'cong1', 'cong2', 'cou4', 'cu1', 'cu4', 'cuan1', 'cuan2', 'cuan4', 'cui1', 'cui3', 'cui4', 'cun1', 'cun2', 'cun4', 'cuo1', 'cuo2', 'cuo4', 'da', 'da1', 'da2', 'da3', 'da4', 'dai1', 'dai3', 'dai4', 'dan1', 'dan2', 'dan3', 'dan4', 'dang1', 'dang2', 'dang3', 'dang4', 'dao1', 'dao2', 'dao3', 'dao4', 'de', 'de1', 'de2', 'dei3', 'den4', 'deng1', 'deng2', 'deng3', 'deng4', 'di1', 'di2', 'di3', 'di4', 'dia3', 'dian1', 'dian2', 'dian3', 'dian4', 'diao1', 'diao3', 'diao4', 'die1', 'die2', 'ding1', 'ding2', 'ding3', 'ding4', 'diu1', 'dong1', 'dong3', 'dong4', 'dou1', 'dou2', 'dou3', 'dou4', 'du1', 'du2', 'du3', 'du4', 'duan1', 'duan2', 'duan3', 'duan4', 'dui1', 'dui4', 'dun1', 'dun3', 'dun4', 'duo1', 'duo2', 'duo3', 'duo4', 'e1', 'e2', 'e3', 'e4', 'ei2', 'en1', 'en4', 'er', 'er2', 'er3', 'er4', 'fa1', 'fa2', 'fa3', 'fa4', 'fan1', 'fan2', 'fan3', 'fan4', 'fang1', 'fang2', 'fang3', 'fang4', 'fei1', 'fei2', 'fei3', 'fei4', 'fen1', 'fen2', 'fen3', 'fen4', 'feng1', 'feng2', 'feng3', 'feng4', 'fo2', 'fou2', 'fou3', 'fu1', 'fu2', 'fu3', 'fu4', 'ga1', 'ga2', 'ga4', 'gai1', 'gai3', 'gai4', 'gan1', 'gan2', 'gan3', 'gan4', 'gang1', 'gang2', 'gang3', 'gang4', 'gao1', 'gao2', 'gao3', 'gao4', 'ge1', 'ge2', 'ge3', 'ge4', 'gei2', 'gei3', 'gen1', 'gen2', 'gen3', 'gen4', 'geng1', 'geng3', 'geng4', 'gong1', 'gong3', 'gong4', 'gou1', 'gou2', 'gou3', 'gou4', 'gu', 'gu1', 'gu2', 'gu3', 'gu4', 'gua1', 'gua2', 'gua3', 'gua4', 'guai1', 'guai2', 'guai3', 'guai4', 'guan1', 'guan2', 'guan3', 'guan4', 'guang1', 'guang2', 'guang3', 'guang4', 'gui1', 'gui2', 'gui3', 'gui4', 'gun3', 'gun4', 'guo1', 'guo2', 'guo3', 'guo4', 'ha1', 'ha2', 'ha3', 'hai1', 'hai2', 'hai3', 'hai4', 'han1', 'han2', 'han3', 'han4', 'hang1', 'hang2', 'hang4', 'hao1', 'hao2', 'hao3', 'hao4', 'he1', 'he2', 'he4', 'hei1', 'hen2', 'hen3', 'hen4', 'heng1', 'heng2', 'heng4', 'hong1', 'hong2', 'hong3', 'hong4', 'hou1', 'hou2', 'hou3', 'hou4', 'hu1', 'hu2', 'hu3', 'hu4', 'hua1', 'hua2', 'hua4', 'huai2', 'huai4', 'huan1', 'huan2', 'huan3', 'huan4', 'huang1', 'huang2', 'huang3', 'huang4', 'hui1', 'hui2', 'hui3', 'hui4', 'hun1', 'hun2', 'hun4', 'huo', 'huo1', 'huo2', 'huo3', 'huo4', 'ji1', 'ji2', 'ji3', 'ji4', 'jia', 'jia1', 'jia2', 'jia3', 'jia4', 'jian1', 'jian2', 'jian3', 'jian4', 'jiang1', 'jiang2', 'jiang3', 'jiang4', 'jiao1', 'jiao2', 'jiao3', 'jiao4', 'jie1', 'jie2', 'jie3', 'jie4', 'jin1', 'jin2', 'jin3', 'jin4', 'jing1', 'jing2', 'jing3', 'jing4', 'jiong3', 'jiu1', 'jiu2', 'jiu3', 'jiu4', 'ju1', 'ju2', 'ju3', 'ju4', 'juan1', 'juan2', 'juan3', 'juan4', 'jue1', 'jue2', 'jue4', 'jun1', 'jun4', 'ka1', 'ka2', 'ka3', 'kai1', 'kai2', 'kai3', 'kai4', 'kan1', 'kan2', 'kan3', 'kan4', 'kang1', 'kang2', 'kang4', 'kao2', 'kao3', 'kao4', 'ke1', 'ke2', 'ke3', 'ke4', 'ken3', 'keng1', 'kong1', 'kong3', 'kong4', 'kou1', 'kou2', 'kou3', 'kou4', 'ku1', 'ku2', 'ku3', 'ku4', 'kua1', 'kua3', 'kua4', 'kuai3', 'kuai4', 'kuan1', 'kuan2', 'kuan3', 'kuang1', 'kuang2', 'kuang4', 'kui1', 'kui2', 'kui3', 'kui4', 'kun1', 'kun3', 'kun4', 'kuo4', 'la', 'la1', 'la2', 'la3', 'la4', 'lai2', 'lai4', 'lan2', 'lan3', 'lan4', 'lang1', 'lang2', 'lang3', 'lang4', 'lao1', 'lao2', 'lao3', 'lao4', 'le', 'le1', 'le4', 'lei', 'lei1', 'lei2', 'lei3', 'lei4', 'leng1', 'leng2', 'leng3', 'leng4', 'li', 'li1', 'li2', 'li3', 'li4', 'lia3', 'lian2', 'lian3', 'lian4', 'liang2', 'liang3', 'liang4', 'liao1', 'liao2', 'liao3', 'liao4', 'lie1', 'lie2', 'lie3', 'lie4', 'lin1', 'lin2', 'lin3', 'lin4', 'ling2', 'ling3', 'ling4', 'liu1', 'liu2', 'liu3', 'liu4', 'long1', 'long2', 'long3', 'long4', 'lou1', 'lou2', 'lou3', 'lou4', 'lu1', 'lu2', 'lu3', 'lu4', 'luan2', 'luan3', 'luan4', 'lun1', 'lun2', 'lun4', 'luo1', 'luo2', 'luo3', 'luo4', 'lv2', 'lv3', 'lv4', 'lve3', 'lve4', 'ma', 'ma1', 'ma2', 'ma3', 'ma4', 'mai2', 'mai3', 'mai4', 'man2', 'man3', 'man4', 'mang2', 'mang3', 'mao1', 'mao2', 'mao3', 'mao4', 'me', 'mei2', 'mei3', 'mei4', 'men', 'men1', 'men2', 'men4', 'meng1', 'meng2', 'meng3', 'meng4', 'mi1', 'mi2', 'mi3', 'mi4', 'mian2', 'mian3', 'mian4', 'miao1', 'miao2', 'miao3', 'miao4', 'mie1', 'mie4', 'min2', 'min3', 'ming2', 'ming3', 'ming4', 'miu4', 'mo1', 'mo2', 'mo3', 'mo4', 'mou1', 'mou2', 'mou3', 'mu2', 'mu3', 'mu4', 'n2', 'na1', 'na2', 'na3', 'na4', 'nai2', 'nai3', 'nai4', 'nan1', 'nan2', 'nan3', 'nan4', 'nang1', 'nang2', 'nao1', 'nao2', 'nao3', 'nao4', 'ne', 'ne2', 'ne4', 'nei3', 'nei4', 'nen4', 'neng2', 'ni1', 'ni2', 'ni3', 'ni4', 'nian1', 'nian2', 'nian3', 'nian4', 'niang2', 'niang4', 'niao2', 'niao3', 'niao4', 'nie1', 'nie4', 'nin2', 'ning2', 'ning3', 'ning4', 'niu1', 'niu2', 'niu3', 'niu4', 'nong2', 'nong4', 'nou4', 'nu2', 'nu3', 'nu4', 'nuan3', 'nuo2', 'nuo4', 'nv2', 'nv3', 'nve4', 'o1', 'o2', 'ou1', 'ou3', 'ou4', 'pa1', 'pa2', 'pa4', 'pai1', 'pai2', 'pai3', 'pai4', 'pan1', 'pan2', 'pan4', 'pang1', 'pang2', 'pang4', 'pao1', 'pao2', 'pao3', 'pao4', 'pei1', 'pei2', 'pei4', 'pen1', 'pen2', 'pen4', 'peng1', 'peng2', 'peng3', 'peng4', 'pi1', 'pi2', 'pi3', 'pi4', 'pian1', 'pian2', 'pian4', 'piao1', 'piao2', 'piao3', 'piao4', 'pie1', 'pie2', 'pie3', 'pin1', 'pin2', 'pin3', 'pin4', 'ping1', 'ping2', 'po1', 'po2', 'po3', 'po4', 'pou1', 'pu1', 'pu2', 'pu3', 'pu4', 'qi1', 'qi2', 'qi3', 'qi4', 'qia1', 'qia3', 'qia4', 'qian1', 'qian2', 'qian3', 'qian4', 'qiang1', 'qiang2', 'qiang3', 'qiang4', 'qiao1', 'qiao2', 'qiao3', 'qiao4', 'qie1', 'qie2', 'qie3', 'qie4', 'qin1', 'qin2', 'qin3', 'qin4', 'qing1', 'qing2', 'qing3', 'qing4', 'qiong1', 'qiong2', 'qiu1', 'qiu2', 'qiu3', 'qu1', 'qu2', 'qu3', 'qu4', 'quan1', 'quan2', 'quan3', 'quan4', 'que1', 'que2', 'que4', 'qun2', 'ran2', 'ran3', 'rang1', 'rang2', 'rang3', 'rang4', 'rao2', 'rao3', 'rao4', 're2', 're3', 're4', 'ren2', 'ren3', 'ren4', 'reng1', 'reng2', 'ri4', 'rong1', 'rong2', 'rong3', 'rou2', 'rou4', 'ru2', 'ru3', 'ru4', 'ruan2', 'ruan3', 'rui3', 'rui4', 'run4', 'ruo4', 'sa1', 'sa2', 'sa3', 'sa4', 'sai1', 'sai4', 'san1', 'san2', 'san3', 'san4', 'sang1', 'sang3', 'sang4', 'sao1', 'sao2', 'sao3', 'sao4', 'se4', 'sen1', 'seng1', 'sha1', 'sha2', 'sha3', 'sha4', 'shai1', 'shai2', 'shai3', 'shai4', 'shan1', 'shan3', 'shan4', 'shang', 'shang1', 'shang3', 'shang4', 'shao1', 'shao2', 'shao3', 'shao4', 'she1', 'she2', 'she3', 'she4', 'shei2', 'shen1', 'shen2', 'shen3', 'shen4', 'sheng1', 'sheng2', 'sheng3', 'sheng4', 'shi', 'shi1', 'shi2', 'shi3', 'shi4', 'shou1', 'shou2', 'shou3', 'shou4', 'shu1', 'shu2', 'shu3', 'shu4', 'shua1', 'shua2', 'shua3', 'shua4', 'shuai1', 'shuai3', 'shuai4', 'shuan1', 'shuan4', 'shuang1', 'shuang3', 'shui2', 'shui3', 'shui4', 'shun3', 'shun4', 'shuo1', 'shuo4', 'si1', 'si2', 'si3', 'si4', 'song1', 'song3', 'song4', 'sou1', 'sou3', 'sou4', 'su1', 'su2', 'su4', 'suan1', 'suan4', 'sui1', 'sui2', 'sui3', 'sui4', 'sun1', 'sun3', 'suo', 'suo1', 'suo2', 'suo3', 'ta1', 'ta3', 'ta4', 'tai1', 'tai2', 'tai4', 'tan1', 'tan2', 'tan3', 'tan4', 'tang1', 'tang2', 'tang3', 'tang4', 'tao1', 'tao2', 'tao3', 'tao4', 'te4', 'teng2', 'ti1', 'ti2', 'ti3', 'ti4', 'tian1', 'tian2', 'tian3', 'tiao1', 'tiao2', 'tiao3', 'tiao4', 'tie1', 'tie2', 'tie3', 'tie4', 'ting1', 'ting2', 'ting3', 'tong1', 'tong2', 'tong3', 'tong4', 'tou', 'tou1', 'tou2', 'tou4', 'tu1', 'tu2', 'tu3', 'tu4', 'tuan1', 'tuan2', 'tui1', 'tui2', 'tui3', 'tui4', 'tun1', 'tun2', 'tun4', 'tuo1', 'tuo2', 'tuo3', 'tuo4', 'wa', 'wa1', 'wa2', 'wa3', 'wa4', 'wai1', 'wai3', 'wai4', 'wan1', 'wan2', 'wan3', 'wan4', 'wang1', 'wang2', 'wang3', 'wang4', 'wei1', 'wei2', 'wei3', 'wei4', 'wen1', 'wen2', 'wen3', 'wen4', 'weng1', 'weng4', 'wo1', 'wo3', 'wo4', 'wu1', 'wu2', 'wu3', 'wu4', 'xi1', 'xi2', 'xi3', 'xi4', 'xia1', 'xia2', 'xia4', 'xian1', 'xian2', 'xian3', 'xian4', 'xiang1', 'xiang2', 'xiang3', 'xiang4', 'xiao1', 'xiao2', 'xiao3', 'xiao4', 'xie1', 'xie2', 'xie3', 'xie4', 'xin1', 'xin2', 'xin4', 'xing1', 'xing2', 'xing3', 'xing4', 'xiong1', 'xiong2', 'xiu1', 'xiu3', 'xiu4', 'xu', 'xu1', 'xu2', 'xu3', 'xu4', 'xuan1', 'xuan2', 'xuan3', 'xuan4', 'xue1', 'xue2', 'xue3', 'xue4', 'xun1', 'xun2', 'xun4', 'ya', 'ya1', 'ya2', 'ya3', 'ya4', 'yan1', 'yan2', 'yan3', 'yan4', 'yang1', 'yang2', 'yang3', 'yang4', 'yao1', 'yao2', 'yao3', 'yao4', 'ye1', 'ye2', 'ye3', 'ye4', 'yi1', 'yi2', 'yi3', 'yi4', 'yin1', 'yin2', 'yin3', 'yin4', 'ying1', 'ying2', 'ying3', 'ying4', 'yo1', 'yong1', 'yong3', 'yong4', 'you1', 'you2', 'you3', 'you4', 'yu1', 'yu2', 'yu3', 'yu4', 'yuan1', 'yuan2', 'yuan3', 'yuan4', 'yue1', 'yue4', 'yun1', 'yun2', 'yun3', 'yun4', 'za1', 'za2', 'za3', 'zai1', 'zai3', 'zai4', 'zan1', 'zan2', 'zan3', 'zan4', 'zang1', 'zang4', 'zao1', 'zao2', 'zao3', 'zao4', 'ze2', 'ze4', 'zei2', 'zen3', 'zeng1', 'zeng4', 'zha1', 'zha2', 'zha3', 'zha4', 'zhai1', 'zhai2', 'zhai3', 'zhai4', 'zhan1', 'zhan2', 'zhan3', 'zhan4', 'zhang1', 'zhang2', 'zhang3', 'zhang4', 'zhao1', 'zhao2', 'zhao3', 'zhao4', 'zhe', 'zhe1', 'zhe2', 'zhe3', 'zhe4', 'zhen1', 'zhen2', 'zhen3', 'zhen4', 'zheng1', 'zheng2', 'zheng3', 'zheng4', 'zhi1', 'zhi2', 'zhi3', 'zhi4', 'zhong1', 'zhong2', 'zhong3', 'zhong4', 'zhou1', 'zhou2', 'zhou3', 'zhou4', 'zhu1', 'zhu2', 'zhu3', 'zhu4', 'zhua1', 'zhua2', 'zhua3', 'zhuai1', 'zhuai3', 'zhuai4', 'zhuan1', 'zhuan2', 'zhuan3', 'zhuan4', 'zhuang1', 'zhuang4', 'zhui1', 'zhui4', 'zhun1', 'zhun2', 'zhun3', 'zhuo1', 'zhuo2', 'zi', 'zi1', 'zi2', 'zi3', 'zi4', 'zong1', 'zong2', 'zong3', 'zong4', 'zou1', 'zou2', 'zou3', 'zou4', 'zu1', 'zu2', 'zu3', 'zuan1', 'zuan3', 'zuan4', 'zui2', 'zui3', 'zui4', 'zun1', 'zuo1', 'zuo2', 'zuo3', 'zuo4']
    ens = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'", ' ']
    ens_U = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    phoneme_to_index = {}
    num_phonemes = 0
    for index, punc in enumerate(puncs):
        phoneme_to_index[punc] = index + num_phonemes
    num_phonemes += len(puncs)
    for index, pinyin in enumerate(pinyins):
        phoneme_to_index[pinyin] = index + num_phonemes
    num_phonemes += len(pinyins)
    for index, en in enumerate(ens):
        phoneme_to_index[en] = index + num_phonemes
    for index, en in enumerate(ens_U):
        phoneme_to_index[en] = index + num_phonemes
    num_phonemes += len(ens)
    #print(num_phonemes, phoneme_to_index)  # 1342

    def encode(
        text: list[str],
        padding_value = -1
    ) -> Int['b nt']:
        phonemes = []        
        for t in text:
            one_phoneme = []
            brk = False
            for word in jieba.cut(t):
                if all_ch(word):
                    seg = lazy_pinyin(word, style=Style.TONE3, tone_sandhi=True)
                    one_phoneme.extend(seg)
                elif all_en(word):
                    for seg in word:
                        one_phoneme.append(seg)
                elif word in ["，", "。", "？", "、", "'", " "]:
                    one_phoneme.append(word)
                else:
                    for ch in word:
                        if all_ch(ch):
                            seg = lazy_pinyin(ch, style=Style.TONE3, tone_sandhi=True)
                            one_phoneme.extend(seg)
                        elif all_en(ch):
                            for seg in ch:
                                one_phoneme.append(seg)
                        else:
                            brk = True
                            break
                if brk:
                    break
            if not brk:
                phonemes.append(one_phoneme)
            else:
                print("Error Tokenized", t, list(jieba.cut(t)))
        list_tensors = [tensor([phoneme_to_index[p] for p in one_phoneme]) for one_phoneme in phonemes]
        padded_tensor = pad_sequence(list_tensors, padding_value = -1)
        return padded_tensor

    return encode, num_phonemes

# tensor helpers

def log(t, eps = 1e-5):
    return t.clamp(min = eps).log()

def lens_to_mask(
    t: Int['b'],
    length: int | None = None
) -> Bool['b n']:

    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device = t.device)
    return einx.less('n, b -> b n', seq, t)

def mask_from_start_end_indices(
    seq_len: Int['b'],
    start: Int['b'],
    end: Int['b']
):
    max_seq_len = seq_len.max().item()  
    seq = torch.arange(max_seq_len, device = start.device).long()
    return einx.greater_equal('n, b -> b n', seq, start) & einx.less('n, b -> b n', seq, end)

def mask_from_frac_lengths(
    seq_len: Int['b'],
    frac_lengths: Float['b'],
    max_length: int | None = None,
    val = False
):
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    if not val:
        rand = torch.rand_like(frac_lengths)
    else:
        rand = torch.tensor([0.5]*frac_lengths.shape[0], device=frac_lengths.device).float()
    start = (max_start * rand).long().clamp(min = 0)
    end = start + lengths

    out = mask_from_start_end_indices(seq_len, start, end)

    if exists(max_length):
        out = pad_to_length(out, max_length)

    return out

def maybe_masked_mean(
    t: Float['b n d'],
    mask: Bool['b n'] | None = None
) -> Float['b d']:

    if not exists(mask):
        return t.mean(dim = 1)

    t = einx.where('b n, b n d, -> b n d', mask, t, 0.)
    num = reduce(t, 'b n d -> b d', 'sum')
    den = reduce(mask.float(), 'b n -> b', 'sum')

    return einx.divide('b d, b -> b d', num, den.clamp(min = 1.))

def pad_to_length(
    t: Tensor,
    length: int,
    value = None
):
    seq_len = t.shape[-1]
    if length > seq_len:
        t = F.pad(t, (0, length - seq_len), value = value)

    return t[..., :length]

def interpolate_1d(
    x: Tensor,
    length: int,
    mode = 'bilinear'
):
    x = rearrange(x, 'n d -> 1 d n 1')
    x = F.interpolate(x, (length, 1), mode = mode)
    return rearrange(x, '1 d n 1 -> n d')

# to mel spec

class MelSpec(Module):
    def __init__(
        self,
        filter_length = 1024,
        hop_length = 256,
        win_length = 1024,
        n_mel_channels = 100,
        sampling_rate = 24_000,
        normalize = False,
        power = 1,
        norm = None,
        center = True,
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate

        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            sample_rate = sampling_rate,
            n_fft = filter_length,
            win_length = win_length,
            hop_length = hop_length,
            n_mels = n_mel_channels,
            power = power,
            center = center,
            normalized = normalize,
            norm = norm,
        )

        self.register_buffer('dummy', tensor(0), persistent = False)

    def forward(self, inp):
        if len(inp.shape) == 3:
            inp = rearrange(inp, 'b 1 nw -> b nw')

        assert len(inp.shape) == 2

        if self.dummy.device != inp.device:
            self.to(inp.device)

        mel = self.mel_stft(inp)
        mel = log(mel)
        return mel

class EncodecWrapper(Module):
    def __init__(self, path):
        super().__init__()
        self.model = EncodecModel.from_pretrained(path)
        self.processor = AutoProcessor.from_pretrained(path)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, waveform):
        with torch.no_grad():
            inputs = self.processor(raw_audio=waveform[0], sampling_rate=self.processor.sampling_rate, return_tensors="pt")
            emb = self.model.encoder(inputs.input_values)
        return emb

    def decode(self, emb):
        with torch.no_grad():
            output = self.model.decoder(emb)
        return output[0]

from audioldm.audio.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL
from audioldm.utils import default_audioldm_config, get_metadata

def build_pretrained_models(name):
    checkpoint = torch.load(get_metadata()[name]["path"], map_location="cpu")
    scale_factor = checkpoint["state_dict"]["scale_factor"].item()

    vae_state_dict = {k[18:]: v for k, v in checkpoint["state_dict"].items() if "first_stage_model." in k}

    config = default_audioldm_config(name)
    vae_config = config["model"]["params"]["first_stage_config"]["params"]
    vae_config["scale_factor"] = scale_factor

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(vae_state_dict)

    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )

    vae.eval()
    fn_STFT.eval()
    return vae, fn_STFT

class VaeWrapper(Module):
    def __init__(self):
        super().__init__()
        vae, stft = build_pretrained_models("audioldm-s-full")
        vae.eval()
        stft.eval()
        stft = stft.cpu()
        self.vae = vae
        for param in self.vae.parameters():
            param.requires_grad = False

    def forward(self, waveform):
        return None

    def decode(self, emb):
        with torch.no_grad():
            b, d, l = emb.shape
            latents = emb.transpose(1,2).reshape(b, l, 8, 16).transpose(1,2)
            mel = self.vae.decode_first_stage(latents)
            wave = self.vae.decode_to_waveform(mel)
        return wave

# convolutional positional generating module
# taken from https://github.com/lucidrains/voicebox-pytorch/blob/main/voicebox_pytorch/voicebox_pytorch.py#L203

class DepthwiseConv(Module):
    def __init__(
        self,
        dim,
        *,
        kernel_size,
        groups = None
    ):
        super().__init__()
        assert not divisible_by(kernel_size, 2)
        groups = default(groups, dim) # full depthwise conv by default

        self.dw_conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups = groups, padding = kernel_size // 2),
            nn.SiLU()
        )

    def forward(
        self,
        x,
        mask = None
    ):

        if exists(mask):
            x = einx.where('b n, b n d, -> b n d', mask, x, 0.)

        x = rearrange(x, 'b n c -> b c n')
        x = self.dw_conv1d(x)
        out = rearrange(x, 'b c n -> b n c')

        if exists(mask):
            out = einx.where('b n, b n d, -> b n d', mask, out, 0.)

        return out

# adaln zero from DiT paper

class AdaLNZero(Module):
    def __init__(
        self,
        dim,
        dim_condition = None,
        init_bias_value = -2.
    ):
        super().__init__()
        dim_condition = default(dim_condition, dim)
        self.to_gamma = nn.Linear(dim_condition, dim)

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.constant_(self.to_gamma.bias, init_bias_value)

    def forward(self, x, *, condition):
        if condition.ndim == 2:
            condition = rearrange(condition, 'b d -> b 1 d')

        gamma = self.to_gamma(condition).sigmoid()
        return x * gamma

# random projection fourier embedding

class RandomFourierEmbed(Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        self.register_buffer('weights', torch.randn(dim // 2))

    def forward(self, x):
        freqs = einx.multiply('i, j -> i j', x, self.weights) * 2 * torch.pi
        fourier_embed, _ = pack((x, freqs.sin(), freqs.cos()), 'b *')
        return fourier_embed

# character embedding

class CharacterEmbed(Module):
    def __init__(
        self,
        dim,
        num_embeds = 256,
    ):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(num_embeds + 1, dim) # will just use 0 as the 'filler token'

    def forward(
        self,
        text: Int['b nt'],
        max_seq_len: int,
        **kwargs
    ) -> Float['b n d']:

        text = text + 1 # shift all other token ids up by 1 and use 0 as filler token

        text = text[:, :max_seq_len] # just curtail if character tokens are more than the mel spec tokens, one of the edge cases the paper did not address
        text = pad_to_length(text, max_seq_len, value = 0)

        return self.embed(text)

class InterpolatedCharacterEmbed(Module):
    def __init__(
        self,
        dim,
        num_embeds = 256,
    ):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(num_embeds, dim)

        self.abs_pos_mlp = Sequential(
            Rearrange('... -> ... 1'),
            Linear(1, dim),
            nn.SiLU(),
            Linear(dim, dim)
        )

    def forward(
        self,
        text: Int['b nt'],
        max_seq_len: int,
        mask: Bool['b n'] | None = None
    ) -> Float['b n d']:

        device = text.device

        mask = default(mask, (None,))

        interp_embeds = []
        interp_abs_positions = []

        for one_text, one_mask in zip_longest(text, mask):

            valid_text = one_text >= 0
            one_text = one_text[valid_text]
            one_text_embed = self.embed(one_text)

            # save the absolute positions

            text_seq_len = one_text.shape[0]

            # determine audio sequence length from mask

            audio_seq_len = max_seq_len
            if exists(one_mask):
                audio_seq_len = one_mask.sum().long().item()

            # interpolate text embedding to audio embedding length

            interp_text_embed = interpolate_1d(one_text_embed, audio_seq_len)
            interp_abs_pos = torch.linspace(0, text_seq_len, audio_seq_len, device = device)

            interp_embeds.append(interp_text_embed)
            interp_abs_positions.append(interp_abs_pos)

        interp_embeds = pad_sequence(interp_embeds)
        interp_abs_positions = pad_sequence(interp_abs_positions)

        interp_embeds = F.pad(interp_embeds, (0, 0, 0, max_seq_len - interp_embeds.shape[-2]))
        interp_abs_positions = pad_to_length(interp_abs_positions, max_seq_len)

        # pass interp absolute positions through mlp for implicit positions

        interp_embeds = interp_embeds + self.abs_pos_mlp(interp_abs_positions)

        if exists(mask):
            interp_embeds = einx.where('b n, b n d, -> b n d', mask, interp_embeds, 0.)

        return interp_embeds

# text audio cross conditioning in multistream setup

class TextAudioCrossCondition(Module):
    def __init__(
        self,
        dim,
        dim_text,
        cond_audio_to_text = True,
    ):
        super().__init__()
        self.text_to_audio = nn.Linear(dim_text + dim, dim, bias = False)
        nn.init.zeros_(self.text_to_audio.weight)

        self.cond_audio_to_text = cond_audio_to_text

        if cond_audio_to_text:
            self.audio_to_text = nn.Linear(dim + dim_text, dim_text, bias = False)
            nn.init.zeros_(self.audio_to_text.weight)

    def forward(
        self,
        audio: Float['b n d'],
        text: Float['b n dt']
    ):
        audio_text, _ = pack((audio, text), 'b n *')

        text_cond = self.text_to_audio(audio_text)
        audio_cond = self.audio_to_text(audio_text) if self.cond_audio_to_text else 0.

        return audio + text_cond, text + audio_cond

# attention and transformer backbone
# for use in both e2tts as well as duration module

class Transformer(Module):
    @beartype
    def __init__(
        self,
        *,
        dim,
        dim_text = None, # will default to half of audio dimension
        depth = 8,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        text_depth = None,
        text_heads = None,
        text_dim_head = None,
        text_ff_mult = None,
        cond_on_time = True,
        abs_pos_emb = True,
        max_seq_len = 8192,
        kernel_size = 31,
        dropout = 0.1,
        num_registers = 32,
        attn_kwargs: dict = dict(
            gate_value_heads = True,
            softclamp_logits = True,
        ),
        ff_kwargs: dict = dict(),
        if_text_modules = True,
        if_cross_attn = True,
        if_audio_conv = True,
        if_text_conv = False
    ):
        super().__init__()
        assert divisible_by(depth, 2), 'depth needs to be even'

        # absolute positional embedding

        self.max_seq_len = max_seq_len
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if abs_pos_emb else None

        self.dim = dim

        dim_text = default(dim_text, dim // 2)
        self.dim_text = dim_text

        text_heads = default(text_heads, heads)
        text_dim_head = default(text_dim_head, dim_head)
        text_ff_mult = default(text_ff_mult, ff_mult)
        text_depth = default(text_depth, depth)

        assert 1 <= text_depth <= depth, 'must have at least 1 layer of text conditioning, but less than total number of speech layers'

        self.depth = depth
        self.layers = ModuleList([])

        # registers

        self.num_registers = num_registers
        self.registers = nn.Parameter(torch.zeros(num_registers, dim))
        nn.init.normal_(self.registers, std = 0.02)

        if if_text_modules:
            self.text_registers = nn.Parameter(torch.zeros(num_registers, dim_text))
            nn.init.normal_(self.text_registers, std = 0.02)

        # rotary embedding

        self.rotary_emb = RotaryEmbedding(dim_head)
        self.text_rotary_emb = RotaryEmbedding(dim_head)

        # time conditioning
        # will use adaptive rmsnorm

        self.cond_on_time = cond_on_time
        rmsnorm_klass = RMSNorm if not cond_on_time else AdaptiveRMSNorm
        postbranch_klass = Identity if not cond_on_time else partial(AdaLNZero, dim = dim)

        self.time_cond_mlp = Identity()

        if cond_on_time:
            self.time_cond_mlp = Sequential(
                RandomFourierEmbed(dim),
                Linear(dim + 1, dim),
                nn.SiLU()
            )

        for ind in range(depth):
            is_later_half = ind >= (depth // 2)
            has_text = ind < text_depth

            # speech related
            if if_audio_conv:
                speech_conv = DepthwiseConv(dim, kernel_size = kernel_size)

            attn_norm = rmsnorm_klass(dim)
            attn = Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = dropout, **attn_kwargs)
            attn_adaln_zero = postbranch_klass()

            if if_cross_attn:
                attn_norm2 = rmsnorm_klass(dim)
                attn2 = Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = dropout, **attn_kwargs)
                attn_adaln_zero2 = postbranch_klass()

            ff_norm = rmsnorm_klass(dim)
            ff = FeedForward(dim = dim, glu = True, mult = ff_mult, dropout = dropout, **ff_kwargs)
            ff_adaln_zero = postbranch_klass()

            skip_proj = Linear(dim * 2, dim, bias = False) if is_later_half else None

            if if_cross_attn:
                if if_audio_conv:
                    speech_modules = ModuleList([
                        skip_proj,
                        speech_conv,
                        attn_norm,
                        attn,
                        attn_adaln_zero,
                        attn_norm2,
                        attn2,
                        attn_adaln_zero2,
                        ff_norm,
                        ff,
                        ff_adaln_zero,
                    ])
                else:
                    speech_modules = ModuleList([
                        skip_proj,
                        attn_norm,
                        attn,
                        attn_adaln_zero,
                        attn_norm2,
                        attn2,
                        attn_adaln_zero2,
                        ff_norm,
                        ff,
                        ff_adaln_zero,
                    ])
            else:
                if if_audio_conv:
                    speech_modules = ModuleList([
                        skip_proj,
                        speech_conv,
                        attn_norm,
                        attn,
                        attn_adaln_zero,
                        ff_norm,
                        ff,
                        ff_adaln_zero,
                    ])
                else:
                    speech_modules = ModuleList([
                        skip_proj,
                        attn_norm,
                        attn,
                        attn_adaln_zero,
                        ff_norm,
                        ff,
                        ff_adaln_zero,
                    ])

            text_modules = None

            if has_text and if_text_modules:
                # text related
                if if_text_conv:
                    text_conv = DepthwiseConv(dim_text, kernel_size = kernel_size)

                text_attn_norm = RMSNorm(dim_text)
                text_attn = Attention(dim = dim_text, heads = text_heads, dim_head = text_dim_head, dropout = dropout, **attn_kwargs)

                text_ff_norm = RMSNorm(dim_text)
                text_ff = FeedForward(dim = dim_text, glu = True, mult = text_ff_mult, dropout = dropout, **ff_kwargs)

                # cross condition

                is_last = ind == (text_depth - 1)

                cross_condition = TextAudioCrossCondition(dim = dim, dim_text = dim_text, cond_audio_to_text = not is_last)

                if if_text_conv:
                    text_modules = ModuleList([
                        text_conv,
                        text_attn_norm,
                        text_attn,
                        text_ff_norm,
                        text_ff,
                        cross_condition
                    ])
                else:
                    text_modules = ModuleList([
                        text_attn_norm,
                        text_attn,
                        text_ff_norm,
                        text_ff,
                        cross_condition
                    ])

            self.layers.append(ModuleList([
                speech_modules,
                text_modules
            ]))

        self.final_norm = RMSNorm(dim)

        self.if_cross_attn = if_cross_attn
        self.if_audio_conv = if_audio_conv
        self.if_text_conv = if_text_conv

    def forward(
        self,
        x: Float['b n d'],
        times: Float['b'] | Float[''] | None = None,
        mask: Bool['b n'] | None = None,
        text_embed: Float['b n dt'] | None = None,
        context: Float['b nc dc'] | None = None,
        context_mask: Float['b nc'] | None = None
    ):
        batch, seq_len, device = *x.shape[:2], x.device

        assert not (exists(times) ^ self.cond_on_time), '`times` must be passed in if `cond_on_time` is set to `True` and vice versa'

        # handle absolute positions if needed

        if exists(self.abs_pos_emb):
            assert seq_len <= self.max_seq_len, f'{seq_len} exceeds the set `max_seq_len` ({self.max_seq_len}) on Transformer'
            seq = torch.arange(seq_len, device = device)
            x = x + self.abs_pos_emb(seq)

        # handle adaptive rmsnorm kwargs

        norm_kwargs = dict()

        if exists(times):
            if times.ndim == 0:
                times = repeat(times, ' -> b', b = batch)

            times = self.time_cond_mlp(times)
            norm_kwargs.update(condition = times)

        # register tokens

        registers = repeat(self.registers, 'r d -> b r d', b = batch)
        x, registers_packed_shape = pack((registers, x), 'b * d')

        if exists(mask):
            mask = F.pad(mask, (self.num_registers, 0), value = True)

        # rotary embedding

        rotary_pos_emb = self.rotary_emb.forward_from_seq_len(x.shape[-2])

        # text related

        if exists(text_embed):
            text_rotary_pos_emb = self.text_rotary_emb.forward_from_seq_len(x.shape[-2])

            text_registers = repeat(self.text_registers, 'r d -> b r d', b = batch)
            text_embed, _ = pack((text_registers, text_embed), 'b * d')

        # skip connection related stuff

        skips = []

        # go through the layers

        for ind, (speech_modules, text_modules) in enumerate(self.layers):
            layer = ind + 1

            if self.if_cross_attn:
                if self.if_audio_conv:
                    (
                        maybe_skip_proj,
                        speech_conv,
                        attn_norm,
                        attn,
                        maybe_attn_adaln_zero,
                        attn_norm2,
                        attn2,
                        maybe_attn_adaln_zero2,
                        ff_norm,
                        ff,
                        maybe_ff_adaln_zero
                    ) = speech_modules
                else:
                    (
                        maybe_skip_proj,
                        attn_norm,
                        attn,
                        maybe_attn_adaln_zero,
                        attn_norm2,
                        attn2,
                        maybe_attn_adaln_zero2,
                        ff_norm,
                        ff,
                        maybe_ff_adaln_zero
                    ) = speech_modules
            else:
                if self.if_audio_conv:
                    (
                        maybe_skip_proj,
                        speech_conv,
                        attn_norm,
                        attn,
                        maybe_attn_adaln_zero,
                        ff_norm,
                        ff,
                        maybe_ff_adaln_zero
                    ) = speech_modules
                else:
                    (
                        maybe_skip_proj,
                        attn_norm,
                        attn,
                        maybe_attn_adaln_zero,
                        ff_norm,
                        ff,
                        maybe_ff_adaln_zero
                    ) = speech_modules

            # smaller text transformer

            if exists(text_embed) and exists(text_modules):

                if self.if_text_conv:
                    (
                        text_conv,
                        text_attn_norm,
                        text_attn,
                        text_ff_norm,
                        text_ff,
                        cross_condition
                    ) = text_modules
                else:
                    (
                        text_attn_norm,
                        text_attn,
                        text_ff_norm,
                        text_ff,
                        cross_condition
                    ) = text_modules

                if self.if_text_conv:
                    text_embed = text_conv(text_embed, mask = mask) + text_embed

                text_embed = text_attn(text_attn_norm(text_embed), rotary_pos_emb = text_rotary_pos_emb, mask = mask) + text_embed

                text_embed = text_ff(text_ff_norm(text_embed)) + text_embed

                x, text_embed = cross_condition(x, text_embed)

            # skip connection logic

            is_first_half = layer <= (self.depth // 2)
            is_later_half = not is_first_half

            if is_first_half:
                skips.append(x)

            if is_later_half:
                skip = skips.pop()
                x = torch.cat((x, skip), dim = -1)
                x = maybe_skip_proj(x)

            # position generating convolution

            if self.if_audio_conv:
                x = speech_conv(x, mask = mask) + x

            # attention and feedforward blocks

            attn_out = attn(attn_norm(x, **norm_kwargs), rotary_pos_emb = rotary_pos_emb, mask = mask)

            x = x + maybe_attn_adaln_zero(attn_out, **norm_kwargs)

            if self.if_cross_attn:
                attn_out = attn2(attn_norm2(x, **norm_kwargs), rotary_pos_emb = rotary_pos_emb, mask = mask, context = context, context_mask = context_mask)
                
                x = x + maybe_attn_adaln_zero2(attn_out, **norm_kwargs)

            ff_out = ff(ff_norm(x, **norm_kwargs))

            x = x + maybe_ff_adaln_zero(ff_out, **norm_kwargs)

        assert len(skips) == 0

        _, x = unpack(x, registers_packed_shape, 'b * d')

        return self.final_norm(x)

# main classes

class DurationPredictor(Module):
    @beartype
    def __init__(
        self,
        transformer: dict | Transformer,
        num_channels = None,
        mel_spec_kwargs: dict = dict(),
        char_embed_kwargs: dict = dict(),
        text_num_embeds = None,
        tokenizer: (
            Literal['char_utf8', 'phoneme_en'] |
            Callable[[list[str]], Int['b nt']]
        ) = 'char_utf8'
    ):
        super().__init__()

        if isinstance(transformer, dict):
            transformer = Transformer(
                **transformer,
                cond_on_time = False
            )

        # mel spec

        self.mel_spec = MelSpec(**mel_spec_kwargs)
        self.num_channels = default(num_channels, self.mel_spec.n_mel_channels)

        self.transformer = transformer

        dim = transformer.dim
        dim_text = transformer.dim_text

        self.dim = dim

        self.proj_in = Linear(self.num_channels, self.dim)

        # tokenizer and text embed

        if callable(tokenizer):
            assert exists(text_num_embeds), '`text_num_embeds` must be given if supplying your own tokenizer encode function'
            self.tokenizer = tokenizer
        elif tokenizer == 'char_utf8':
            text_num_embeds = 256
            self.tokenizer = list_str_to_tensor
        elif tokenizer == 'phoneme_en':
            self.tokenizer, text_num_embeds = get_g2p_en_encode()
        elif tokenizer == 'phoneme_zh':
            self.tokenizer, text_num_embeds = get_g2p_zh_encode()
        else:
            raise ValueError(f'unknown tokenizer string {tokenizer}')

        self.embed_text = CharacterEmbed(dim_text, num_embeds = text_num_embeds, **char_embed_kwargs)

        # to prediction

        self.to_pred = Sequential(
            Linear(dim, 1, bias = False),
            nn.Softplus(),
            Rearrange('... 1 -> ...')
        )

    def forward(
        self,
        x: Float['b n d'] | Float['b nw'],
        *,
        text: Int['b nt'] | list[str] | None = None,
        lens: Int['b'] | None = None,
        return_loss = True
    ):
        # raw wave

        if x.ndim == 2:
            x = self.mel_spec(x)
            x = rearrange(x, 'b d n -> b n d')
            assert x.shape[-1] == self.dim

        x = self.proj_in(x)

        batch, seq_len, device = *x.shape[:2], x.device

        # text

        text_embed = None

        if exists(text):
            if isinstance(text, list):
                text = list_str_to_tensor(text).to(device)
                assert text.shape[0] == batch

            text_embed = self.embed_text(text, seq_len)

        # handle lengths (duration)

        if not exists(lens):
            lens = torch.full((batch,), seq_len, device = device)

        mask = lens_to_mask(lens, length = seq_len)

        # if returning a loss, mask out randomly from an index and have it predict the duration

        if return_loss:
            rand_frac_index = x.new_zeros(batch).uniform_(0, 1)
            rand_index = (rand_frac_index * lens).long()

            seq = torch.arange(seq_len, device = device)
            mask &= einx.less('n, b -> b n', seq, rand_index)

        # attending

        x = self.transformer(
            x,
            mask = mask,
            text_embed = text_embed,
        )

        x = maybe_masked_mean(x, mask)

        pred = self.to_pred(x)

        # return the prediction if not returning loss

        if not return_loss:
            return pred

        # loss

        return F.mse_loss(pred, lens.float())

class E2TTS(Module):

    @beartype
    def __init__(
        self,
        transformer: dict | Transformer = None,
        duration_predictor: dict | DurationPredictor | None = None,
        odeint_kwargs: dict = dict(
            #atol = 1e-5,
            #rtol = 1e-5,
            #method = 'midpoint'
            method = "euler"
        ),
        audiocond_drop_prob = 0.30,
        cond_drop_prob = 0.20,
        prompt_drop_prob = 0.10,
        num_channels = None,
        mel_spec_module: Module | None = None,
        char_embed_kwargs: dict = dict(),
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.),
        audiocond_snr: tuple[float, float] | None = None,
        concat_cond = False,
        interpolated_text = False,
        text_num_embeds: int | None = None,
        tokenizer: (
            Literal['char_utf8', 'phoneme_en', 'phoneme_zh'] |
            Callable[[list[str]], Int['b nt']]
        ) = 'char_utf8',
        use_vocos = True,
        pretrained_vocos_path = 'charactr/vocos-mel-24khz',
        sampling_rate: int | None = None,
        frame_size: int = 320,
        velocity_consistency_weight = -1e-5,
        
        if_cond_proj_in = True,
        cond_proj_in_bias = True,
        if_embed_text = True,
        if_text_encoder2 = True,
        if_clip_encoder = False,
        video_encoder = "clip_vit"
    ):
        super().__init__()

        if isinstance(transformer, dict):
            transformer = Transformer(
                **transformer,
                cond_on_time = True
            )

        if isinstance(duration_predictor, dict):
            duration_predictor = DurationPredictor(**duration_predictor)

        self.transformer = transformer

        dim = transformer.dim
        dim_text = transformer.dim_text

        self.dim = dim
        self.dim_text = dim_text

        self.frac_lengths_mask = frac_lengths_mask
        self.audiocond_snr = audiocond_snr

        self.duration_predictor = duration_predictor

        # sampling

        self.odeint_kwargs = odeint_kwargs

        # mel spec

        self.mel_spec = default(mel_spec_module, None)
        num_channels = default(num_channels, None)
 
        self.num_channels = num_channels
        self.sampling_rate = default(sampling_rate, None)
        self.frame_size = frame_size

        # whether to concat condition and project rather than project both and sum

        self.concat_cond = concat_cond

        if concat_cond:
            self.proj_in = nn.Linear(num_channels * 2, dim)
        else:
            self.proj_in = nn.Linear(num_channels, dim)
            self.cond_proj_in = nn.Linear(num_channels, dim, bias=cond_proj_in_bias) if if_cond_proj_in else None

        # to prediction

        self.to_pred = Linear(dim, num_channels)

        # tokenizer and text embed

        if callable(tokenizer):
            assert exists(text_num_embeds), '`text_num_embeds` must be given if supplying your own tokenizer encode function'
            self.tokenizer = tokenizer
        elif tokenizer == 'char_utf8':
            text_num_embeds = 256
            self.tokenizer = list_str_to_tensor
        elif tokenizer == 'phoneme_en':
            self.tokenizer, text_num_embeds = get_g2p_en_encode()
        elif tokenizer == 'phoneme_zh':
            self.tokenizer, text_num_embeds = get_g2p_zh_encode()
        else:
            raise ValueError(f'unknown tokenizer string {tokenizer}')

        self.audiocond_drop_prob = audiocond_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.prompt_drop_prob = prompt_drop_prob

        # text embedding

        text_embed_klass = CharacterEmbed if not interpolated_text else InterpolatedCharacterEmbed

        self.embed_text = text_embed_klass(dim_text, num_embeds = text_num_embeds, **char_embed_kwargs) if if_embed_text else None

        # weight for velocity consistency

        self.register_buffer('zero', torch.tensor(0.), persistent = False)
        self.velocity_consistency_weight = velocity_consistency_weight

        # default vocos for mel -> audio

        if pretrained_vocos_path == 'charactr/vocos-mel-24khz':
            self.vocos = Vocos.from_pretrained(pretrained_vocos_path) if use_vocos else None
        elif pretrained_vocos_path == 'facebook/encodec_24khz':
            self.vocos = EncodecWrapper("facebook/encodec_24khz") if use_vocos else None
        elif pretrained_vocos_path == 'vae':
            self.vocos = VaeWrapper() if use_vocos else None

        if if_text_encoder2:
            self.tokenizer2 = AutoTokenizer.from_pretrained("/ailab-train/speech/zhanghaomin/models/flan-t5-large")
            self.text_encoder2 = T5EncoderModel.from_pretrained("/ailab-train/speech/zhanghaomin/models/flan-t5-large")
            for param in self.text_encoder2.parameters():
                param.requires_grad = False
            self.text_encoder2.eval()

        self.proj_text = None
        if if_clip_encoder:
            if video_encoder == "clip_vit":
                ####pass
                self.image_processor = CLIPImageProcessor()
                #self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("/ailab-train/speech/zhanghaomin/models/IP-Adapter/", subfolder="models/image_encoder")
                self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("/ailab-train/speech/zhanghaomin/models/IP-Adapter/", subfolder="sdxl_models/image_encoder")
            elif video_encoder == "clip_vit2":
                self.image_processor = AutoProcessor.from_pretrained("/ailab-train/speech/zhanghaomin/models/clip-vit-large-patch14-336/")
                self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("/ailab-train/speech/zhanghaomin/models/clip-vit-large-patch14-336/")
            elif video_encoder == "clip_convnext":
                self.image_encoder, _, self.image_processor = open_clip.create_model_and_transforms("hf-hub:laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup")
            elif video_encoder == "dinov2":
                self.image_processor = AutoImageProcessor.from_pretrained("/ailab-train/speech/zhanghaomin/models/dinov2-giant/")
                self.image_encoder = AutoModel.from_pretrained("/ailab-train/speech/zhanghaomin/models/dinov2-giant/")
            elif video_encoder == "mixed":
                pass
                #self.image_processor1 = CLIPImageProcessor()
                #self.image_encoder1 = CLIPVisionModelWithProjection.from_pretrained("/ailab-train/speech/zhanghaomin/models/IP-Adapter/", subfolder="sdxl_models/image_encoder")
                #self.image_processor2 = AutoProcessor.from_pretrained("/ailab-train/speech/zhanghaomin/models/clip-vit-large-patch14-336/")
                #self.image_encoder2 = CLIPVisionModelWithProjection.from_pretrained("/ailab-train/speech/zhanghaomin/models/clip-vit-large-patch14-336/")
                #self.image_encoder3, _, self.image_processor3 = open_clip.create_model_and_transforms("hf-hub:laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup")
                #self.image_processor4 = AutoImageProcessor.from_pretrained("/ailab-train/speech/zhanghaomin/models/dinov2-giant/")
                #self.image_encoder4 = AutoModel.from_pretrained("/ailab-train/speech/zhanghaomin/models/dinov2-giant/")
            else:
                self.image_processor = None
                self.image_encoder = None
            if video_encoder != "mixed":
                for param in self.image_encoder.parameters():
                    param.requires_grad = False
                self.image_encoder.eval()
                if dim_text != 1280:
                    self.dim_text_raw = 1280
                    self.proj_text = Linear(self.dim_text_raw, dim_text)
            else:
                #for param in self.image_encoder1.parameters():
                #    param.requires_grad = False
                #self.image_encoder1.eval()
                #for param in self.image_encoder2.parameters():
                #    param.requires_grad = False
                #self.image_encoder2.eval()
                #for param in self.image_encoder3.parameters():
                #    param.requires_grad = False
                #self.image_encoder3.eval()
                #for param in self.image_encoder4.parameters():
                #    param.requires_grad = False
                #self.image_encoder4.eval()
                self.dim_text_raw = 4608
                self.proj_text = Linear(self.dim_text_raw, dim_text)
        self.video_encoder = video_encoder

        for param in self.vocos.parameters():
            param.requires_grad = False
        self.vocos.eval()

    @property
    def device(self):
        return next(self.parameters()).device

    def encode_text(self, prompt):
        device = self.device
        batch = self.tokenizer2(prompt, max_length=self.tokenizer2.model_max_length, padding=True, truncation=True, return_tensors="pt")
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

        with torch.no_grad():
            encoder_hidden_states = self.text_encoder2(input_ids=input_ids, attention_mask=attention_mask)[0]

        boolean_encoder_mask = (attention_mask == 1).to(device)
        return encoder_hidden_states, boolean_encoder_mask

    def encode_video(self, video_paths, l):
        if self.proj_text is None:
            d = self.dim_text
        else:
            d = self.dim_text_raw
        device = self.device
        b = 20
        with torch.no_grad():
            video_embeddings = []
            video_lens = []
            for video_path in video_paths:
                if video_path is None:
                    video_embeddings.append(None)
                    video_lens.append(0)
                    continue
                if isinstance(video_path, tuple):
                    video_path, start_sample, max_sample = video_path
                else:
                    start_sample = 0
                    max_sample = None
                if video_path.startswith("/ailab-train/speech/zhanghaomin/VGGSound/"):
                    if self.video_encoder == "clip_vit":
                        feature_path = video_path.replace("/video/", "/feature/").replace(".mp4", ".npz")
                    elif self.video_encoder == "clip_vit2":
                        feature_path = video_path.replace("/video/", "/feature_clip_vit2/").replace(".mp4", ".npz")
                    elif self.video_encoder == "clip_convnext":
                        feature_path = video_path.replace("/video/", "/feature_clip_convnext/").replace(".mp4", ".npz")
                    elif self.video_encoder == "dinov2":
                        feature_path = video_path.replace("/video/", "/feature_dinov2/").replace(".mp4", ".npz")
                    elif self.video_encoder == "mixed":
                        feature_path = video_path.replace("/video/", "/feature_mixed/").replace(".mp4", ".npz")
                    else:
                        raise Exception("Invalid video_encoder " + self.video_encoder)
                else:
                    if self.video_encoder == "clip_vit":
                        feature_path = video_path.replace(".mp4", ".generated.npz")
                    elif self.video_encoder == "clip_vit2":
                        feature_path = video_path.replace(".mp4", ".generated.clip_vit2.npz")
                    elif self.video_encoder == "clip_convnext":
                        feature_path = video_path.replace(".mp4", ".generated.clip_convnext.npz")
                    elif self.video_encoder == "dinov2":
                        feature_path = video_path.replace(".mp4", ".generated.dinov2.npz")
                    elif self.video_encoder == "mixed":
                        feature_path = video_path.replace(".mp4", ".generated.mixed.npz")
                    else:
                        raise Exception("Invalid video_encoder " + self.video_encoder)
                
                if not os.path.exists(feature_path):
                    #print("video not exist", video_path)
                    frames, duration, fps = read_frames_with_moviepy(video_path, max_frame_nums=None)
                    if frames is None:
                        video_embeddings.append(None)
                        video_lens.append(0)
                        continue
                    ####low fps####
                    ##frames = frames[0::3,...]
                    #ds = max(round(fps/8.0), 1)
                    #print("fps", fps, duration, frames.shape, ds)
                    #frames = frames[0::ds,...]
                    ####low fps####
                    if self.video_encoder in ["clip_vit", "clip_vit2", "dinov2"]:
                        images = self.image_processor(images=frames, return_tensors="pt").to(device)
                        #print("images", images["pixel_values"].shape, images["pixel_values"].max(), images["pixel_values"].min(), torch.abs(images["pixel_values"]).mean())
                    elif self.video_encoder in ["clip_convnext"]:
                        images = []
                        for i in range(frames.shape[0]):
                            images.append(self.image_processor(Image.fromarray(frames[i])).unsqueeze(0))
                        images = torch.cat(images, dim=0).to(device)
                        #print("images", images.shape, images.max(), images.min(), torch.abs(images).mean())
                    elif self.video_encoder in ["mixed"]:
                        #images1 = self.image_processor1(images=frames, return_tensors="pt").to(device)
                        images2 = self.image_processor2(images=frames, return_tensors="pt").to(device)
                        images4 = self.image_processor4(images=frames, return_tensors="pt").to(device)
                        images3 = []
                        for i in range(frames.shape[0]):
                            images3.append(self.image_processor3(Image.fromarray(frames[i])).unsqueeze(0))
                        images3 = torch.cat(images3, dim=0).to(device)
                    else:
                        raise Exception("Invalid video_encoder " + self.video_encoder)
                    image_embeddings = []
                    if self.video_encoder == "clip_vit":
                        for i in range(math.ceil(images["pixel_values"].shape[0] / b)):
                            image_embeddings.append(self.image_encoder(pixel_values=images["pixel_values"][i*b: (i+1)*b]).image_embeds.cpu())
                    elif self.video_encoder == "clip_vit2":
                        for i in range(math.ceil(images["pixel_values"].shape[0] / b)):
                            image_embeddings.append(self.image_encoder(pixel_values=images["pixel_values"][i*b: (i+1)*b]).image_embeds.cpu())
                    elif self.video_encoder == "clip_convnext":
                        for i in range(math.ceil(images.shape[0] / b)):
                            image_embeddings.append(self.image_encoder.encode_image(images[i*b: (i+1)*b]).cpu())
                    elif self.video_encoder == "dinov2":
                        for i in range(math.ceil(images["pixel_values"].shape[0] / b)):
                            image_embeddings.append(self.image_encoder(pixel_values=images["pixel_values"][i*b: (i+1)*b]).pooler_output.cpu())
                    elif self.video_encoder == "mixed":
                        feature_path1 = feature_path.replace("/feature_mixed/", "/feature/")
                        if not os.path.exists(feature_path1):
                            image_embeddings1 = []
                            for i in range(math.ceil(images1["pixel_values"].shape[0] / b)):
                                image_embeddings1.append(self.image_encoder1(pixel_values=images1["pixel_values"][i*b: (i+1)*b]).image_embeds.cpu())
                            image_embeddings1 = torch.cat(image_embeddings1, dim=0)
                            #np.savez(feature_path1, image_embeddings1, duration)
                        else:
                            data1 = np.load(feature_path1)
                            image_embeddings1 = torch.from_numpy(data1["arr_0"])
                        feature_path2 = feature_path.replace("/feature_mixed/", "/feature_clip_vit2/")
                        if not os.path.exists(feature_path2):
                            image_embeddings2 = []
                            for i in range(math.ceil(images2["pixel_values"].shape[0] / b)):
                                image_embeddings2.append(self.image_encoder2(pixel_values=images2["pixel_values"][i*b: (i+1)*b]).image_embeds.cpu())
                            image_embeddings2 = torch.cat(image_embeddings2, dim=0)
                            np.savez(feature_path2, image_embeddings2, duration)
                        else:
                            data2 = np.load(feature_path2)
                            image_embeddings2 = torch.from_numpy(data2["arr_0"])
                        feature_path3 = feature_path.replace("/feature_mixed/", "/feature_clip_convnext/")
                        if not os.path.exists(feature_path3):
                            image_embeddings3 = []
                            for i in range(math.ceil(images3.shape[0] / b)):
                                image_embeddings3.append(self.image_encoder3.encode_image(images3[i*b: (i+1)*b]).cpu())
                            image_embeddings3 = torch.cat(image_embeddings3, dim=0)
                            np.savez(feature_path3, image_embeddings3, duration)
                        else:
                            data3 = np.load(feature_path3)
                            image_embeddings3 = torch.from_numpy(data3["arr_0"])
                        feature_path4 = feature_path.replace("/feature_mixed/", "/feature_dinov2/")
                        if not os.path.exists(feature_path4):
                            image_embeddings4 = []
                            for i in range(math.ceil(images4["pixel_values"].shape[0] / b)):
                                image_embeddings4.append(self.image_encoder4(pixel_values=images4["pixel_values"][i*b: (i+1)*b]).pooler_output.cpu())
                            image_embeddings4 = torch.cat(image_embeddings4, dim=0)
                            np.savez(feature_path4, image_embeddings4, duration)
                        else:
                            data4 = np.load(feature_path4)
                            image_embeddings4 = torch.from_numpy(data4["arr_0"])
                        mixed_l = min([image_embeddings1.shape[0], image_embeddings2.shape[0], image_embeddings3.shape[0], image_embeddings4.shape[0]])
                        for i in range(mixed_l):
                            image_embeddings.append(torch.cat([image_embeddings1[i:i+1,:], image_embeddings2[i:i+1,:], image_embeddings3[i:i+1,:], image_embeddings4[i:i+1,:]], dim=1))
                    else:
                        raise Exception("Invalid video_encoder " + self.video_encoder)
                    image_embeddings = torch.cat(image_embeddings, dim=0)
                    #print("image_embeddings", image_embeddings.shape, image_embeddings.max(), image_embeddings.min(), torch.abs(image_embeddings).mean())
                    ####low fps####
                    #np.savez(feature_path, image_embeddings, duration)
                    np.savez(feature_path, image_embeddings, duration, fps)
                    ####low fps####
                else:
                    #print("video exist", feature_path)
                    data = np.load(feature_path)
                    image_embeddings = torch.from_numpy(data["arr_0"])
                    #print("image_embeddings", image_embeddings.shape, image_embeddings.max(), image_embeddings.min(), torch.abs(image_embeddings).mean())
                    duration = data["arr_1"].item()
                    ####low fps####
                    if "arr_2" not in data:
                        fps = float(image_embeddings.shape[0]) / duration
                        ##image_embeddings = image_embeddings[0::3,...]
                        #ds = max(round(fps/8.0), 1)
                        ##print("fps", fps, duration, image_embeddings.shape, ds)
                        #image_embeddings = image_embeddings[0::ds,...]
                    else:
                        fps = data["arr_2"].item()
                    ####low fps####
                if max_sample is None:
                    max_sample = int(duration * self.sampling_rate)
                interpolated = []
                for i in range(start_sample, max_sample, self.frame_size):
                    j = min(round((i+self.frame_size//2) / self.sampling_rate / (duration / (image_embeddings.shape[0] - 1))), image_embeddings.shape[0] - 1)
                    interpolated.append(image_embeddings[j:j+1])
                    if len(interpolated) >= l:
                        break
                interpolated = torch.cat(interpolated, dim=0)
                #ll = list(range(start_sample, max_sample, self.frame_size))
                #print("encode_video l", len(ll), l, round((ll[-1]+self.frame_size//2) / self.sampling_rate / (duration / (image_embeddings.shape[0] - 1))), image_embeddings.shape[0] - 1)
                #print("encode_video one", video_path, duration, image_embeddings.shape, interpolated.shape, l)
                video_embeddings.append(interpolated.unsqueeze(0))
                video_lens.append(interpolated.shape[1])
            max_length = max(video_lens)
            if max_length == 0:
                max_length = l
            else:
                max_length = l
            for i in range(len(video_embeddings)):
                if video_embeddings[i] is None:
                    video_embeddings[i] = torch.zeros(1, max_length, d)
                    continue
                if video_embeddings[i].shape[1] < max_length:
                    video_embeddings[i] = torch.cat([video_embeddings[i], torch.zeros(1, max_length-video_embeddings[i].shape[1], d)], 1)
            video_embeddings = torch.cat(video_embeddings, 0)
        #print("encode_video", l, video_embeddings.shape, video_lens)
        return video_embeddings.to(device)

    def transformer_with_pred_head(
        self,
        x: Float['b n d'],
        cond: Float['b n d'] | None = None,
        times: Float['b'] | None = None,
        mask: Bool['b n'] | None = None,
        text: Int['b nt'] | Float['b nt dt'] | None = None,
        prompt = None,
        video_drop_prompt = None,
        audio_drop_prompt = None,
        drop_audio_cond: bool | None = None,
        drop_text_cond: bool | None = None,
        drop_text_prompt: bool | None = None,
        return_drop_conditions = False
    ):
        seq_len = x.shape[-2]
        bs = x.shape[0]
        drop_audio_cond = [default(drop_audio_cond, self.training and random() < self.audiocond_drop_prob) for _ in range(bs)]
        drop_text_cond = default(drop_text_cond, self.training and random() < self.cond_drop_prob)
        drop_text_prompt = [default(drop_text_prompt, self.training and random() < self.prompt_drop_prob) for _ in range(bs)]

        if cond is not None:
            for b in range(bs):
                if drop_audio_cond[b]:
                    cond[b] = 0
                if audio_drop_prompt is not None and audio_drop_prompt[b]:
                    cond[b] = 0

        if cond is not None:
            if self.concat_cond:
                # concat condition, given as using voicebox-like scheme
                x = torch.cat((cond, x), dim = -1)

        x = self.proj_in(x)

        if cond is not None:
            if not self.concat_cond:
                # an alternative is to simply sum the condition
                # seems to work fine

                cond = self.cond_proj_in(cond)
                x = x + cond

        # whether to use a text embedding

        text_embed = None
        if exists(text) and len(text.shape) == 3:
            text_embed = text.clone()
            if drop_text_cond:
                for b in range(bs):
                    text_embed[b] = 0
        elif exists(text) and not drop_text_cond:
            text_embed = self.embed_text(text, seq_len, mask = mask)

        context, context_mask = None, None
        if prompt is not None:
            #for b in range(bs):
            #    if drop_text_prompt[b]:
            #        prompt[b] = ""
            if video_drop_prompt is not None:
                for b in range(bs):
                    if video_drop_prompt[b]:
                        prompt[b] = "the sound of X X"
            context, context_mask = self.encode_text(prompt)
            for b in range(bs):
                if drop_text_prompt[b]:
                    context[b] = 0
                if video_drop_prompt is not None and video_drop_prompt[b]:
                    context[b] = 0
        #print("cross attention", context.shape, context_mask.shape, x.shape, mask.shape, text_embed.shape if text_embed is not None else None, torch.mean(torch.abs(text_embed), dim=(1,2)))
        #print("video_drop_prompt", prompt, video_drop_prompt, context.shape, torch.mean(torch.abs(context), dim=(1,2)))
        #print("audio_drop_prompt", audio_drop_prompt, cond.shape, torch.mean(torch.abs(cond), dim=(1,2)))

        if self.proj_text is not None:
            text_embed = self.proj_text(text_embed)

        # attend

        attended = self.transformer(
            x,
            times = times,
            mask = mask,
            text_embed = text_embed,
            context = context,
            context_mask = context_mask
        )

        pred =  self.to_pred(attended)

        if not return_drop_conditions:
            return pred

        return pred, drop_audio_cond, drop_text_cond, drop_text_prompt

    def cfg_transformer_with_pred_head(
        self,
        *args,
        cfg_strength: float = 1.,
        remove_parallel_component: bool = True,
        keep_parallel_frac: float = 0.,
        **kwargs,
    ):
        
        pred = self.transformer_with_pred_head(*args, drop_audio_cond = False, drop_text_cond = False, drop_text_prompt = False, **kwargs)

        if cfg_strength < 1e-5:
            return pred

        null_pred = self.transformer_with_pred_head(*args, drop_audio_cond = True, drop_text_cond = True, drop_text_prompt = True, **kwargs)

        cfg_update = pred - null_pred

        if remove_parallel_component:
            # https://arxiv.org/abs/2410.02416
            parallel, orthogonal = project(cfg_update, pred)
            cfg_update = orthogonal + parallel * keep_parallel_frac

        return pred + cfg_update * cfg_strength

    def add_noise(self, signal, mask, val):
        if self.audiocond_snr is None:
            return signal
        if not val:
            snr = np.random.uniform(self.audiocond_snr[0], self.audiocond_snr[1])
        else:
            snr = (self.audiocond_snr[0] + self.audiocond_snr[1]) / 2.0
        #print("add_noise", self.audiocond_snr, snr, signal.shape, mask)  # [True, ..., False]
        noise = torch.randn_like(signal)
        w = torch.abs(signal[mask]).mean() / (torch.abs(noise[mask]).mean() + 1e-6) / snr
        return signal + noise * w

    @torch.no_grad()
    def sample(
        self,
        cond: Float['b n d'] | Float['b nw'] | None = None,
        *,
        text: Int['b nt'] | list[str] | None = None,
        lens: Int['b'] | None = None,
        duration: int | Int['b'] | None = None,
        steps = 32,
        cfg_strength = 1.,   # they used a classifier free guidance strength of 1.
        remove_parallel_component = True,
        sway_sampling = True,
        max_duration = 4096, # in case the duration predictor goes haywire
        vocoder: Callable[[Float['b d n']], list[Float['_']]] | None = None,
        return_raw_output: bool | None = None,
        save_to_filename: str | None = None,
        prompt = None,
        video_drop_prompt = None,
        audio_drop_prompt = None,
        video_paths = None
    ) -> (
        Float['b n d'],
        list[Float['_']]
    ):
        self.eval()

        # raw wave

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = rearrange(cond, 'b d n -> b n d')
            assert cond.shape[-1] == self.num_channels

        batch, cond_seq_len, device = *cond.shape[:2], cond.device

        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device = device, dtype = torch.long)

        if video_paths is not None:
            if len(video_paths) == batch:
                text = self.encode_video(video_paths, cond_seq_len)
            else:
                text = self.encode_video(video_paths, cond_seq_len*batch)
                print("clip frames", text.shape, cond.shape)
                text = text.reshape(batch, -1, text.shape[-1])
        # text
        elif isinstance(text, list):
            text = self.tokenizer(text).to(device)
            assert text.shape[0] == batch
            
            if exists(text):
                text_lens = (text != -1).sum(dim = -1)
                lens = torch.maximum(text_lens, lens) # make sure lengths are at least those of the text characters

        # duration

        cond_mask = lens_to_mask(lens)

        if exists(duration):
            if isinstance(duration, int):
                duration = torch.full((batch,), duration, device = device, dtype = torch.long)

        elif exists(self.duration_predictor):
            duration = self.duration_predictor(cond, text = text, lens = lens, return_loss = False).long()

        duration = torch.maximum(lens, duration) # just add one token so something is generated
        duration = duration.clamp(max = max_duration)

        assert duration.shape[0] == batch

        max_duration = duration.amax()

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value = 0.)
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value = False)
        cond_mask = rearrange(cond_mask, '... -> ... 1')

        mask = lens_to_mask(duration)
        #print("####", duration, mask, mask.shape, lens, cond_mask, cond_mask.shape, text)

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed

            if lens[0] == duration[0]:
                print("No cond", lens, duration)
                step_cond = None
            else:
                step_cond = torch.where(cond_mask, self.add_noise(cond, cond_mask, True), torch.zeros_like(cond))

            # predict flow

            return self.cfg_transformer_with_pred_head(
                x,
                step_cond,
                times = t,
                text = text,
                mask = mask,
                prompt = prompt,
                video_drop_prompt = video_drop_prompt,
                audio_drop_prompt = audio_drop_prompt,
                cfg_strength = cfg_strength,
                remove_parallel_component = remove_parallel_component
            )

        ####torch.manual_seed(0)
        y0 = torch.randn_like(cond)
        t = torch.linspace(0, 1, steps, device = self.device)
        if sway_sampling:
            t = t + -1.0 * (torch.cos(torch.pi / 2 * t) - 1 + t)
        #print("@@@@", t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        sampled = trajectory[-1]

        out = sampled

        if lens[0] != duration[0]:
            out = torch.where(cond_mask, cond, out)

        # able to return raw untransformed output, if not using mel rep

        if exists(return_raw_output) and return_raw_output:
            return out

        # take care of transforming mel to audio if `vocoder` is passed in, or if `use_vocos` is turned on

        if exists(vocoder):
            assert not exists(self.vocos), '`use_vocos` should not be turned on if you are passing in a custom `vocoder` on sampling'
            out = rearrange(out, 'b n d -> b d n')
            out = vocoder(out)

        elif exists(self.vocos):

            audio = []
            for mel, one_mask in zip(out, mask):
                #one_out = DB_to_amplitude(mel[one_mask], ref = 1., power = 0.5)
                one_out = mel[one_mask]

                one_out = rearrange(one_out, 'n d -> 1 d n')
                one_audio = self.vocos.decode(one_out)
                one_audio = rearrange(one_audio, '1 nw -> nw')
                audio.append(one_audio)

            out = audio

        if exists(save_to_filename):
            assert exists(vocoder) or exists(self.vocos)
            assert exists(self.sampling_rate)

            path = Path(save_to_filename)
            parent_path = path.parents[0]
            parent_path.mkdir(exist_ok = True, parents = True)

            for ind, one_audio in enumerate(out):
                one_audio = rearrange(one_audio, 'nw -> 1 nw')
                if len(out) == 1:
                    save_path = str(parent_path / f'{path.name}')
                else:
                    save_path = str(parent_path / f'{ind + 1}.{path.name}')
                torchaudio.save(save_path, one_audio.detach().cpu(), sample_rate = self.sampling_rate)

        return out

    def forward(
        self,
        inp: Float['b n d'] | Float['b nw'], # mel or raw wave
        *,
        text: Int['b nt'] | list[str] | None = None,
        times: int | Int['b'] | None = None,
        lens: Int['b'] | None = None,
        velocity_consistency_model: E2TTS | None = None,
        velocity_consistency_delta = 1e-3,
        prompt = None,
        video_drop_prompt=None,
        audio_drop_prompt=None,
        val = False,
        video_paths=None
    ):
        need_velocity_loss = exists(velocity_consistency_model) and self.velocity_consistency_weight > 0.

        # handle raw wave

        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = rearrange(inp, 'b d n -> b n d')
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device

        if video_paths is not None:
            text = self.encode_video(video_paths, seq_len)
        # handle text as string
        elif isinstance(text, list):
            text = self.tokenizer(text).to(device)
            #print("text tokenized", text[0])
            assert text.shape[0] == batch

        # lens and mask

        if not exists(lens):
            lens = torch.full((batch,), seq_len, device = device)

        mask = lens_to_mask(lens, length = seq_len)

        # get a random span to mask out for training conditionally

        if not val:
            if self.audiocond_drop_prob > 1.0:
                frac_lengths = torch.zeros((batch,), device = self.device).float().uniform_(1.0,1.0)
            else:
                frac_lengths = torch.zeros((batch,), device = self.device).float().uniform_(*self.frac_lengths_mask)
        else:
            frac_lengths = torch.tensor([(0.7+1.0)/2.0]*batch, device = self.device).float()
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths, max_length = seq_len, val = val)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1

        x1 = inp

        # main conditional flow training logic
        # just ~5 loc

        # x0 is gaussian noise

        if val:
            torch.manual_seed(0)
        x0 = torch.randn_like(x1)
        if val:
            torch.manual_seed(int(time.time()*1000))

        # t is random times from above

        if times is None:
            times = torch.rand((batch,), dtype = dtype, device = self.device)
        else:
            times = torch.tensor((times,)*batch, dtype = dtype, device = self.device)
        t = rearrange(times, 'b -> b 1 1')

        # if need velocity consistency, make sure time does not exceed 1.

        if need_velocity_loss:
            t = t * (1. - velocity_consistency_delta)

        # sample xt (w in the paper)

        w = (1. - t) * x0 + t * x1

        flow = x1 - x0

        # only predict what is within the random mask span for infilling

        if self.audiocond_drop_prob > 1.0:
            cond = None
        else:
            cond = einx.where(
                'b n, b n d, b n d -> b n d',
                rand_span_mask,
                torch.zeros_like(x1), self.add_noise(x1, ~rand_span_mask, val)
            )

        # transformer and prediction head

        if not val:
            pred, did_drop_audio_cond, did_drop_text_cond, did_drop_text_prompt = self.transformer_with_pred_head(
                w,
                cond,
                times = times,
                text = text,
                mask = mask,
                prompt = prompt,
                video_drop_prompt = video_drop_prompt,
                audio_drop_prompt = audio_drop_prompt,
                return_drop_conditions = True
            )
        else:
            pred, did_drop_audio_cond, did_drop_text_cond, did_drop_text_prompt = self.transformer_with_pred_head(
                w,
                cond,
                times = times,
                text = text,
                mask = mask,
                prompt = prompt,
                video_drop_prompt = video_drop_prompt,
                audio_drop_prompt = audio_drop_prompt,
                drop_audio_cond = False,
                drop_text_cond = False,
                drop_text_prompt = False,
                return_drop_conditions = True
            )

        # maybe velocity consistency loss

        velocity_loss = self.zero

        if need_velocity_loss:

            t_with_delta = t + velocity_consistency_delta
            w_with_delta = (1. - t_with_delta) * x0 + t_with_delta * x1

            with torch.no_grad():
                ema_pred = velocity_consistency_model.transformer_with_pred_head(
                    w_with_delta,
                    cond,
                    times = times + velocity_consistency_delta,
                    text = text,
                    mask = mask,
                    prompt = prompt,
                    video_drop_prompt = video_drop_prompt,
                    audio_drop_prompt = audio_drop_prompt,
                    drop_audio_cond = did_drop_audio_cond,
                    drop_text_cond = did_drop_text_cond,
                    drop_text_prompt = did_drop_text_prompt
                )

            velocity_loss = F.mse_loss(pred, ema_pred, reduction = 'none')
            velocity_loss = velocity_loss[rand_span_mask].mean()

        # flow matching loss

        loss = F.mse_loss(pred, flow, reduction = 'none')

        #print("loss", loss.shape, loss, "rand_span_mask", rand_span_mask.shape, rand_span_mask, "loss[rand_span_mask]", loss[rand_span_mask].shape, loss[rand_span_mask])
        loss = loss[rand_span_mask].mean()

        # total loss and get breakdown

        total_loss = loss + velocity_loss * self.velocity_consistency_weight
        breakdown = LossBreakdown(loss, velocity_loss)
        #print("loss", loss, velocity_loss, self.velocity_consistency_weight)

        # return total loss and bunch of intermediates

        return E2TTSReturn(total_loss, cond if cond is not None else w, pred, x0 + pred, breakdown)
