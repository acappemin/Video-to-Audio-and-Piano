import torch
from e2_tts_pytorch.e2_tts_crossatt3 import E2TTS, DurationPredictor
from e2_tts_pytorch.e2_tts_crossatt3 import MelSpec, EncodecWrapper

from torch.optim import Adam
from datasets import load_dataset

#from e2_tts_pytorch.trainer import (
from e2_tts_pytorch.trainer_multigpus_alldatas3 import (
    HFDataset,
    Text2AudioDataset,
    Text2SpeechDataset,
    E2Trainer,
)

import json
import numpy as np


audiocond_drop_prob = 1.1
#audiocond_drop_prob = -0.1
#cond_proj_in_bias = True
#cond_drop_prob = 1.1
cond_drop_prob = -0.1
prompt_drop_prob = -0.1
#prompt_drop_prob = 1.1
video_text = True


def main():
    #duration_predictor = DurationPredictor(
    #    transformer = dict(
    #        dim = 512,
    #        depth = 6,
    #    )
    #)
    duration_predictor = None

    e2tts = E2TTS(
        duration_predictor = duration_predictor,
        transformer = dict(
            #depth = 12,
            #dim = 512,
            #heads = 8,
            #dim_head = 64,
            depth = 12,
            dim = 1024,
            dim_text = 1280,
            dim_frames = 512,
            heads = 16,
            dim_head = 64,
            if_text_modules = (cond_drop_prob < 1.0),
            if_cross_attn = (prompt_drop_prob < 1.0),
            if_audio_conv = True,
            if_text_conv = True,
        ),
        #tokenizer = 'char_utf8',
        tokenizer = 'phoneme_zh',
        audiocond_drop_prob = audiocond_drop_prob,
        cond_drop_prob = cond_drop_prob,
        prompt_drop_prob = prompt_drop_prob,
        frac_lengths_mask = (0.7, 1.0),
        #audiocond_snr = None,
        #audiocond_snr = (5.0, 10.0),
        
        if_cond_proj_in = (audiocond_drop_prob < 1.0),
        #cond_proj_in_bias = cond_proj_in_bias,
        if_embed_text = (cond_drop_prob < 1.0) and (not video_text),
        if_text_encoder2 = (prompt_drop_prob < 1.0),
        if_clip_encoder = video_text,
        video_encoder = "clip_vit",
        
        pretrained_vocos_path = 'facebook/encodec_24khz',
        num_channels = 128,
        sampling_rate = 24000,
    )
    
    if True:
        video2roll_path = './audeo/models/Video2Roll_50_0.4/14.pth'
        video2roll_state_dict = torch.load(video2roll_path)
        e2tts.video2roll_net.load_state_dict(video2roll_state_dict)
    
    ####
    if False:
        checkpoint = torch.load('/ailab-train/speech/zhanghaomin/e2/e2_tts_experiment_v2a_encodec_more_more_more_more/80000.pt', map_location='cpu')
        model0 = {}
        for key in list(checkpoint['model_state_dict'].keys()):
            model0[key] = checkpoint['model_state_dict'][key].shape
        model1 = {}
        for key, param in e2tts.state_dict().items():
            model1[key] = param.shape
        #print("model0", model0)
        #print("model1", model1)
        for key in model0.keys():
            if key not in model1:
                pass
                #print("Missing key found", key, model0[key])
            else:
                if model0[key] != model1[key]:
                    pass
                    #print("Miss match", key, model0[key], model1[key])
        freeze_keys = []
        for key in model1.keys():
            if key not in model0:
                pass
                #print("New key found", key, model1[key])
            else:
                freeze_keys.append(key)
                if model0[key] != model1[key]:
                    pass
                    #print("Miss match", key, model0[key], model1[key])
        
        for name, param in e2tts.named_parameters():
            if name in freeze_keys:
                #if name.startswith("transformer."):
                #    print("FREEZE", name)
                param.requires_grad = False

    if False:
        model1 = {}
        for key, param in e2tts.state_dict().items():
            model1[key] = param.shape
        freeze_keys = []
        for key in model1.keys():
            if key.startswith("video2roll_net."):
                freeze_keys.append(key)
        for name, param in e2tts.named_parameters():
            if name in freeze_keys:
                param.requires_grad = False

    #optimizer = Adam(e2tts.parameters(), lr=7.5e-5)
    ####optimizer = Adam(e2tts.parameters(), lr=3e-5)
    ####optimizer = Adam(e2tts.parameters(), lr=1.5e-5)
    optimizer = Adam(e2tts.parameters(), lr=1e-6)

    trainer = E2Trainer(
        e2tts,
        optimizer,
        #num_warmup_steps=20000,
        #num_warmup_steps=10000,
        num_warmup_steps=100,
        ####grad_accumulation_steps = 2,
        grad_accumulation_steps = 4,
        #checkpoint_path = '/ailab-train/speech/zhanghaomin/e2/e2_tts_experiment_audio_encodec_selfcrossattn2_maxeng_rptaug2_more_more_more/200000.pt',
        #log_file = '/ailab-train/speech/zhanghaomin/e2/e2_tts_experiment_v2a_encodec/e2tts.txt',
        #tensorboard_log_dir = '/ailab-train/speech/zhanghaomin/e2/e2_tts_experiment_v2a_encodec/',
        ####checkpoint_path = '/ailab-train/speech/zhanghaomin/e2/e2_tts_experiment_v2a_encodec_more_more_more_more/80000.pt',
        checkpoint_path = '/ailab-train/speech/zhanghaomin/e2/e2_tts_experiment_v2a_encodec_more_more_more_more_piano5/4_2_8000.pt',
        log_file = '/ailab-train/speech/zhanghaomin/e2/e2_tts_experiment_v2a_encodec_more_more_more_more_piano5/e2tts.txt',
        tensorboard_log_dir = '/ailab-train/speech/zhanghaomin/e2/e2_tts_experiment_v2a_encodec_more_more_more_more_piano5/',
        max_grad_norm = 0.2,
        ema_kwargs = {"power": 0.75},
        if_text = (cond_drop_prob < 1.0) and (not video_text),
        if_prompt = (prompt_drop_prob < 1.0),
    )

    epochs = 20
    #batch_size = 4
    ####batch_size = 2
    batch_size = 1
    batch_size_val = 16
    batch_size_val2 = 1

    #train_dataset = HFDataset(load_dataset("MushanW/GLOBE")["train"])
    #train_dataset = HFDataset(load_dataset("parquet", data_files={"train": "/ckptstorage/zhanghaomin/tts/GLOBE/data/train-*.parquet"})["train"])
    SCORE_THRESHOLD_TRAIN = '{"/zhanghaomin/datas/audiocaps": -9999.0, "/radiostorage/WavCaps": -9999.0, "/radiostorage/AudioGroup": 9999.0, "/ckptstorage/zhanghaomin/audioset": -9999.0, "/ckptstorage/zhanghaomin/BBCSoundEffects": 9999.0, "/ckptstorage/zhanghaomin/CLAP_freesound": 9999.0, "/zhanghaomin/datas/musiccap": -9999.0, "/ckptstorage/zhanghaomin/TangoPromptBank": -9999.0, "audioset": "af-audioset", "/ckptstorage/zhanghaomin/audiosetsl": 9999.0, "/ckptstorage/zhanghaomin/giantsoundeffects": -9999.0}'  # /root/datasets/ /radiostorage/
    SCORE_THRESHOLD_TRAIN = json.loads(SCORE_THRESHOLD_TRAIN)
    for key in SCORE_THRESHOLD_TRAIN:
        if key == "audioset":
            continue
        if SCORE_THRESHOLD_TRAIN[key] <= -9000.0:
            SCORE_THRESHOLD_TRAIN[key] = -np.inf
    print("SCORE_THRESHOLD_TRAIN", SCORE_THRESHOLD_TRAIN)
    #stft = MelSpec(sampling_rate=24000)
    stft = EncodecWrapper("facebook/encodec_24khz")
    ####train_dataset = Text2AudioDataset(None, "train_val_audioset_sl", None, None, None, -1, batch_size, stft, 0, trainer.is_main, SCORE_THRESHOLD_TRAIN, "/zhanghaomin/codes2/audiocaption/msclapcap_v1.list", -1.0, 4, [False]*8, None, trainer.device_id, None, e2tts.video_encoder, None, trainer.num_processes)
    train_dataset = Text2AudioDataset(None, "train_val_audioset_sl", None, None, None, -1, batch_size, stft, 0, trainer.is_main, SCORE_THRESHOLD_TRAIN, "/zhanghaomin/codes2/audiocaption/msclapcap_v1.list", -1.0, 1, 2, [False]*1+[True]*3, None, trainer.device_id, None, e2tts.video_encoder, None, trainer.num_processes)
    eval1_dataset = None  #Text2AudioDataset(None, "val_boom_epic", None, None, None, -1, -1, stft, 0, trainer.is_main, SCORE_THRESHOLD_TRAIN, "/zhanghaomin/codes2/audiocaption/msclapcap_v1.list", -1.0, 1, 1, None, None, trainer.device_id, None, e2tts.video_encoder, None, trainer.num_processes)
    eval2_dataset = Text2AudioDataset(None, "val_audiocaps", None, None, None, -1, -1, stft, 0, trainer.is_main, SCORE_THRESHOLD_TRAIN, "/zhanghaomin/codes2/audiocaption/msclapcap_v1.list", -1.0, 1, 1, None, None, trainer.device_id, None, e2tts.video_encoder, None, trainer.num_processes)
    eval3_dataset = Text2AudioDataset(None, "val_vggsound", None, None, None, -1, -1, stft, 0, trainer.is_main, SCORE_THRESHOLD_TRAIN, "/zhanghaomin/codes2/audiocaption/msclapcap_v1.list", -1.0, 1, 1, [False]*batch_size_val, None, trainer.device_id, None, e2tts.video_encoder, None, trainer.num_processes)
    eval4_dataset = None #Text2AudioDataset(None, "val_vggsound", None, None, None, -1, -1, stft, 0, trainer.is_main, SCORE_THRESHOLD_TRAIN, "/zhanghaomin/codes2/audiocaption/msclapcap_v1.list", -1.0, 1, 1, [True]*batch_size_val, None, trainer.device_id, None, e2tts.video_encoder, None, trainer.num_processes)
    eval5_dataset = Text2AudioDataset(None, "val_instruments", None, None, None, -1, -1, stft, 0, trainer.is_main, SCORE_THRESHOLD_TRAIN, "/zhanghaomin/codes2/audiocaption/msclapcap_v1.list", -1.0, 1, 1, [False]*batch_size_val2, None, trainer.device_id, None, e2tts.video_encoder, None, trainer.num_processes)
    eval6_dataset = None #Text2AudioDataset(None, "val_instruments", None, None, None, -1, -1, stft, 0, trainer.is_main, SCORE_THRESHOLD_TRAIN, "/zhanghaomin/codes2/audiocaption/msclapcap_v1.list", -1.0, 1, 1, [True]*batch_size_val2, None, trainer.device_id, None, e2tts.video_encoder, None, trainer.num_processes)

    eval1_dataset, eval2_dataset, eval3_dataset, eval4_dataset, eval6_dataset = None, None, None, None, None
    trainer.train((train_dataset, eval1_dataset, eval2_dataset, eval3_dataset, eval4_dataset, eval5_dataset, eval6_dataset), epochs, batch_size, num_workers=12, save_step=50)

if __name__ == "__main__":
    main()

