import sys

if len(sys.argv) >= 6:
    ckpt = sys.argv[1]
    drop_prompt = bool(int(sys.argv[2]))
    test_scp = sys.argv[3]
    start = int(sys.argv[4])
    end = int(sys.argv[5])
    step = 1
    out_dir = sys.argv[6]
    print("inference", ckpt, drop_prompt, test_scp, start, end, out_dir)
else:
    #ckpt = "/ailab-train/speech/zhanghaomin/e2/e2_tts_experiment_v2a_encodec_more_more/98500.pt"
    #ckpt = "/ailab-train/speech/zhanghaomin/e2/e2_tts_experiment_v2a_encodec_more_more_more/190000.pt"
    #ckpt = "/ailab-train/speech/zhanghaomin/e2/e2_tts_experiment_v2a_encodec_more_more_more/315000.pt"
    #ckpt = "/ailab-train/speech/zhanghaomin/e2/e2_tts_experiment_v2a_encodec_more_more_more_more/60000.pt"
    #ckpt = "/ailab-train/speech/zhanghaomin/e2/e2_tts_experiment_v2a_encodec_more_more_more_more_piano5/4_2_8000.pt"
    ckpt = "./ckpts/piano5_4_2_8000.pt"
    #ckpt = "/ailab-train/speech/zhanghaomin/e2/e2_tts_experiment_v2a_encodec_more_more_more_more_piano6/dpo_100.pt"
    drop_prompt = False
    test_scp = "/ailab-train/speech/zhanghaomin/scps/VGGSound/test.scp"
    #test_scp = "./tests/vgg_test.scp"
    ####test_scp = "/ailab-train/speech/zhanghaomin/scps/instruments/test.scp"
    ####test_scp = "/ailab-train/speech/zhanghaomin/scps/instruments/piano_2h/test.scp"
    ####test_scp = "/ailab-train/speech/zhanghaomin/scps/instruments/piano_20h/v2a_giant_piano2/test.scp"
    start = 0
    end = 2
    step = 1
    out_dir = "./outputs/outputs_vgg/"
    ####out_dir = "./outputs_piano/"
    #####out_dir = "./outputs2t_20h_dpo/"


import torch
from e2_tts_pytorch.e2_tts_crossatt3 import E2TTS, DurationPredictor
from e2_tts_pytorch.e2_tts_crossatt3 import MelSpec, EncodecWrapper

from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from e2_tts_pytorch.trainer_multigpus_alldatas3 import HFDataset, Text2AudioDataset

from einops import einsum, rearrange, repeat, reduce, pack, unpack
import torchaudio

from datetime import datetime
import json
import numpy as np
import os
from moviepy.editor import VideoFileClip, AudioFileClip
import traceback


audiocond_drop_prob = 1.1
#audiocond_drop_prob = 0.3
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
    e2tts = e2tts.to("cuda")

    #checkpoint = torch.load("/ckptstorage/zhanghaomin/e2/e2_tts_experiment_v2a_encodec/3000.pt", map_location="cpu")
    #checkpoint = torch.load("/ckptstorage/zhanghaomin/e2/e2_tts_experiment_v2a_encodec_more/500.pt", map_location="cpu")
    #checkpoint = torch.load("/ckptstorage/zhanghaomin/e2/e2_tts_experiment_v2a_encodec_more_more/98500.pt", map_location="cpu")
    #checkpoint = torch.load("/ckptstorage/zhanghaomin/e2/e2_tts_experiment_v2a_encodec_more_more_more/190000.pt", map_location="cpu")
    checkpoint = torch.load(ckpt, map_location="cpu")

    #for key in list(checkpoint['model_state_dict'].keys()):
    #    if key.startswith('mel_spec.'):
    #        del checkpoint['model_state_dict'][key]
    #    if key.startswith('transformer.text_registers'):
    #        del checkpoint['model_state_dict'][key]
    e2tts.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    e2tts.vocos = EncodecWrapper("facebook/encodec_24khz")
    for param in e2tts.vocos.parameters():
        param.requires_grad = False
    e2tts.vocos.eval()
    e2tts.vocos.to("cuda")

    #dataset = HFDataset(load_dataset("parquet", data_files={"test": "/ckptstorage/zhanghaomin/tts/GLOBE/data/test-*.parquet"})["test"])
    #sample = dataset[1]
    #mel_spec_raw = sample["mel_spec"].unsqueeze(0)
    #mel_spec = rearrange(mel_spec_raw, 'b d n -> b n d')
    #print(mel_spec.shape, sample["text"])

    #out_dir = "/user-fs/zhanghaomin/v2a_generated/v2a_190000_tests/"
    #out_dir = "/user-fs/zhanghaomin/v2a_generated/tv2a_98500_clips/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #bs = list(range(10)) + [14,16]
    bs = None
    
    SCORE_THRESHOLD_TRAIN = '{"/zhanghaomin/datas/audiocaps": -9999.0, "/radiostorage/WavCaps": -9999.0, "/radiostorage/AudioGroup": 9999.0, "/ckptstorage/zhanghaomin/audioset": -9999.0, "/ckptstorage/zhanghaomin/BBCSoundEffects": 9999.0, "/ckptstorage/zhanghaomin/CLAP_freesound": 9999.0, "/zhanghaomin/datas/musiccap": -9999.0, "/ckptstorage/zhanghaomin/TangoPromptBank": -9999.0, "audioset": "af-audioset", "/ckptstorage/zhanghaomin/audiosetsl": 9999.0, "/ckptstorage/zhanghaomin/giantsoundeffects": -9999.0}'  # /root/datasets/ /radiostorage/
    SCORE_THRESHOLD_TRAIN = json.loads(SCORE_THRESHOLD_TRAIN)
    for key in SCORE_THRESHOLD_TRAIN:
        if key == "audioset":
            continue
        if SCORE_THRESHOLD_TRAIN[key] <= -9000.0:
            SCORE_THRESHOLD_TRAIN[key] = -np.inf
    print("SCORE_THRESHOLD_TRAIN", SCORE_THRESHOLD_TRAIN)
    stft = EncodecWrapper("facebook/encodec_24khz")
    ####eval_dataset = Text2AudioDataset(None, "val_instruments", None, None, None, -1, -1, stft, 0, True, SCORE_THRESHOLD_TRAIN, "/zhanghaomin/codes2/audiocaption/msclapcap_v1.list", -1.0, 1, 1, [drop_prompt], None, 0, vgg_test=[test_scp, start, end, step], video_encoder="clip_vit")
    eval_dataset = Text2AudioDataset(None, "val_vggsound", None, None, None, -1, -1, stft, 0, True, SCORE_THRESHOLD_TRAIN, "/zhanghaomin/codes2/audiocaption/msclapcap_v1.list", -1.0, 1, 1, [drop_prompt], None, 0, vgg_test=[test_scp, start, end, step], video_encoder="clip_vit")
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=1, collate_fn=eval_dataset.collate_fn, num_workers=1, drop_last=False, pin_memory=True)
    i = 0
    for b, batch in enumerate(eval_dataloader):
        if (bs is not None) and (b not in bs):
            continue
        #text, mel_spec, _, mel_lengths = batch
        text, mel_spec, video_paths, mel_lengths, video_drop_prompt, audio_drop_prompt, frames, midis = batch
        print(mel_spec.shape, mel_lengths, text, video_paths, video_drop_prompt, audio_drop_prompt, frames.shape if frames is not None and not isinstance(frames, float) else frames, midis.shape if midis is not None else midis, midis.sum() if midis is not None else midis)
        text = text[i:i+1]
        mel_spec = mel_spec[i:i+1, 0:mel_lengths[i], :]
        mel_lengths = mel_lengths[i:i+1]
        video_paths = video_paths[i:i+1]
        video_path = out_dir + video_paths[0].replace("/", "__")
        audio_path = video_path.replace(".mp4", ".wav")
        
        name = video_paths[0].rsplit("/", 1)[1].rsplit(".", 1)[0]
        
        num = 1
        
        l = mel_lengths[0]
        #cond = mel_spec.repeat(num, 1, 1)
        cond = torch.randn(num, l, e2tts.num_channels)
        duration = torch.tensor([l]*num, dtype=torch.int32)
        lens = torch.tensor([l]*num, dtype=torch.int32)
        print(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "start")
        #e2tts.sample(text=[""]*num, duration=duration.to("cuda"), lens=lens.to("cuda"), cond=cond.to("cuda"), save_to_filename="test.wav", steps=16, cfg_strength=3.0, remove_parallel_component=False, sway_sampling=True)
        e2tts.sample(text=None, duration=duration.to("cuda"), lens=lens.to("cuda"), cond=cond.to("cuda"), save_to_filename=audio_path, steps=64, prompt=text*num, video_drop_prompt=video_drop_prompt, audio_drop_prompt=audio_drop_prompt, cfg_strength=2.0, remove_parallel_component=False, sway_sampling=True, video_paths=video_paths, frames=(frames if frames is None or isinstance(frames, float) else frames.to("cuda")), midis=(midis if midis is None else midis.to("cuda")))
        print(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "sample")
        #one_audio = e2tts.vocos.decode(mel_spec_raw.to("cuda"))
        #one_audio = e2tts.vocos.decode(cond.transpose(-1,-2).to("cuda"))
        #print(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "vocoder")
        #torchaudio.save("ref.wav", one_audio.detach().cpu(), sample_rate = e2tts.sampling_rate)
        
        try:
            os.system("cp \"" + video_paths[0] + "\" \"" + video_path + "\"")
            video = VideoFileClip(video_path)
            audio = AudioFileClip(audio_path)
            print("duration", video.duration, audio.duration)
            if video.duration >= audio.duration:
                video = video.subclip(0, audio.duration)
            else:
                audio = audio.subclip(0, video.duration)
            final_video = video.set_audio(audio)
            final_video.write_videofile(video_path.replace(".mp4", ".v2a.mp4"), codec="libx264", audio_codec="aac")
            print("\"" + video_path.replace(".mp4", ".v2a.mp4") + "\"")
        except Exception as e:
            print("Exception write_videofile:", video_path.replace(".mp4", ".v2a.mp4"))
            traceback.print_exc()
        
        if False:
            if not os.path.exists(out_dir+"groundtruth/"):
                os.makedirs(out_dir+"groundtruth/")
            if not os.path.exists(out_dir+"generated/"):
                os.makedirs(out_dir+"generated/")
            duration_gt = video.duration
            duration_gr = final_video.duration
            duration = min(duration_gt, duration_gr)
            audio_gt = video.audio.subclip(0, duration)
            audio_gr = final_video.audio.subclip(0, duration)
            audio_gt.write_audiofile(out_dir+"groundtruth/"+name+".wav", fps=24000)
            audio_gr.write_audiofile(out_dir+"generated/"+name+".wav", fps=24000)


if __name__ == "__main__":
    main()

