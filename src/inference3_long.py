import sys
sys.path.insert(0, "/zhanghaomin/codes3/e2-tts-pytorch-main/")

if len(sys.argv) >= 6:
    ckpt = sys.argv[1]
    drop_prompt = bool(int(sys.argv[2]))
    test_scp = sys.argv[3]
    start = int(sys.argv[4])
    end = int(sys.argv[5])
    out_dir = sys.argv[6]
    print("inference", ckpt, drop_prompt, test_scp, start, end, out_dir)
else:
    #ckpt = "/ailab-train/speech/zhanghaomin/e2/e2_tts_experiment_v2a_encodec_more_more/98500.pt"
    #ckpt = "/ailab-train/speech/zhanghaomin/e2/e2_tts_experiment_v2a_encodec_more_more_more/190000.pt"
    #ckpt = "/ailab-train/speech/zhanghaomin/e2/e2_tts_experiment_v2a_encodec_more_more_more/315000.pt"
    #ckpt = "/ailab-train/speech/zhanghaomin/e2/e2_tts_experiment_v2a_encodec_more_more_more_more/60000.pt"
    #ckpt = "/ailab-train/speech/zhanghaomin/e2/e2_tts_experiment_v2a_encodec_more_more_more_more_nospeech/235000.pt"
    #ckpt = "/ailab-train/speech/zhanghaomin/e2/e2_tts_experiment_v2a_encodec_more_more_more_more_nospeech_vgganimation/_60000.pt"
    ckpt = "/ailab-train/speech/zhanghaomin/e2/e2_tts_experiment_v2a_encodec_more_more_more_more_nospeech_animation/40000.pt"
    #ckpt = "/ailab-train/speech/zhanghaomin/e2/e2_tts_experiment_v2a_encodec_small_nospeech_vgganimation/120000.pt"
    drop_prompt = True
    #test_scp = "/ailab-train/speech/zhanghaomin/scps/VGGSound/test.scp"
    test_scp = "/ailab-train/speech/zhanghaomin/datas/v2adata/shengcheng.scp"
    start = 0
    end = 100
    out_dir = "./outputs-animation-80k-noprompt-all-30s-shengcheng/"
    #out_dir = "./outputs-small-vgganimation-120k-noprompt-all-30s-shengcheng/"
    nsteps = 25


import torch
from e2_tts_pytorch.e2_tts_crossatt_nospeech import E2TTS, DurationPredictor
from e2_tts_pytorch.e2_tts_crossatt_nospeech import MelSpec, EncodecWrapper

from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from e2_tts_pytorch.trainer_multigpus_alldatas_nospeech import HFDataset, Text2AudioDataset

from e2_tts_pytorch import torch_tools

from einops import einsum, rearrange, repeat, reduce, pack, unpack
import torchaudio

from datetime import datetime
import json
import numpy as np
import os
from moviepy.editor import VideoFileClip, AudioFileClip
import traceback
import math


audiocond_drop_prob = 1.1
#audiocond_drop_prob = 0.3
#cond_drop_prob = 1.1
cond_drop_prob = -0.1
prompt_drop_prob = -0.1
#prompt_drop_prob = 1.1
video_text = True


def read_audio_from_video(video_path):
    if video_path.startswith("/ailab-train/speech/zhanghaomin/VGGSound/"):
        audio_path = video_path.replace("/video/", "/audio/").replace(".mp4", ".wav")
    else:
        audio_path = video_path.replace(".mp4", ".generated.wav")
    if os.path.exists(audio_path):
        # print("video wav exist", audio_path)
        waveform, sr = torchaudio.load(audio_path)
    else:
        # print("video wav not exist", video_path)
        try:
            clip = VideoFileClip(video_path)
            return torch.zeros(1, int(24000 * min(clip.duration, 30.0)))
            clip = AudioFileClip(video_path)
            sound_array = np.array(list(clip.iter_frames()))
            waveform = torch.from_numpy(sound_array).transpose(0, 1).to(torch.float32)
            waveform = waveform[0:1, :]
            if clip.fps != torch_tools.new_freq:
                waveform = torchaudio.functional.resample(waveform, orig_freq=clip.fps, new_freq=torch_tools.new_freq)
            waveform = torch_tools.normalize_wav(waveform)
            ####torchaudio.save(audio_path, waveform, torch_tools.new_freq)
        except:
            print("Error read_audio_from_video", audio_path)
            traceback.print_exc()
            return None
    return waveform


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
        
        if_cond_proj_in = (audiocond_drop_prob < 1.0),
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
    #bs = None
    
    #SCORE_THRESHOLD_TRAIN = '{"/zhanghaomin/datas/audiocaps": -9999.0, "/radiostorage/WavCaps": -9999.0, "/radiostorage/AudioGroup": 9999.0, "/ckptstorage/zhanghaomin/audioset": -9999.0, "/ckptstorage/zhanghaomin/BBCSoundEffects": 9999.0, "/ckptstorage/zhanghaomin/CLAP_freesound": 9999.0, "/zhanghaomin/datas/musiccap": -9999.0, "/ckptstorage/zhanghaomin/TangoPromptBank": -9999.0, "audioset": "af-audioset", "/ckptstorage/zhanghaomin/audiosetsl": 9999.0, "/ckptstorage/zhanghaomin/giantsoundeffects": -9999.0}'  # /root/datasets/ /radiostorage/
    #SCORE_THRESHOLD_TRAIN = json.loads(SCORE_THRESHOLD_TRAIN)
    #for key in SCORE_THRESHOLD_TRAIN:
    #    if key == "audioset":
    #        continue
    #    if SCORE_THRESHOLD_TRAIN[key] <= -9000.0:
    #        SCORE_THRESHOLD_TRAIN[key] = -np.inf
    #print("SCORE_THRESHOLD_TRAIN", SCORE_THRESHOLD_TRAIN)
    stft = EncodecWrapper("facebook/encodec_24khz")
    ####eval_dataset = Text2AudioDataset(None, "val_vggsound", None, None, None, -1, -1, stft, 0, True, SCORE_THRESHOLD_TRAIN, "/zhanghaomin/codes2/audiocaption/msclapcap_v1.list", -1.0, 1, [drop_prompt], None, 0, vgg_test=[test_scp, start, end], video_encoder="clip_vit")
    ####eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=1, collate_fn=eval_dataset.collate_fn, num_workers=1, drop_last=False, pin_memory=True)
    
    with open(test_scp, "r") as fr:
        for line in fr.readlines()[start:end]:
            video, text = line.strip().split("\t")
            run(e2tts, stft, video, text, drop_prompt, nsteps)


def run(e2tts, stft, arg1, arg2, arg3, arg4):
    fbanks = []
    fbank_lens = []
    video_paths = []
    text_selected = []
    for audio, txt in [[arg1, arg2]]:
        waveform = read_audio_from_video(audio)
        if waveform is None:
            continue
        # length = self.val_length
        # waveform = waveform[:, :length*torch_tools.hop_size]
        fbank = stft(waveform).transpose(-1, -2)
        fbanks.append(fbank)
        fbank_lens.append(fbank.shape[1])
        video_paths.append(audio)
        text_selected.append(txt)
        # print("stft", waveform.shape, fbank.shape)
    # max_length = max(fbank_lens)
    # for i in range(len(fbanks)):
    #    if fbanks[i].shape[1] < max_length:
    #        fbanks[i] = torch.cat([fbanks[i], torch.zeros(fbanks[i].shape[0], max_length-fbanks[i].shape[1], fbanks[i].shape[2])], 1)
    mel = torch.cat(fbanks, 0)
    mel_len = torch.Tensor(fbank_lens).to(torch.int32)

    batches = [[text_selected, mel, video_paths, mel_len, [arg3], None]]
    
    i = 0
    print(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "##########")
    ####for b, batch in enumerate(eval_dataloader):
    for b, batch in enumerate(batches):
        #if (bs is not None) and (b not in bs):
        #    continue
        #text, mel_spec, _, mel_lengths = batch
        text, mel_spec, video_paths, mel_lengths, video_drop_prompt, audio_drop_prompt = batch
        print(mel_spec.shape, mel_lengths, text, video_paths, video_drop_prompt, audio_drop_prompt)
        text = text[i:i+1]
        mel_spec = mel_spec[i:i+1, 0:mel_lengths[i], :]
        mel_lengths = mel_lengths[i:i+1]
        video_paths = video_paths[i:i+1]
        video_path = out_dir + video_paths[0].replace("/", "__")
        audio_path = video_path.replace(".mp4", ".wav")
        
        name = video_paths[0].rsplit("/", 1)[1].rsplit(".", 1)[0]
        
        outputs = []

        l = mel_lengths[0]

        num = math.ceil(l / 750.0)
        l0 = l // num
        #video_paths=[(video_paths[0], i*l0*320, (i+1)*l0*320) for i in range(num)]
        video_paths=[(video_paths[0], 0, num*l0*320)]
        
        # cond = mel_spec.repeat(num, 1, 1)
        cond = torch.randn(num, l0, e2tts.num_channels)
        duration = torch.tensor([l0] * num, dtype=torch.int32)
        lens = torch.tensor([l0] * num, dtype=torch.int32)
        print(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "start")
        # e2tts.sample(text=[""]*num, duration=duration.to("cuda"), lens=lens.to("cuda"), cond=cond.to("cuda"), save_to_filename="test.wav", steps=16, cfg_strength=3.0, remove_parallel_component=False, sway_sampling=True)
        outputs = e2tts.sample(text=None, duration=duration.to(e2tts.device), lens=lens.to(e2tts.device),
                     cond=cond.to(e2tts.device), save_to_filename=audio_path, steps=arg4, prompt=text * num,
                     video_drop_prompt=video_drop_prompt*num if video_drop_prompt is not None else video_drop_prompt, audio_drop_prompt=audio_drop_prompt*num if audio_drop_prompt is not None else audio_drop_prompt, cfg_strength=2.0,
                     remove_parallel_component=False, sway_sampling=True, video_paths=video_paths, return_raw_output=True)
        print(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "sample")
        # one_audio = e2tts.vocos.decode(mel_spec_raw.to("cuda"))
        # one_audio = e2tts.vocos.decode(cond.transpose(-1,-2).to("cuda"))
        # print(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "vocoder")
        # torchaudio.save("ref.wav", one_audio.detach().cpu(), sample_rate = e2tts.sampling_rate)

        outputs = outputs.reshape(1, -1, outputs.shape[-1])
        audio_final = e2tts.vocos.decode(outputs.transpose(-1,-2))
        torchaudio.save(audio_path, audio_final.detach().cpu(), sample_rate = e2tts.sampling_rate)
        
        try:
            os.system("cp " + video_paths[0][0] + " " + video_path)
            video = VideoFileClip(video_path)
            audio = AudioFileClip(audio_path)
            print("duration", video.duration, audio.duration)
            if video.duration >= audio.duration:
                video = video.subclip(0, audio.duration)
            else:
                audio = audio.subclip(0, video.duration)
            final_video = video.set_audio(audio)
            final_video.write_videofile(video_path.replace(".mp4", ".v2a.mp4"), codec="libx264", audio_codec="aac")
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
    print(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "@@@@@@@@@@")


if __name__ == "__main__":
    main()

