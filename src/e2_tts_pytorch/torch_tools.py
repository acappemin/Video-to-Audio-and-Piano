import torch
import torchaudio
import random
import itertools
import numpy as np
####from tools.mix import mix
from e2_tts_pytorch.mix import mix

import time
import traceback
import os
#from datasets import load_dataset
####from transformers import ClapModel, ClapProcessor


####clap = ClapModel.from_pretrained("/ckptstorage/zhanghaomin/models/EnCLAP/larger_clap_general/").to("cpu")
####clap.eval()
####for param in clap.parameters():
####    param.requires_grad = False
####clap_processor = ClapProcessor.from_pretrained("/ckptstorage/zhanghaomin/models/EnCLAP/larger_clap_general/")


#from msclap import CLAP
#clap_model = CLAP("/ckptstorage/zhanghaomin/models/msclap/clapcap_weights_2023.pth", version="clapcap", use_cuda=False)
#clap_model.clapcap.eval()
#for param in clap_model.clapcap.parameters():
#    param.requires_grad = False


#new_freq = 16000
#hop_size = 160
new_freq = 24000
#hop_size = 256
hop_size = 320
#total_length = 1024
#MIN_TARGET_LEN = 281
#MAX_TARGET_LEN = 937
total_length = 750
MIN_TARGET_LEN = 750
MAX_TARGET_LEN = 750
#LEN_D = 1
LEN_D = 0

clap_freq = 48000
msclap_freq = 44100
max_len_in_seconds = 10
max_len_in_seconds_msclap = 7
#period_length = 30
period_length = 7
cut_length = 10


def normalize_wav(waveform):
    waveform = waveform - torch.mean(waveform)
    waveform = waveform / (torch.max(torch.abs(waveform[0, :])) + 1e-8)
    return waveform * 0.5


def _pad_spec(fbank, target_length=total_length):
    batch, n_frames, channels = fbank.shape
    p = target_length - n_frames
    if p > 0:
        pad = torch.zeros(batch, p, channels).to(fbank.device)
        fbank = torch.cat([fbank, pad], 1)
    elif p < 0:
        fbank = fbank[:, :target_length, :]

    if channels % 2 != 0:
        fbank = fbank[:, :, :-1]

    return fbank


SCORE_THRESHOLD_VAL = 0.15
#SCORE_THRESHOLD_TRAIN = {
#    "/zhanghaomin/datas/audiocaps": -np.inf,
#    "/radiostorage/WavCaps": -np.inf,
#    "/radiostorage/AudioGroup": -np.inf,
#    "/ckptstorage/zhanghaomin/audioset": -np.inf,
#    "/ckptstorage/zhanghaomin/BBCSoundEffects": -np.inf,
#    "/ckptstorage/zhanghaomin/CLAP_freesound": -np.inf,
#}
SOUNDEFFECT = {
    "/zhanghaomin/datas/audiocaps": False,
    "/radiostorage/WavCaps": False,
    "/radiostorage/AudioGroup": True,
    "/ckptstorage/zhanghaomin/audioset": False,
    "/ckptstorage/zhanghaomin/BBCSoundEffects": False,
    "/ckptstorage/zhanghaomin/CLAP_freesound": False,
    "/zhanghaomin/datas/musiccap": False,
    "/ckptstorage/zhanghaomin/TangoPromptBank": False,
    "/ckptstorage/zhanghaomin/audiosetsl": False,
    "/ckptstorage/zhanghaomin/giantsoundeffects": True,
}
FILTER_NUM = {
    "/zhanghaomin/datas/audiocaps": [0,0],
    "/radiostorage/WavCaps": [0,0],
    "/radiostorage/AudioGroup": [0,0],
    "/ckptstorage/zhanghaomin/audioset": [0,0],
    "/ckptstorage/zhanghaomin/BBCSoundEffects": [0,0],
    "/ckptstorage/zhanghaomin/CLAP_freesound": [0,0],
    "/zhanghaomin/datas/musiccap": [0,0],
    "/ckptstorage/zhanghaomin/TangoPromptBank": [0,0],
    "/ckptstorage/zhanghaomin/audiosetsl": [0,0],
    "/ckptstorage/zhanghaomin/giantsoundeffects": [0,0],
}


TURNOFF_CLAP_FILTER_GLOBAL = False


def pad_wav(waveform, segment_length, text, prefix, val):
    waveform_length = waveform.shape[1]
    
    if segment_length is None or waveform_length == segment_length:
        return waveform, text
    elif waveform_length > segment_length:
        return waveform[:, :segment_length], text
    else:
        if val:
            if (not SOUNDEFFECT[prefix]) or (waveform_length > segment_length / 3.0):
                pad_wav = torch.zeros((waveform.shape[0], segment_length-waveform_length)).to(waveform.device)
                waveform = torch.cat([waveform, pad_wav], 1)
                return waveform, text
            else:
                min_repeats = max(int(segment_length / 3.0 // waveform_length), 2)
                max_repeats = segment_length // waveform_length
                if val:
                    repeats = (min_repeats + max_repeats) // 2
                else:
                    repeats = random.randint(min_repeats, max_repeats)
                waveform = torch.cat([waveform]*repeats, 1)
                if waveform.shape[1] < segment_length:
                    pad_wav = torch.zeros((waveform.shape[0], segment_length-waveform.shape[1])).to(waveform.device)
                    waveform = torch.cat([waveform, pad_wav], 1)
                #if text[-1] in [",", "."]:
                #    text = text[:-1] + " repeat " + str(repeats) + " times" + text[-1]
                #else:
                #    text = text + " repeat " + str(repeats) + " times"
                return waveform, text
        else:
            repeats = segment_length // waveform_length + 1
            waveform = torch.cat([waveform]*repeats, 1)
            assert(waveform.shape[0] == 1 and waveform.shape[1] >= segment_length)
            return waveform[:, :segment_length], text


def msclap_generate(waveform, freq):
    waveform_msclap = torchaudio.functional.resample(waveform, orig_freq=freq, new_freq=msclap_freq)[0]
    start = 0
    end = waveform_msclap.shape[0]
    if waveform_msclap.shape[0] > msclap_freq*max_len_in_seconds_msclap:
        start = random.randint(waveform_msclap.shape[0]-msclap_freq*max_len_in_seconds_msclap)
        end = start+msclap_freq*max_len_in_seconds_msclap
        waveform_msclap = waveform_msclap[start: end]
    if waveform_msclap.shape[0] < msclap_freq*max_len_in_seconds_msclap:
        waveform_msclap = torch.cat([waveform_msclap, torch.zeros(msclap_freq*max_len_in_seconds_msclap-waveform_msclap.shape[0])])
    waveform_msclap = waveform_msclap.reshape(1,1,msclap_freq*max_len_in_seconds_msclap)
    caption = clap_model.generate_caption(waveform_msclap)[0]
    return caption, (start/float(msclap_freq), end/float(msclap_freq))


def do_clap_filter(waveform, text, filename, val, if_clap_filter, main_process, SCORE_THRESHOLD_TRAIN):
    global FILTER_NUM
    
    if isinstance(filename, tuple):
        filename = filename[0]
    if filename.startswith("/radiostorage/"):
        prefix = "/".join(filename.split("/")[:3])
    else:
        prefix = "/".join(filename.split("/")[:4])
    soundeffect = SOUNDEFFECT[prefix]
    
    if not if_clap_filter:
        return np.inf, False, (None, None, soundeffect)
    
    score_threshold = SCORE_THRESHOLD_VAL if val else SCORE_THRESHOLD_TRAIN
    if not if_clap_filter or TURNOFF_CLAP_FILTER_GLOBAL:
        score_threshold = -np.inf
    else:
        if not val:
            score_threshold = SCORE_THRESHOLD_TRAIN[prefix]
            #print(prefix, score_threshold)
    
    resampled = torchaudio.functional.resample(waveform.reshape(1, -1), orig_freq=new_freq, new_freq=clap_freq)[0].numpy()
    resampled = resampled[:clap_freq*max_len_in_seconds]

    inputs = clap_processor(text=[text], audios=[resampled], return_tensors="pt", padding=True, sampling_rate=clap_freq)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clap(**inputs)
        score = torch.dot(outputs.text_embeds[0,:], outputs.audio_embeds[0,:]).item()
    #print("do_clap_filter:", filename, text, resampled.shape, outputs.logits_per_audio, outputs.logits_per_text, score, score < score_threshold)
    if torch.any(torch.isnan(outputs.text_embeds)) or torch.any(torch.isnan(outputs.audio_embeds)):
        return -np.inf, True, None
    
    if main_process and if_clap_filter and not TURNOFF_CLAP_FILTER_GLOBAL:
        FILTER_NUM[prefix][0] += 1
        if score < score_threshold:
            FILTER_NUM[prefix][1] += 1
        if FILTER_NUM[prefix][0] % 10000 == 0 or FILTER_NUM[prefix][0] == 1000:
            print(prefix, FILTER_NUM[prefix][0], FILTER_NUM[prefix][1]/float(FILTER_NUM[prefix][0]))
    return score, score < score_threshold, (outputs.text_embeds, outputs.audio_embeds, soundeffect)


def read_wav_file(filename, text, segment_length, val, if_clap_filter, main_process, SCORE_THRESHOLD_TRAIN, nch):
    try:
        if isinstance(filename, tuple):
            if filename[0].startswith("/radiostorage/"):
                prefix = "/".join(filename[0].split("/")[:3])
            else:
                prefix = "/".join(filename[0].split("/")[:4])
            
            #print(filename, text, segment_length, val)
            wav, utt, period = filename
            
            #size = os.path.getsize(wav)
            #if size > 200000000:
            #    print("Exception too large file:", filename, text, size)
            #    return None, None, None
            
            base, name = wav.rsplit("/", 1)
            temp_base = "/ailab-train/speech/zhanghaomin/wav_temp/" + base.replace("/", "__") + "/"
            temp_filename = temp_base + name
            if os.path.exists(temp_filename):
                waveform, sr = torchaudio.load(temp_filename)
            else:
                #start = time.time()
                waveform0, sr = torchaudio.load(wav)
                #end = time.time()
                #print("load", end-start, wav)
                waveform = torchaudio.functional.resample(waveform0, orig_freq=sr, new_freq=new_freq)[0:nch, :]
                #if nch >= 2:
                #    waveform = torch.cat([waveform.mean(axis=0, keepdims=True), waveform], 0)
                #print("resample", time.time()-end, wav)
                waveform = waveform[:, new_freq*period*period_length: new_freq*(period+1)*period_length]  # 0~period_length s
                waveform = waveform[:, :new_freq*cut_length]
                os.makedirs(temp_base, exist_ok=True)
                torchaudio.save(temp_filename, waveform, new_freq)
            
            start = 0
            if waveform.shape[1] > new_freq*max_len_in_seconds:
                if not val:
                    start = random.randint(0, waveform.shape[1]-new_freq*max_len_in_seconds)
                waveform = waveform[:, start: start+new_freq*max_len_in_seconds]
            
            if val:
                text_index = 0
            else:
                #text_index = random.choice([0,1,2])
                #text_index = random.choice([0,1])
                text_index = 0
            text = text[text_index]
            #text, timestamps = msclap_generate(waveform0[:, sr*period*period_length: sr*(period+1)*period_length], sr)
            #waveform = waveform[int(timestamps[0]*new_freq): int(timestamps[1]*new_freq)]
            #print(waveform.shape, text)
            
            score, filtered, embeddings = do_clap_filter(waveform[0, :], text, filename, val, if_clap_filter, main_process, SCORE_THRESHOLD_TRAIN)
            if filtered:
                print("Exception below threshold:", filename, text, score)
                return None, None, None
        else:
            if filename.startswith("/radiostorage/"):
                prefix = "/".join(filename.split("/")[:3])
            else:
                prefix = "/".join(filename.split("/")[:4])
            
            #size = os.path.getsize(filename)
            #if size > 200000000:
            #    print("Exception too large file:", filename, text, size)
            #    return None, None, None
            
            base, name = filename.rsplit("/", 1)
            temp_base = "/ailab-train/speech/zhanghaomin/wav_temp/" + base.replace("/", "__") + "/"
            temp_filename = temp_base + name
            if os.path.exists(temp_filename):
                #print("wav exist", temp_filename)
                waveform, sr = torchaudio.load(temp_filename)
            else:
                #print("wav not exist", filename)
                #start = time.time()
                waveform, sr = torchaudio.load(filename)  # Faster!!!
                #end = time.time()
                #print("load", end-start, filename)
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=new_freq)[0:nch, :]
                #if nch >= 2:
                #    waveform = torch.cat([waveform.mean(axis=0, keepdims=True), waveform], 0)
                #print("resample", time.time()-end, filename)
                waveform = waveform[:, :new_freq*cut_length]
                os.makedirs(temp_base, exist_ok=True)
                torchaudio.save(temp_filename, waveform, new_freq)
            
            start = 0
            if waveform.shape[1] > new_freq*max_len_in_seconds:
                if not val:
                    start = random.randint(0, waveform.shape[1]-new_freq*max_len_in_seconds)
                waveform = waveform[:, start: start+new_freq*max_len_in_seconds]
            
            if isinstance(text, tuple):
                if val:
                    text_index = 0
                else:
                    text_index = random.choice(list(range(len(text))))
                text = text[text_index]
            
            score, filtered, embeddings = do_clap_filter(waveform[0, :], text, filename, val, if_clap_filter, main_process, SCORE_THRESHOLD_TRAIN)
            if filtered:
                print("Exception below threshold:", filename, text, score)
                return None, None, None
    except Exception as e:
        print("Exception load:", filename, text)
        traceback.print_exc()
        return None, None, None
    #try:
    #    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=new_freq)[0]
    #except Exception as e:
    #    print("Exception resample:", waveform.shape, sr, filename, text)
    #    return None, None, None
    if (waveform.shape[1] / float(new_freq) < 0.2) and (not SOUNDEFFECT[prefix]):
        print("Exception too short wav:", waveform.shape, sr, new_freq, filename, text)
        traceback.print_exc()
        return None, None, None
    try:
        waveform = normalize_wav(waveform)
    except Exception as e:
        print ("Exception normalizing:", waveform.shape, sr, new_freq, filename, text)
        traceback.print_exc()
        #waveform = torch.ones(sample_freq*max_len_in_seconds)
        return None, None, None
    waveform, text = pad_wav(waveform, segment_length, text, prefix, val)
    waveform = waveform / (torch.max(torch.abs(waveform[0, :])) + 1e-8)
    waveform = 0.5 * waveform
    #print(text)
    return waveform, text, embeddings


def get_mel_from_wav(audio, _stft):
    audio = torch.nan_to_num(torch.clip(audio, -1, 1))
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, log_magnitudes_stft, energy = _stft.mel_spectrogram(audio)
    return melspec, log_magnitudes_stft, energy


def argmax_lst(lst):
	return max(range(len(lst)), key=lst.__getitem__)


def select_segment(waveform, target_length):
    ch, wav_length = waveform.shape
    assert(ch == 1 and wav_length == total_length * hop_size)
    energy = []
    for i in range(total_length):
        energy.append(torch.mean(torch.abs(waveform[:, i*hop_size: (i+1)*hop_size])))
    #sum_energy = []
    #for i in range(total_length-target_length+1):
    #    sum_energy.append(sum(energy[i: i+target_length]))
    sum_energy = [sum(energy[:target_length])]
    for i in range(1, total_length-target_length+1):
        sum_energy.append(sum_energy[-1]-energy[i-1]+energy[i+target_length-1])
    
    start = argmax_lst(sum_energy)
    segment = waveform[:, start*hop_size: (start+target_length)*hop_size]
    ch, wav_length = segment.shape
    assert(ch == 1 and wav_length == target_length * hop_size)
    return segment


def wav_to_fbank(paths, texts, num, target_length=total_length, fn_STFT=None, val=False, if_clap_filter=True, main_process=True, SCORE_THRESHOLD_TRAIN="", nch=1):
    assert fn_STFT is not None

    #raw_results = [read_wav_file(path, text, target_length * hop_size, val, if_clap_filter, main_process, SCORE_THRESHOLD_TRAIN, nch) for path, text in zip(paths, texts)]
    results = []
    #for result in raw_results:
    #    if result[0] is not None:
    #        results.append(result)
    for path, text in zip(paths, texts):
        result = read_wav_file(path, text, target_length * hop_size, val, if_clap_filter, main_process, SCORE_THRESHOLD_TRAIN, nch)
        if result[0] is not None:
            results.append(result)
        if num > 0 and len(results) >= num:
            break
    if len(results) == 0:
        ####return None, None, None, None, None
        return None, None, None, None, None, None
    
    ####waveform = torch.cat([result[0] for result in results], 0)
    texts = [result[1] for result in results]
    embeddings = [result[2] for result in results]

    ####fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)
    ####fbank = fbank.transpose(1, 2)
    ####log_magnitudes_stft = log_magnitudes_stft.transpose(1, 2)

    ####fbank, log_magnitudes_stft = _pad_spec(fbank, target_length), _pad_spec(
    ####    log_magnitudes_stft, target_length
    ####)

    ####return fbank, texts, embeddings, log_magnitudes_stft, waveform

    ####fbank = fn_STFT(waveform)
    fbanks = []
    fbank_lens = []
    for result in results:
        if not val:
            length = random.randint(MIN_TARGET_LEN, MAX_TARGET_LEN)
        else:
            length = (MIN_TARGET_LEN + MAX_TARGET_LEN) // 2
        fbank_lens.append(length+LEN_D)
        if not val:
            waveform = select_segment(result[0], length)
        else:
            waveform = result[0][:, :length*hop_size]
        fbank = fn_STFT(waveform).transpose(-1,-2)
        #print("stft", waveform.shape, fbank.shape)
        fbanks.append(fbank)
    max_length = max(fbank_lens)
    for i in range(len(fbanks)):
        if fbanks[i].shape[1] < max_length:
            fbanks[i] = torch.cat([fbanks[i], torch.zeros(fbanks[i].shape[0], max_length-fbanks[i].shape[1], fbanks[i].shape[2])], 1)
    fbank = torch.cat(fbanks, 0)
    fbank_lens = torch.Tensor(fbank_lens).to(torch.int32)
    #print("fbank", fbank.shape, fbank_lens)
    return fbank, texts, None, None, None, fbank_lens


def uncapitalize(s):
    if s:
        return s[:1].lower() + s[1:]
    else:
        return ""

    
def mix_wavs_and_captions(path1, path2, caption1, caption2, target_length=total_length, main_process=True, SCORE_THRESHOLD_TRAIN="", nch=1):
    sound1, caption1, embeddings1 = read_wav_file(path1, caption1, target_length * hop_size, False, False, main_process, SCORE_THRESHOLD_TRAIN, nch)#[0].numpy()
    sound2, caption2, embeddings2 = read_wav_file(path2, caption2, target_length * hop_size, False, False, main_process, SCORE_THRESHOLD_TRAIN, nch)#[0].numpy()
    if sound1 is not None and sound2 is not None:
        mixed_sound = mix(sound1.numpy(), sound2.numpy(), 0.5, new_freq)
        mixed_sound = mixed_sound.astype(np.float32)
        mixed_caption = "{} and {}".format(caption1, uncapitalize(caption2))
        
        #resampled = torchaudio.functional.resample(torch.from_numpy(mixed_sound).reshape(1, -1), orig_freq=new_freq, new_freq=clap_freq)[0].numpy()
        #resampled = resampled[:clap_freq*max_len_in_seconds]
        
        #inputs = clap_processor(text=[mixed_caption], audios=[resampled], return_tensors="pt", padding=True, sampling_rate=clap_freq)
        #inputs = {k: v.to("cpu") for k, v in inputs.items()}
        #with torch.no_grad():
        #    outputs = clap(**inputs)
        if not (embeddings1[2] or embeddings2[2]):
            filename = path1
        else:
            filename = "/radiostorage/AudioGroup"
        score, filtered, embeddings = do_clap_filter(torch.from_numpy(mixed_sound)[0, :], mixed_caption, filename, False, False, main_process, SCORE_THRESHOLD_TRAIN)
        #print(score, filtered, embeddings if embeddings is None else embeddings[2], path1, path2, filename)
        if filtered:
            #print("Exception below threshold:", path1, path2, caption1, caption2, filename, score)
            return None, None, None
        
        return mixed_sound, mixed_caption, embeddings
    else:
        return None, None, None


def augment(paths, texts, num_items=4, target_length=total_length, main_process=True, SCORE_THRESHOLD_TRAIN="", nch=1):
    mixed_sounds, mixed_captions, mixed_embeddings = [], [], []
    combinations = list(itertools.combinations(list(range(len(texts))), 2))
    random.shuffle(combinations)
    if len(combinations) < num_items:
        selected_combinations = combinations
    else:
        selected_combinations = combinations[:num_items]
        
    for (i, j) in selected_combinations:
        new_sound, new_caption, new_embeddings = mix_wavs_and_captions(paths[i], paths[j], texts[i], texts[j], target_length, main_process, SCORE_THRESHOLD_TRAIN, nch)
        if new_sound is not None:
            mixed_sounds.append(new_sound)
            mixed_captions.append(new_caption)
            mixed_embeddings.append(new_embeddings)
    
    if len(mixed_sounds) == 0:
        return None, None, None
    
    waveform = torch.tensor(np.concatenate(mixed_sounds, 0))
    waveform = waveform / (torch.max(torch.abs(waveform[0, :])) + 1e-8)
    waveform = 0.5 * waveform
    
    return waveform, mixed_captions, mixed_embeddings


def augment_wav_to_fbank(paths, texts, num_items=4, target_length=total_length, fn_STFT=None, main_process=True, SCORE_THRESHOLD_TRAIN="", nch=1):
    assert fn_STFT is not None
    
    waveform, captions, embeddings = augment(paths, texts, num_items, target_length, main_process, SCORE_THRESHOLD_TRAIN, nch)
    if waveform is None:
        ####return None, None, None, None, None
        return None, None, None, None, None, None
    
    ####fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)
    ####fbank = fbank.transpose(1, 2)
    ####log_magnitudes_stft = log_magnitudes_stft.transpose(1, 2)
    ####
    ####fbank, log_magnitudes_stft = _pad_spec(fbank, target_length), _pad_spec(
    ####    log_magnitudes_stft, target_length
    ####)
    ####
    ####return fbank, log_magnitudes_stft, waveform, captions, embeddings
    
    ####fbank = fn_STFT(waveform)
    fbanks = []
    fbank_lens = []
    for i in range(waveform.shape[0]):
        length = random.randint(MIN_TARGET_LEN, MAX_TARGET_LEN)
        fbank_lens.append(length+LEN_D)
        ####fbank = fn_STFT(waveform[i:i+1, :length*hop_size]).transpose(-1,-2)
        fbank = fn_STFT(select_segment(waveform[i:i+1, :], length)).transpose(-1,-2)
        fbanks.append(fbank)
    max_length = max(fbank_lens)
    for i in range(len(fbanks)):
        if fbanks[i].shape[1] < max_length:
            fbanks[i] = torch.cat([fbanks[i], torch.zeros(fbanks[i].shape[0], max_length-fbanks[i].shape[1], fbanks[i].shape[2])], 1)
    fbank = torch.cat(fbanks, 0)
    fbank_lens = torch.Tensor(fbank_lens).to(torch.int32)
    return fbank, None, None, captions, None, fbank_lens