from __future__ import annotations

import os
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, SequentialLR

import torchaudio

from einops import rearrange

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from ema_pytorch import EMA

from loguru import logger

from e2_tts_pytorch.e2_tts_crossatt3_2 import (
    E2TTS,
    DurationPredictor,
    MelSpec
)

import traceback
import numpy as np
from moviepy.editor import AudioFileClip, VideoFileClip

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def to_numpy(t):
    return t.detach().cpu().numpy()

# plot spectrogram

def plot_spectrogram(spectrogram):
    spectrogram = to_numpy(spectrogram)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(spectrogram.T, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    plt.close()
    return fig

# collation

def collate_fn(batch):
    mel_specs = [item['mel_spec'].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value = 0)
        padded_mel_specs.append(padded_spec)
    
    mel_specs = torch.stack(padded_mel_specs)

    text = [item['text'] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    return dict(
        mel = mel_specs,
        mel_lengths = mel_lengths,
        text = text,
        text_lengths = text_lengths,
    )

# dataset

class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate = 24_000,
        hop_length = 256
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.mel_spectrogram = MelSpec(sampling_rate=target_sample_rate)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data[index]
        audio = row['audio']['array']

        #logger.info(f"Audio shape: {audio.shape}")

        sample_rate = row['audio']['sampling_rate']
        duration = audio.shape[-1] / sample_rate

        if duration > 20 or duration < 0.3:
            logger.warning(f"Skipping due to duration out of bound: {duration}")
            return self.__getitem__((index + 1) % len(self.data))
        
        audio_tensor = torch.from_numpy(audio).float()
        
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)
        
        audio_tensor = rearrange(audio_tensor, 't -> 1 t')
        
        mel_spec = self.mel_spectrogram(audio_tensor)
        
        mel_spec = rearrange(mel_spec, '1 d t -> d t')
        
        text = row['transcript']
        
        return dict(
            mel_spec = mel_spec,
            text = text,
        )

# trainer

class E2Trainer:
    def __init__(
        self,
        model: E2TTS,
        optimizer,
        num_warmup_steps=20000,
        grad_accumulation_steps=1,
        duration_predictor: DurationPredictor | None = None,
        checkpoint_path = None,
        log_file = "logs.txt",
        max_grad_norm = 1.0,
        sample_rate = 22050,
        tensorboard_log_dir = 'runs/e2_tts_experiment',
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        use_switch_ema = False,
        if_text = False,
        if_prompt = False
    ):
        logger.add(log_file)

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters = True)

        self.accelerator = Accelerator(
            log_with = "all",
            kwargs_handlers = [ddp_kwargs],
            gradient_accumulation_steps = grad_accumulation_steps,
            **accelerate_kwargs
        )
        self.accelerator.wait_for_everyone()

        self.target_sample_rate = sample_rate

        self.model = model

        self.need_velocity_consistent_loss = model.velocity_consistency_weight > 0.

        #self.ema_model = EMA(
        #    model,
        #    include_online_model = False,
        #    **ema_kwargs
        #)

        self.use_switch_ema = use_switch_ema

        self.duration_predictor = duration_predictor
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.checkpoint_path = default(checkpoint_path, 'model.pth')
        self.mel_spectrogram = MelSpec(sampling_rate=self.target_sample_rate)

        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )
        #self.ema_model = self.accelerator.prepare(self.ema_model)
        self.max_grad_norm = max_grad_norm
        
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
        self.tensorboard_log_dir = tensorboard_log_dir
        self.if_text = if_text
        self.if_prompt = if_prompt
        
        self.device_id = self.accelerator.device.index
        self.num_processes = self.accelerator.num_processes

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, step, finetune=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict = self.accelerator.unwrap_model(self.model).state_dict(),
                #optimizer_state_dict = self.accelerator.unwrap_model(self.optimizer).state_dict(),
                #ema_model_state_dict = self.ema_model.state_dict(),
                #scheduler_state_dict = self.scheduler.state_dict(),
                #step = step,
            )

            self.accelerator.save(checkpoint, self.tensorboard_log_dir + "/" + str(step) + ".pt")

    def load_checkpoint(self):
        if not exists(self.checkpoint_path) or not os.path.exists(self.checkpoint_path):
            return 0

        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        for key in list(checkpoint['model_state_dict'].keys()):
            #if key.startswith('mel_spec.'):
            #    del checkpoint['model_state_dict'][key]
            if key.startswith('transformer.text_registers'):
                if checkpoint['model_state_dict'][key].shape[1] != self.accelerator.unwrap_model(self.model).transformer.text_registers.shape[1]:
                    print('miss match: transformer.text_registers', checkpoint['model_state_dict'][key].shape, self.accelerator.unwrap_model(self.model).transformer.text_registers.shape)
                    del checkpoint['model_state_dict'][key]
            ####
            #if key.startswith("conv") or key.startswith("pool"):
            #    print('del video encoder', key, checkpoint['model_state_dict'][key].shape)
            #    del checkpoint['model_state_dict'][key]
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'], strict=False)
        #self.accelerator.unwrap_model(self.optimizer).load_state_dict(checkpoint['optimizer_state_dict'])

        if self.is_main:
            model0 = {}
            for key in list(checkpoint['model_state_dict'].keys()):
                model0[key] = checkpoint['model_state_dict'][key].shape
            model1 = {}
            for key, param in self.accelerator.unwrap_model(self.model).state_dict().items():
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
            for key in model1.keys():
                if key not in model0:
                    pass
                    #print("New key found", key, model1[key])
                else:
                    if model0[key] != model1[key]:
                        pass
                        #print("Miss match", key, model0[key], model1[key])

        #if self.is_main:
        #    self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])

        #if self.scheduler:
        #    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        #return checkpoint['step']
        return 0

    def evaluate(self, eval_dataloader, epoch, epochs, global_step):
        if eval_dataloader is None:
            return

        total_val_loss, N, total_lossmore1, total_lossmore2, total_lossmore3, total_lossmore4 = 0, 0, 0, 0, 0, 0
        self.model.eval()
        eval_progress_bar = tqdm(eval_dataloader, desc=f"Epoch {epoch}/{epochs}", unit="step", disable=not self.accelerator.is_local_main_process)
        for step, batch in enumerate(eval_dataloader):
            with self.accelerator.accumulate(self.model) and torch.no_grad():
                text, mel_spec, video_paths, mel_lengths, video_drop_prompt, audio_drop_prompt, frames, midis = batch

                val_loss, cond, pred, pred_data, lossmore = self.model(
                    mel_spec,
                    text=(text if self.if_text else None),
                    times=0.5,
                    lens=mel_lengths,
                    velocity_consistency_model=None,
                    prompt=(text if self.if_prompt else None),
                    video_drop_prompt=video_drop_prompt,
                    audio_drop_prompt=audio_drop_prompt,
                    val=True,
                    video_paths=video_paths,
                    frames=frames,
                    midis=midis
                )
                a = torch.tensor(val_loss.item()*len(text), dtype=torch.float32).reshape(1).to(val_loss.device)
                b = torch.tensor(len(text), dtype=torch.int32).reshape(1).to(val_loss.device)
                c = torch.tensor(lossmore[0].item()*len(text), dtype=torch.float32).reshape(1).to(lossmore[0].device)
                d = torch.tensor(lossmore[1].item()*len(text), dtype=torch.float32).reshape(1).to(lossmore[1].device)
                e = torch.tensor(lossmore[2].item()*len(text), dtype=torch.float32).reshape(1).to(lossmore[2].device)
                f = torch.tensor(lossmore[3].item()*len(text), dtype=torch.float32).reshape(1).to(lossmore[3].device)
                val_loss_gather, N_gather, lossmore_gather1, lossmore_gather2, lossmore_gather3, lossmore_gather4 = self.accelerator.gather_for_metrics((a, b, c, d, e, f))
                for i in range(val_loss_gather.shape[0]):
                    total_val_loss += val_loss_gather[i].item()
                    N += N_gather[i].item()
                    total_lossmore1 += lossmore_gather1[i].item()
                    total_lossmore2 += lossmore_gather2[i].item()
                    total_lossmore3 += lossmore_gather3[i].item()
                    total_lossmore4 += lossmore_gather4[i].item()
                eval_progress_bar.update(1)

        if self.accelerator.is_local_main_process:
            total_val_loss = round(total_val_loss/float(N), 4)
            total_lossmore1 = round(total_lossmore1/float(N), 4)
            total_lossmore2 = round(total_lossmore2/float(N), 4)
            total_lossmore3 = round(total_lossmore3/float(N), 4)
            total_lossmore4 = round(total_lossmore4/float(N), 4)
            result_string = "Epoch: {}, GlobalStep: {}, ValLoss: {}, N: {}, Lossmore1: {}, Lossmore2: {}, Lossmore3: {}, Lossmore4: {} (average loss)\n".format(epoch, global_step, total_val_loss, N, total_lossmore1, total_lossmore2, total_lossmore3, total_lossmore4)
            #tp, fp, fn, tn = total_lossmore1, total_lossmore2, total_lossmore3, total_lossmore4
            #result_string = "Epoch: {}, GlobalStep: {}, ValLoss: {}, N: {}, Lossmore1: {}, Lossmore2: {}, Lossmore3: {}, Lossmore4: {} (average loss)\n".format(epoch, global_step, total_val_loss, N, tp / (tp + fp) if (tp + fp) != 0 else 0, tp / (tp + fn) if (tp + fn) != 0 else 0, 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0, tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 0)
            logger.info(result_string)

        torch.cuda.empty_cache()
        self.model.train()

    def train(self, datasets, epochs, batch_size, num_workers=12, save_step=1000):

        params_d = {}
        trainable_d = {}
        for n, p in self.model.named_parameters():
            key = ".".join(n.split(".")[:2])
            if key not in params_d:
                params_d[key] = [0, 0]
                trainable_d[key] = p.requires_grad
            if p.requires_grad:
                params_d[key][0] += p.numel()
            else:
                params_d[key][1] += p.numel()
            if key != "module.transformer":
                assert(trainable_d[key] == p.requires_grad)
        print(params_d)
        print(trainable_d)
        num_trainable_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Num trainable parameters: {}".format(num_trainable_parameters))

        train_dataset = datasets[0]
        eval_datasets = datasets[1:]
        #train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=num_workers, pin_memory=True)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size*train_dataset.multi, collate_fn=train_dataset.collate_fn, num_workers=num_workers, drop_last=True, pin_memory=True)
        eval_dataloaders = [DataLoader(eval_dataset, shuffle=False, batch_size=(len(eval_dataset.video_drop_prompt) if eval_dataset.video_drop_prompt is not None else 16), collate_fn=eval_dataset.collate_fn, num_workers=num_workers, drop_last=False, pin_memory=True) if eval_dataset is not None else None for eval_dataset in eval_datasets]
        print("eval_datasets batch_size", [(len(eval_dataset.video_drop_prompt) if eval_dataset.video_drop_prompt is not None else 16) if eval_dataset is not None else None for eval_dataset in eval_datasets])
        
        total_steps = len(train_dataloader) * epochs
        decay_steps = total_steps - self.num_warmup_steps
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=self.num_warmup_steps)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
        self.scheduler = SequentialLR(self.optimizer, 
                                      schedulers=[warmup_scheduler, decay_scheduler],
                                      milestones=[self.num_warmup_steps])
        train_dataloader, self.scheduler = self.accelerator.prepare(train_dataloader, self.scheduler)
        eval_dataloaders = [self.accelerator.prepare(eval_dataloader) for eval_dataloader in eval_dataloaders if eval_dataloader is not None]
        start_step = self.load_checkpoint()
        global_step = start_step

        #### dpo
        velocity_consistency_model = None
        ####velocity_consistency_model = self.accelerator.unwrap_model(self.model)
        #### dpo
        for epoch in range(epochs):

            if epoch == 0:
                [self.evaluate(eval_dataloader, 1, epochs, 0) for eval_dataloader in eval_dataloaders]

            self.model.train()
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="step", disable=not self.accelerator.is_local_main_process)
            epoch_loss = 0.0

            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    #text_inputs = batch['text']
                    #mel_spec = rearrange(batch['mel'], 'b d n -> b n d')
                    #mel_lengths = batch["mel_lengths"]
                    text, mel_spec, video_paths, mel_lengths, video_drop_prompt, audio_drop_prompt, frames, midis = batch
                    #print("batchsize", len(text))
                    for i, video_path in enumerate(video_paths):
                        if video_path is not None:
                            if random.random() >= 0.5:
                                video_drop_prompt[i] = True
                            else:
                                video_drop_prompt[i] = False
                    #print("batch", text, mel_spec.shape, video_paths, mel_lengths, video_drop_prompt, audio_drop_prompt, frames.shape if frames is not None and not isinstance(frames, float) else frames, midis.shape)

                    if exists(self.duration_predictor):
                        dur_loss = self.duration_predictor(mel_spec, lens=batch.get('durations'))
                        self.writer.add_scalar('duration loss', dur_loss.detach().cpu().item(), global_step)

                    #velocity_consistency_model = None
                    #if self.need_velocity_consistent_loss and self.ema_model.initted:
                    #    velocity_consistency_model = self.accelerator.unwrap_model(self.ema_model).ema_model

                    loss, cond, pred, pred_data, lossmore = self.model(
                        mel_spec,
                        text=(text if self.if_text else None),
                        lens=mel_lengths,
                        velocity_consistency_model=velocity_consistency_model,
                        prompt=(text if self.if_prompt else None),
                        video_drop_prompt=video_drop_prompt,
                        audio_drop_prompt=audio_drop_prompt,
                        video_paths=video_paths,
                        frames=frames,
                        midis=midis
                    )

                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                #self.accelerator.unwrap_model(self.ema_model).update()

                if self.accelerator.is_local_main_process:
                    logger.info(f"step {global_step+1}: loss = {loss.detach().cpu().item():.4f}")
                    self.writer.add_scalar('loss', loss.detach().cpu().item(), global_step)
                    self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_step)
                
                global_step += 1
                epoch_loss += loss.detach().cpu().item()
                progress_bar.set_postfix(loss=loss.detach().cpu().item())
                
                if global_step % save_step == 0:
                    self.save_checkpoint(global_step)
                    self.writer.add_figure("mel/target", plot_spectrogram(mel_spec[0,:,:]), global_step)
                    self.writer.add_figure("mel/mask", plot_spectrogram(cond[0,:,:]), global_step)
                    self.writer.add_figure("mel/prediction", plot_spectrogram(pred_data[0,:,:]), global_step)
                    [self.evaluate(eval_dataloader, epoch+1, epochs, global_step) for eval_dataloader in eval_dataloaders]
                
                #if global_step % 10 == 0:
                #    torch.cuda.empty_cache()
            
            epoch_loss /= len(train_dataloader)
            if self.accelerator.is_local_main_process:
                logger.info(f"epoch {epoch+1}/{epochs} - average loss = {epoch_loss:.4f}")
                self.writer.add_scalar('epoch average loss', epoch_loss, epoch)

        #if self.use_switch_ema:
        #    self.ema_model.update_model_with_ema()

        self.writer.close()


import json
import random
import pandas as pd
from e2_tts_pytorch import torch_tools

DURATION = torch_tools.total_length
#DURATION = 3000
#beta = 1.5960
#theta = 0.3259
cand = 99999999

class Text2AudioDataset(Dataset):
    def __init__(self, dataset, part, prefix, text_column, audio_column, num_examples=-1, samples=-1, stft=None, augment=-1, main_process=True, SCORE_THRESHOLD_TRAIN="", train_file="", theta=0.0, vggsound=0, instruments=0, video_drop_prompt=None, audio_drop_prompt=None, device_id=0, vgg_test=None, video_encoder="clip_vit", val_length=None, num_processes=8):

        #inputs = list(dataset[text_column])
        #self.inputs = [prefix + inp for inp in inputs]
        #self.audios = list(dataset[audio_column])
        #self.indices = list(range(len(self.inputs)))
        #
        #print("audios", len(self.audios))
        #self.new_audios = []
        #for index, audio in enumerate(self.audios):
        #    utt, fmt = audio.split(".")
        #    new_audio = "/zhanghaomin/datas/audioset_sl/mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/AudioSet_SL_flac/" + utt + ".flac"
        #    #if os.path.exists(new_audio):
        #    self.new_audios.append(new_audio)
        #self.audios = self.new_audios
        #N = len(self.audios)
        #print("audios", len(self.new_audios))
        
        
        test_final = "/ailab-train/speech/zhanghaomin/scps/tango-master/data/test_audiocaps_subset.json"
        test_utts = {}
        with open(test_final, "r") as fr:
            for line in fr.readlines():
                wav = json.loads(line.strip())["location"]
                utt = wav.rsplit("/", 1)[-1].rsplit("_", 1)[0]
                utt = "Y"+utt
                assert(utt not in test_utts)
                test_utts[utt] = 1
        main_process and print("test_final", len(test_utts))
        
        bbc_soundeffects_utts = {}
        freesound_utts = {}
        
        audioset_filter_labels = {"Music": 0, "Speech": 0, "Vehicle": 0, "Musical instrument": 0}
        
        
        self.inputs = []
        self.audios = []
        self.indices = []
        N = 0
        
        
        audiocaps = True
        if SCORE_THRESHOLD_TRAIN["/zhanghaomin/datas/audiocaps"] >= 9000.0:
            audiocaps = False
        
        audioset_sl = True
        bbc_soundeffects = True
        freesound = True
        soundbible = True
        if SCORE_THRESHOLD_TRAIN["/radiostorage/WavCaps"] >= 9000.0:
            audioset_sl = False
            bbc_soundeffects = False
            freesound = False
            soundbible = False
        
        soundeffects = True
        if SCORE_THRESHOLD_TRAIN["/radiostorage/AudioGroup"] >= 9000.0:
            soundeffects = False
        self.soundeffects = soundeffects
        
        audioset = True
        if SCORE_THRESHOLD_TRAIN["/ckptstorage/zhanghaomin/audioset"] >= 9000.0:
            audioset = False
        
        bbc_soundeffects2 = True
        if SCORE_THRESHOLD_TRAIN["/ckptstorage/zhanghaomin/BBCSoundEffects"] >= 9000.0:
            bbc_soundeffects2 = False
        
        freesound2 = True
        if SCORE_THRESHOLD_TRAIN["/ckptstorage/zhanghaomin/CLAP_freesound"] >= 9000.0:
            freesound2 = False
        
        musiccaps = True
        if SCORE_THRESHOLD_TRAIN["/zhanghaomin/datas/musiccap"] >= 9000.0:
            musiccaps = False
        
        tangopromptbank = True
        if SCORE_THRESHOLD_TRAIN["/ckptstorage/zhanghaomin/TangoPromptBank"] >= 9000.0:
            tangopromptbank = False
        
        audioset_sl_2ch = True
        if SCORE_THRESHOLD_TRAIN["/ckptstorage/zhanghaomin/audiosetsl"] >= 9000.0:
            audioset_sl_2ch = False
        self.audioset_sl_2ch = audioset_sl_2ch
        
        boom_epic = True
        if SCORE_THRESHOLD_TRAIN["/ckptstorage/zhanghaomin/giantsoundeffects"] >= 9000.0:
            boom_epic = False
        self.boom_epic = boom_epic
        
        if isinstance(part, list):
            part, scp_ac, start_ac, end_ac = part
            assert(part == "val_audiocaps")
        else:
            scp_ac = None
        
        if (audioset_sl and part in ["train", "train_val_audioset_sl"]) or (part == "val_audioset_sl"):
            self.audioset_sl_inputs = []
            self.audioset_sl_audios = []
            self.audioset_sl_indices = []
            audioset_sl_path_train = "/zhanghaomin/codes2/tango-master/data/train_audioset_sl.json"
            audioset_sl_path_val = "/zhanghaomin/codes2/tango-master/data/val_audioset_sl.json"
            audioset_sl_path_train_val = "/ailab-train/speech/zhanghaomin/scps/tango-master/data/train_val_audioset_sl.json"
            if part == "train":
                audioset_sl_path = audioset_sl_path_train
            elif part == "train_val_audioset_sl":
                audioset_sl_path = audioset_sl_path_train_val
            else:
                audioset_sl_path = audioset_sl_path_val
            FN = 0
            with open(audioset_sl_path, "r") as fr:
                for index, line in enumerate(fr.readlines()):
                    jsondata = json.loads(line.strip())
                    utt = jsondata["id"].rsplit(".", 1)[0]
                    if part in ["train", "train_val_audioset_sl"] and utt in test_utts:
                        FN += 1
                        continue
                    caption = jsondata["caption"]
                    audio = "/radiostorage/WavCaps/Zip_files/AudioSet_SL/mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/AudioSet_SL_flac/" + utt + ".flac"
                    self.audioset_sl_inputs.append(caption)
                    self.audioset_sl_audios.append(audio)
                    self.audioset_sl_indices.append(N + index)
            main_process and print(part, "audioset_sl", len(self.audioset_sl_audios), "filtered", FN)
            self.inputs.extend(self.audioset_sl_inputs)
            self.audios.extend(self.audioset_sl_audios)
            self.indices.extend(self.audioset_sl_indices)
            N = len(self.audios)
            main_process and print(part, "audioset_sl audios", len(self.audios))
        
        if (audiocaps and part in ["train", "train_val_audioset_sl"]) or (part == "val_audiocaps"):
            self.audiocaps_inputs = []
            self.audiocaps_audios = []
            self.audiocaps_indices = []
            audiocaps_path_train = "/ailab-train/speech/zhanghaomin/scps/tango-master/data/audiocaps/train_audiocaps.json"
            audiocaps_path_val = "/ailab-train/speech/zhanghaomin/scps/tango-master/data/audiocaps/test_audiocaps.json"
            if scp_ac is not None:
                audiocaps_path_val = scp_ac
            if part in ["train", "train_val_audioset_sl"]:
                audiocaps_path = audiocaps_path_train
            else:
                audiocaps_path = audiocaps_path_val
            FN = 0
            with open(audiocaps_path, "r") as fr:
                lines = fr.readlines()
                if scp_ac is not None:
                    lines = lines[start_ac: end_ac]
                for index, line in enumerate(lines):
                    jsondata = json.loads(line.strip())
                    utt = jsondata["id"]
                    if part in ["train", "train_val_audioset_sl"] and utt in test_utts:
                        FN += 1
                        continue
                    caption = jsondata["caption"]
                    audio = jsondata["audio"]
                    self.audiocaps_inputs.append(caption)
                    self.audiocaps_audios.append(audio)
                    self.audiocaps_indices.append(N + index)
            main_process and print(part, "audiocaps", len(self.audiocaps_audios), "filtered", FN)
            self.inputs.extend(self.audiocaps_inputs)
            self.audios.extend(self.audiocaps_audios)
            self.indices.extend(self.audiocaps_indices)
            N = len(self.audios)
            main_process and print(part, "audiocaps audios", len(self.audios))
        
        if bbc_soundeffects and part in ["train", "train_val_audioset_sl"]:
            self.bbc_soundeffects_inputs = []
            self.bbc_soundeffects_audios = []
            self.bbc_soundeffects_indices = []
            with open("/ailab-train/speech/zhanghaomin/scps/tango-master/data/train_bbc_sound_effects.json", "r") as fr:
                for index, line in enumerate(fr.readlines()):
                    jsondata = json.loads(line.strip())
                    utt = jsondata["id"]
                    bbc_soundeffects_utts[utt] = 1
                    caption = jsondata["caption"]
                    audio = "/radiostorage/WavCaps/Zip_files/BBC_Sound_Effects/mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/BBC_Sound_Effects_flac/" + utt + ".flac"
                    self.bbc_soundeffects_inputs.append(caption)
                    self.bbc_soundeffects_audios.append(audio)
                    self.bbc_soundeffects_indices.append(N + index)
            main_process and print(part, "bbc_soundeffects", len(self.bbc_soundeffects_audios))
            self.inputs.extend(self.bbc_soundeffects_inputs)
            self.audios.extend(self.bbc_soundeffects_audios)
            self.indices.extend(self.bbc_soundeffects_indices)
            N = len(self.audios)
            main_process and print(part, "bbc_soundeffects audios", len(self.audios))
        
        if freesound and part in ["train", "train_val_audioset_sl"]:
            self.freesound_inputs = []
            self.freesound_audios = []
            self.freesound_indices = []
            with open("/ailab-train/speech/zhanghaomin/scps/tango-master/data/train_freesound.json", "r") as fr:
                for index, line in enumerate(fr.readlines()):
                    jsondata = json.loads(line.strip())
                    utt = jsondata["id"]
                    freesound_utts[utt] = 1
                    caption = jsondata["caption"]
                    audio = "/radiostorage/WavCaps/Zip_files/FreeSound/mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/FreeSound_flac/" + utt + ".flac"
                    self.freesound_inputs.append(caption)
                    self.freesound_audios.append(audio)
                    self.freesound_indices.append(N + index)
            main_process and print(part, "freesound", len(self.freesound_audios))
            self.inputs.extend(self.freesound_inputs)
            self.audios.extend(self.freesound_audios)
            self.indices.extend(self.freesound_indices)
            N = len(self.audios)
            main_process and print(part, "freesound audios", len(self.audios))
        
        if soundbible and part in ["train", "train_val_audioset_sl"]:
            self.soundbible_inputs = []
            self.soundbible_audios = []
            self.soundbible_indices = []
            with open("/ailab-train/speech/zhanghaomin/scps/tango-master/data/train_soundbible.json", "r") as fr:
                for index, line in enumerate(fr.readlines()):
                    jsondata = json.loads(line.strip())
                    utt = jsondata["id"]
                    caption = jsondata["caption"]
                    audio = "/radiostorage/WavCaps/Zip_files/SoundBible/mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/SoundBible_flac/" + utt + ".flac"
                    self.soundbible_inputs.append(caption)
                    self.soundbible_audios.append(audio)
                    self.soundbible_indices.append(N + index)
            main_process and print(part, "soundbible", len(self.soundbible_audios))
            self.inputs.extend(self.soundbible_inputs)
            self.audios.extend(self.soundbible_audios)
            self.indices.extend(self.soundbible_indices)
            N = len(self.audios)
            main_process and print(part, "soundbible audios", len(self.audios))
        
        if (soundeffects and part in ["train", "train_val_audioset_sl"]) or (part == "val_soundeffects"):
            self.soundeffects_inputs = []
            self.soundeffects_audios = []
            self.soundeffects_indices = []
            #soundeffects_path_train = "/zhanghaomin/codes2/audiocaption/wav_all_train.scp"
            #soundeffects_path_val = "/zhanghaomin/codes2/audiocaption/wav_all_val.scp"
            #soundeffects_path_train = "/zhanghaomin/codes2/audiocaption/wav_msclap_all_train.scp"
            soundeffects_path_train = train_file
            soundeffects_path_val = "/zhanghaomin/codes2/audiocaption/wav_msclap_all_val.scp"
            if part in ["train", "train_val_audioset_sl"]:
                soundeffects_path = soundeffects_path_train
            else:
                soundeffects_path = soundeffects_path_val
            with open(soundeffects_path, 'r') as fr:
                for index, line in enumerate(fr.readlines()):
                    if soundeffects_path.endswith("msclapcap_v1.list"):
                        utt, wav, caption1, score = line.strip().split('"@$&#"')
                        caption2 = "blank"
                        name = "blank"
                    else:
                        utt, wav, name, caption1, caption2 = line.strip().split('"@$&#"')
                    wav = wav.replace("/radiostorage/AudioGroup/", "/radiostorage/AudioGroup/")
                    period = int(utt.split('_')[-1])
                    self.soundeffects_inputs.append((caption1, caption2, name))
                    self.soundeffects_audios.append((wav, utt, period))
                    self.soundeffects_indices.append(N + index)
            main_process and print(part, "soundeffects", len(self.soundeffects_audios))
            self.inputs.extend(self.soundeffects_inputs)
            self.audios.extend(self.soundeffects_audios)
            self.indices.extend(self.soundeffects_indices)
            N = len(self.audios)
            main_process and print(part, "soundeffects audios", len(self.audios))
        
        if audioset and part in ["train", "train_val_audioset_sl"]:
            self.audioset_inputs = []
            self.audioset_audios = []
            self.audioset_indices = []
            FN = 0
            FN2 = 0
            if SCORE_THRESHOLD_TRAIN["audioset"] == "af-audioset":
                audioset_path = "/ailab-train/speech/zhanghaomin/scps/audioset/audioset_train_af.json"
            else:
                audioset_path = "/ckptstorage/zhanghaomin/audioset/audioset_train.json"
            with open(audioset_path, "r") as fr:
                for index, line in enumerate(fr.readlines()):
                    jsondata = json.loads(line.strip())
                    if SCORE_THRESHOLD_TRAIN["audioset"] == "af-audioset":
                        utt = jsondata["id"]
                        if part in ["train", "train_val_audioset_sl"] and utt in test_utts:
                            FN += 1
                            continue
                        caption = jsondata["caption"]
                        audio = jsondata["audio"]
                    else:
                        utt = jsondata["id"]
                        if part in ["train", "train_val_audioset_sl"] and utt in test_utts:
                            FN += 1
                            continue
                        caption = jsondata["caption"]
                        #caption = caption.replace("@", ", ")
                        captions = caption.split("@")
                        captions_new = []
                        for c in captions:
                            if c in audioset_filter_labels:
                                audioset_filter_labels[c] += 1
                            else:
                                captions_new.append(c)
                        if len(captions_new) == 0:
                            FN2 += 1
                            continue
                        caption = "".join(captions_new)
                        audio = jsondata["audio"]
                    self.audioset_inputs.append(caption)
                    self.audioset_audios.append(audio)
                    self.audioset_indices.append(N + index)
            main_process and print(part, "audioset", len(self.audioset_audios), "filtered", FN, FN2, audioset_filter_labels)
            self.inputs.extend(self.audioset_inputs)
            self.audios.extend(self.audioset_audios)
            self.indices.extend(self.audioset_indices)
            N = len(self.audios)
            main_process and print(part, "audioset audios", len(self.audios))

        if bbc_soundeffects2 and part in ["train", "train_val_audioset_sl"]:
            self.bbc_soundeffects2_inputs = []
            self.bbc_soundeffects2_audios = []
            self.bbc_soundeffects2_indices = []
            FN = 0
            with open("/ckptstorage/zhanghaomin/BBCSoundEffects/bbcsoundeffects_train.json", "r") as fr:
                for index, line in enumerate(fr.readlines()):
                    jsondata = json.loads(line.strip())
                    utt = jsondata["id"]
                    if part in ["train", "train_val_audioset_sl"] and utt in bbc_soundeffects_utts:
                        FN += 1
                        continue
                    caption = jsondata["caption"]
                    caption = caption.split("(")[0].strip()
                    audio = jsondata["audio"]
                    self.bbc_soundeffects2_inputs.append(caption)
                    self.bbc_soundeffects2_audios.append(audio)
                    self.bbc_soundeffects2_indices.append(N + index)
            main_process and print(part, "bbc_soundeffects2", len(self.bbc_soundeffects2_audios), "filtered", FN)
            self.inputs.extend(self.bbc_soundeffects2_inputs)
            self.audios.extend(self.bbc_soundeffects2_audios)
            self.indices.extend(self.bbc_soundeffects2_indices)
            N = len(self.audios)
            main_process and print(part, "bbc_soundeffects2 audios", len(self.audios))
        
        if freesound2 and part in ["train", "train_val_audioset_sl"]:
            self.freesound2_inputs = []
            self.freesound2_audios = []
            self.freesound2_indices = []
            FN = 0
            with open("/ckptstorage/zhanghaomin/CLAP_freesound/freesound_train.json", "r") as fr:
                for index, line in enumerate(fr.readlines()):
                    jsondata = json.loads(line.strip())
                    utt = jsondata["id"]
                    if part in ["train", "train_val_audioset_sl"] and utt in freesound_utts:
                        FN += 1
                        continue
                    caption = jsondata["caption"]
                    caption = caption.split('"@$&#"')
                    #caption = caption[0].split("(")[0].strip()
                    caption = tuple([c.split("(")[0].strip() for c in caption])
                    audio = jsondata["audio"]
                    self.freesound2_inputs.append(caption)
                    self.freesound2_audios.append(audio)
                    self.freesound2_indices.append(N + index)
            main_process and print(part, "freesound2", len(self.freesound2_audios), "filtered", FN)
            self.inputs.extend(self.freesound2_inputs)
            self.audios.extend(self.freesound2_audios)
            self.indices.extend(self.freesound2_indices)
            N = len(self.audios)
            main_process and print(part, "freesound2 audios", len(self.audios))

        if tangopromptbank and part in ["train", "train_val_audioset_sl"]:
            self.tangopromptbank_inputs = []
            self.tangopromptbank_audios = []
            self.tangopromptbank_indices = []
            with open("/ailab-train/speech/zhanghaomin/scps/TangoPromptBank/data.json", "r") as fr:
                for index, line in enumerate(fr.readlines()):
                    jsondata = json.loads(line.strip())
                    caption = jsondata["captions"]
                    audio = jsondata["location"]
                    self.tangopromptbank_inputs.append(caption)
                    self.tangopromptbank_audios.append(audio)
                    self.tangopromptbank_indices.append(N + index)
            main_process and print(part, "tangopromptbank", len(self.tangopromptbank_audios))
            self.inputs.extend(self.tangopromptbank_inputs)
            self.audios.extend(self.tangopromptbank_audios)
            self.indices.extend(self.tangopromptbank_indices)
            N = len(self.audios)
            main_process and print(part, "tangopromptbank audios", len(self.audios))
        
        if musiccaps and part in ["train", "train_val_audioset_sl"]:
            self.musiccaps_inputs = []
            self.musiccaps_audios = []
            self.musiccaps_indices = []
            with open("/ailab-train/speech/zhanghaomin/scps/musiccap/musiccaps.jsonl", "r") as fr:
                for index, line in enumerate(fr.readlines()):
                    jsondata = json.loads(line.strip())
                    caption = jsondata["caption"]
                    audio = jsondata["audio"]
                    self.musiccaps_inputs.append(caption)
                    self.musiccaps_audios.append(audio)
                    self.musiccaps_indices.append(N + index)
            main_process and print(part, "musiccaps", len(self.musiccaps_audios))
            self.inputs.extend(self.musiccaps_inputs)
            self.audios.extend(self.musiccaps_audios)
            self.indices.extend(self.musiccaps_indices)
            N = len(self.audios)
            main_process and print(part, "musiccaps audios", len(self.audios))
        
        if (audioset_sl_2ch and part in ["train", "train_val_audioset_sl"]) or (part == "val_audioset_sl_2ch"):
            self.audioset_sl_2ch_inputs = []
            self.audioset_sl_2ch_audios = []
            self.audioset_sl_2ch_indices = []
            audioset_sl_2ch_train = "/ckptstorage/zhanghaomin/audiosetsl/wavs/train.jsonl"
            audioset_sl_2ch_val = "/ckptstorage/zhanghaomin/audiosetsl/wavs/test.jsonl"
            if part in ["train", "train_val_audioset_sl"]:
                audioset_sl_2ch_path = audioset_sl_2ch_train
            else:
                audioset_sl_2ch_path = audioset_sl_2ch_val
            with open(audioset_sl_2ch_path, "r") as fr:
                for index, line in enumerate(fr.readlines()):
                    jsondata = json.loads(line.strip())
                    caption = jsondata["caption"]
                    audio = jsondata["audio"]
                    self.audioset_sl_2ch_inputs.append(caption)
                    self.audioset_sl_2ch_audios.append(audio)
                    self.audioset_sl_2ch_indices.append(N + index)
            main_process and print(part, "audioset_sl_2ch", len(self.audioset_sl_2ch_audios))
            self.inputs.extend(self.audioset_sl_2ch_inputs)
            self.audios.extend(self.audioset_sl_2ch_audios)
            self.indices.extend(self.audioset_sl_2ch_indices)
            N = len(self.audios)
            main_process and print(part, "audioset_sl_2ch audios", len(self.audios))
        
        if (boom_epic and part in ["train", "train_val_audioset_sl"]) or (part == "val_boom_epic"):
            self.boom_epic_inputs = []
            self.boom_epic_audios = []
            self.boom_epic_indices = []
            #boom_epic_train = "/ckptstorage/zhanghaomin/giantsoundeffects/train_animals_mixture2.jsonl"
            #boom_epic_val = "/ckptstorage/zhanghaomin/giantsoundeffects/test_animals_mixture2.jsonl"
            boom_epic_train = "/ailab-train/speech/zhanghaomin/scps/giantsoundeffects/train.jsonl"
            boom_epic_val = "/ailab-train/speech/zhanghaomin/scps/giantsoundeffects/test.jsonl"
            if part in ["train", "train_val_audioset_sl"]:
                boom_epic_path = boom_epic_train
            else:
                boom_epic_path = boom_epic_val
            with open(boom_epic_path, "r") as fr:
                for index, line in enumerate(fr.readlines()):
                    jsondata = json.loads(line.strip())
                    caption = jsondata["caption"]
                    audio = jsondata["audio"]
                    self.boom_epic_inputs.append(caption)
                    self.boom_epic_audios.append(audio)
                    self.boom_epic_indices.append(N + index)
            main_process and print(part, "boom_epic", len(self.boom_epic_audios))
            repeats = 1
            for _ in range(repeats):
                self.inputs.extend(self.boom_epic_inputs)
                self.audios.extend(self.boom_epic_audios)
                self.indices.extend(self.boom_epic_indices)
            N = len(self.audios)
            main_process and print(part, "boom_epic audios", len(self.audios))
        self.boom_epic = boom_epic
        
        if vggsound:
            self.inputs_vggsound = []
            self.audios_vggsound = []
            self.indices_vggsound = []
            if part in ["train", "train_val_audioset_sl"]:
                path = "/ailab-train/speech/zhanghaomin/scps/VGGSound/train.scp"
                with open(path, "r") as fr:
                    for index, line in enumerate(fr.readlines()):
                        video_path, text = line.strip().split("\t")
                        self.inputs_vggsound.append("the sound of " + text.strip().replace("(", "").replace(")", ""))
                        self.audios_vggsound.append(video_path)
                        self.indices_vggsound.append(index)
                N = len(self.audios_vggsound)
                print(part, "vggsound train audios", len(self.audios_vggsound), device_id, num_processes)
            elif part == "val_vggsound":
                if vgg_test is not None:
                    path = vgg_test[0]
                    start = vgg_test[1]
                    end = vgg_test[2]
                    step = vgg_test[3]
                else:
                    path = "/ailab-train/speech/zhanghaomin/scps/VGGSound/test.scp"
                    start = 0
                    end = 200
                    step = 1
                with open(path, "r") as fr:
                    for index, line in enumerate(fr.readlines()[start:end:step]):
                        video_path, text = line.strip().split("\t")
                        self.inputs.append("the sound of " + text.strip().replace("(", "").replace(")", ""))
                        self.audios.append(video_path)
                        self.indices.append(N + index)
                N = len(self.audios)
                print(part, "vggsound eval audios", len(self.audios), device_id, num_processes)
        
        if instruments:
            self.inputs_instruments = []
            self.audios_instruments = []
            self.indices_instruments = []
            if part in ["train", "train_val_audioset_sl"]:
                ####path = "/ailab-train/speech/zhanghaomin/scps/instruments/train.scp"
                ####path = "/ailab-train/speech/zhanghaomin/scps/instruments/piano_2h/train.scp"
                path = "/ailab-train/speech/zhanghaomin/scps/instruments/piano_20h/v2a_giant_piano2/train.scp"
                with open(path, "r") as fr:
                    for index, line in enumerate(fr.readlines()):
                        video_path, text = line.strip().split("\t")
                        self.inputs_instruments.append("the sound of " + text.strip().replace("(", "").replace(")", ""))
                        self.audios_instruments.append(video_path)
                        self.indices_instruments.append(index)
                N = len(self.audios_instruments)
                print(part, "instruments train audios", len(self.audios_instruments), device_id, num_processes)
            elif part == "val_instruments":
                if vgg_test is not None:
                    path = vgg_test[0]
                    start = vgg_test[1]
                    end = vgg_test[2]
                    step = vgg_test[3]
                else:
                    ####path = "/ailab-train/speech/zhanghaomin/scps/instruments/test.scp"
                    ####path = "/ailab-train/speech/zhanghaomin/scps/instruments/piano_2h/test.scp"
                    path = "/ailab-train/speech/zhanghaomin/scps/instruments/piano_20h/v2a_giant_piano2/test.scp"
                    start = 0
                    end = 200
                    step = 1
                with open(path, "r") as fr:
                    for index, line in enumerate(fr.readlines()[start:end:step]):
                        video_path, text = line.strip().split("\t")
                        self.inputs.append("the sound of " + text.strip().replace("(", "").replace(")", ""))
                        self.audios.append(video_path)
                        self.indices.append(N + index)
                N = len(self.audios)
                print(part, "instruments eval audios", len(self.audios), device_id, num_processes)
        
        self.vggsound = vggsound
        self.instruments = instruments
        self.video_drop_prompt = video_drop_prompt
        self.audio_drop_prompt = audio_drop_prompt
        self.device_id = device_id
        self.num_processes = num_processes
        self.bad_ids = {}
        self.bad_ids_instruments = {}
        self.video_encoder = video_encoder
        self.val_length = val_length if val_length is not None else torch_tools.MAX_TARGET_LEN
        print("val_length", self.val_length)
        
        #self.mapper = {}
        #for index, audio, text in zip(self.indices, self.audios, self.inputs):
        #    self.mapper[index] = [audio, text]

        if num_examples != -1:
            self.inputs, self.audios = self.inputs[:num_examples], self.audios[:num_examples]
            self.indices = self.indices[:num_examples]
        
        self.samples = samples
        self.stft = stft
        self.target_length = DURATION
        self.augment = augment
        self.part = part
        self.main_process = main_process
        self.SCORE_THRESHOLD_TRAIN = SCORE_THRESHOLD_TRAIN
        self.theta = theta
        self.multi = 4

    def __len__(self):
        return len(self.inputs)

    def get_num_instances(self):
        return len(self.inputs)

    def __getitem__(self, index):
        s1, s2, s3 = self.inputs[index], self.audios[index], self.indices[index]
        return s1, s2, s3

    def read_audio_from_video(self, video_path):
        if video_path.startswith("/ailab-train/speech/zhanghaomin/VGGSound/"):
            audio_path = video_path.replace("/video/", "/audio/").replace(".mp4", ".wav")
        else:
            audio_path = video_path.replace(".mp4", ".generated.wav")
        if os.path.exists(audio_path):
            #print("video wav exist", audio_path)
            waveform, sr = torchaudio.load(audio_path)
        else:
            #print("video wav not exist", video_path)
            try:
                clip = AudioFileClip(video_path)
                sound_array = np.array(list(clip.iter_frames()))
                waveform = torch.from_numpy(sound_array).transpose(0,1).to(torch.float32)
                waveform = waveform[0:1, :]
                if clip.fps != torch_tools.new_freq:
                    waveform = torchaudio.functional.resample(waveform, orig_freq=clip.fps, new_freq=torch_tools.new_freq)
                waveform = torch_tools.normalize_wav(waveform)
                torchaudio.save(audio_path, waveform, torch_tools.new_freq)
            except:
                print("Error read_audio_from_video", audio_path)
                traceback.print_exc()
                return None
        return waveform

    def collate_fn(self, data):
        # 452463+1471396->452463+3430704->452463+2978587 more 452463+1037241+15973+310169 real 1183416+2000
        # theta (1183416)*0.5/(452463+1037241+15973+310169)=0.3259
        # beta (452463+1037241+15973+310169+3430704)/(452463+1037241+15973+310169+1471396)=1.5960 (452463+1037241+15973+310169+2978587)/(452463+1037241+15973+310169+1471396)=1.4585
        if self.part in ["train", "train_val_audioset_sl"]:
            val = False
        else:
            val = True
        if self.audioset_sl_2ch:
            nch = 2
        else:
            nch = 1
        while True:
            if self.part in ["train", "train_val_audioset_sl"]:
                #print("data raw", len(data), data[0])
                #data_sampled = random.sample(data, self.samples)
                
                if (self.soundeffects or self.boom_epic) and self.theta > 0:
                    data_len = len(data)
                    data_1 = []
                    data_2 = []
                    for sample in data:
                        if isinstance(sample[1], tuple):
                            if sample[1][0].startswith("/radiostorage/"):
                                prefix = "/".join(sample[1][0].split("/")[:3])
                            else:
                                prefix = "/".join(sample[1][0].split("/")[:4])
                        else:
                            if sample[1].startswith("/radiostorage/"):
                                prefix = "/".join(sample[1].split("/")[:3])
                            else:
                                prefix = "/".join(sample[1].split("/")[:4])
                        if torch_tools.SOUNDEFFECT[prefix]:
                            data_2.append(sample)
                        else:
                            data_1.append(sample)
                    #print("data splitted", len(data_1), len(data_2), float(len(data_1))/len(data_2))
                    data_len_1 = len(data_1)
                    data_len_2 = len(data_2)
                    if data_len_1 == 0 or data_len_2 == 0:
                        data_1_sampled = data_1
                        data_2_sampled = data_2
                    else:
                        data_len_1_sampled = int(data_len_2 / self.theta)
                        data_len_2_sampled = int(data_len_1 * self.theta)
                        if data_len_1_sampled < data_len_1:
                            data_1_sampled = random.sample(data_1, data_len_1_sampled)
                            data_2_sampled = data_2
                        else:
                            data_1_sampled = data_1
                            data_2_sampled = random.sample(data_2, data_len_2_sampled)
                    #print("data sampled", len(data_1_sampled), len(data_2_sampled), float(len(data_1_sampled))/len(data_2_sampled), self.samples*cand)
                    data_sampled = data_1_sampled
                    data_sampled.extend(data_2_sampled)
                    data_sampled = random.sample(data_sampled, min(self.samples*cand, len(data_sampled)))
                    #print("data sampled", len(data_sampled))
                else:
                    data_sampled = random.sample(data, min(self.samples*cand, len(data)))
                    #print("data sampled", len(data_sampled))
            else:
                data_sampled = data
            dat = pd.DataFrame(data_sampled)
            text, audios, indices = [dat[i].tolist() for i in dat]
            
            if self.vggsound and val and self.part in ["val_vggsound", "val_instruments"]:
                #print("vggsound val", len(audios), text)
                fbanks = []
                fbank_lens = []
                video_paths = []
                text_selected = []
                for audio, txt in zip(audios, text):
                    waveform = self.read_audio_from_video(audio)
                    if waveform is None:
                        continue
                    length = self.val_length
                    waveform = waveform[:, :length*torch_tools.hop_size]
                    fbank = self.stft(waveform).transpose(-1,-2)
                    fbanks.append(fbank)
                    fbank_lens.append(fbank.shape[1])
                    video_paths.append(audio)
                    text_selected.append(txt)
                    #print("stft", waveform.shape, fbank.shape)
                max_length = max(fbank_lens)
                for i in range(len(fbanks)):
                    if fbanks[i].shape[1] < max_length:
                        fbanks[i] = torch.cat([fbanks[i], torch.zeros(fbanks[i].shape[0], max_length-fbanks[i].shape[1], fbanks[i].shape[2])], 1)
                mel = torch.cat(fbanks, 0)
                mel_len = torch.Tensor(fbank_lens).to(torch.int32)
                break
            
            if_clap_filter = False
            if self.part in ["val_audiocaps", "val_audioset_sl_2ch", "val_boom_epic"]:
                if_clap_filter = False
            mel, text_selected, _, _, _, mel_len = torch_tools.wav_to_fbank(audios, text, self.samples, self.target_length, self.stft, val, if_clap_filter, self.main_process, self.SCORE_THRESHOLD_TRAIN, nch)
            if mel is not None:
                if self.part in ["train", "train_val_audioset_sl"]:
                    if len(text_selected) > self.samples:
                        mel = mel[:self.samples,...]
                        text_selected = text_selected[:self.samples]
                        #waveform = waveform[:self.samples,...]
                        mel_len = mel_len[:self.samples]
                if self.vggsound:
                    video_paths = [None] * len(text_selected)
                else:
                    video_paths = None
                #print("mel", mel.shape if mel is not None else None, len(text_selected) if text_selected is not None else 0, mel_len, video_paths)
                break
        
        #mel = mel.unsqueeze(1)
        if self.augment != 0 and len(text_selected) > 1 and (not val):
            aug_num = len(text_selected) if self.augment == -1 else self.augment
            # the last batch of the training data may have only one instance
            # we check the length here so that the augmentation function doesn't throw an error
            mixed_mel, _, _, mixed_captions, _, mixed_mel_len = torch_tools.augment_wav_to_fbank(audios, text, aug_num, self.target_length, self.stft, self.main_process, self.SCORE_THRESHOLD_TRAIN, nch)
            #print("mixed_mel", mixed_mel.shape if mixed_mel is not None else None, len(mixed_captions) if mixed_captions is not None else 0, mixed_mel_len)
            if mixed_mel is not None:
                if mel.shape[1] < mixed_mel.shape[1]:
                    mel = torch.cat([mel, torch.zeros(mel.shape[0], mixed_mel.shape[1]-mel.shape[1], mel.shape[2])], 1)
                elif mixed_mel.shape[1] < mel.shape[1]:
                    mixed_mel = torch.cat([mixed_mel, torch.zeros(mixed_mel.shape[0], mel.shape[1]-mixed_mel.shape[1], mixed_mel.shape[2])], 1)
                #mixed_mel = mixed_mel.unsqueeze(1)
                mel = torch.cat([mel, mixed_mel], 0)
                text_selected += mixed_captions
                mel_len = torch.cat([mel_len, mixed_mel_len], 0)
                if self.vggsound:
                    video_paths.extend([None] * len(mixed_captions))
                else:
                    video_paths = None
            #print("mel_final", mel.shape if mel is not None else None, len(text_selected) if text_selected is not None else 0, mel_len)
        
        if self.vggsound and (not val):
            video_paths = [None] * len(text_selected)
            fbanks = []
            fbank_lens = []
            audios = []
            video_captions = []
            indices = random.sample([self.indices_vggsound[i] for i in range(self.device_id, len(self.indices_vggsound), self.num_processes)], self.vggsound*10)
            indices_featured = []
            indices_nonfeatured = []
            for i in indices:
                if i in self.bad_ids:
                    continue
                if self.audios_vggsound[i].startswith("/ailab-train/speech/zhanghaomin/VGGSound/"):
                    if self.video_encoder == "clip_vit":
                        feature_path = self.audios_vggsound[i].replace("/video/", "/feature/").replace(".mp4", ".npz")
                    elif self.video_encoder == "clip_vit2":
                        feature_path = self.audios_vggsound[i].replace("/video/", "/feature_clip_vit2/").replace(".mp4", ".npz")
                    elif self.video_encoder == "clip_convnext":
                        feature_path = self.audios_vggsound[i].replace("/video/", "/feature_clip_convnext/").replace(".mp4", ".npz")
                    elif self.video_encoder == "dinov2":
                        feature_path = self.audios_vggsound[i].replace("/video/", "/feature_dinov2/").replace(".mp4", ".npz")
                    elif self.video_encoder == "mixed":
                        feature_path = self.audios_vggsound[i].replace("/video/", "/feature_mixed/").replace(".mp4", ".npz")
                    else:
                        raise Exception("Invalid video_encoder " + self.video_encoder)
                else:
                    if self.video_encoder == "clip_vit":
                        feature_path = self.audios_vggsound[i].replace(".mp4", ".generated.npz")
                    elif self.video_encoder == "clip_vit2":
                        feature_path = self.audios_vggsound[i].replace(".mp4", ".generated.clip_vit2.npz")
                    elif self.video_encoder == "clip_convnext":
                        feature_path = self.audios_vggsound[i].replace(".mp4", ".generated.clip_convnext.npz")
                    elif self.video_encoder == "dinov2":
                        feature_path = self.audios_vggsound[i].replace(".mp4", ".generated.dinov2.npz")
                    elif self.video_encoder == "mixed":
                        feature_path = self.audios_vggsound[i].replace(".mp4", ".generated.mixed.npz")
                    else:
                        raise Exception("Invalid video_encoder " + self.video_encoder)
                if os.path.exists(feature_path):
                    indices_featured.append(i)
                else:
                    indices_nonfeatured.append(i)
                    if len(indices_nonfeatured) >= self.vggsound:
                        break
            #print(self.device_id, self.bad_ids, indices, indices_featured, indices_nonfeatured)
            indices = indices_nonfeatured[:self.vggsound]
            if len(indices) < self.vggsound:
                indices.extend(indices_featured[:self.vggsound-len(indices)])
            for i in indices:
                waveform = self.read_audio_from_video(self.audios_vggsound[i])
                if waveform is None:
                    print("Error audio in video", i, self.audios_vggsound[i], self.bad_ids)
                    self.bad_ids[i] = 1
                    continue
                length = random.randint(torch_tools.MIN_TARGET_LEN, torch_tools.MAX_TARGET_LEN)
                waveform = waveform[:, :length*torch_tools.hop_size]
                fbank = self.stft(waveform).transpose(-1,-2)
                fbanks.append(fbank)
                fbank_lens.append(fbank.shape[1])
                audios.append(self.audios_vggsound[i])
                video_captions.append(self.inputs_vggsound[i])
                #print("stft", waveform.shape, fbank.shape)
            max_length = max(fbank_lens)
            for i in range(len(fbanks)):
                if fbanks[i].shape[1] < max_length:
                    fbanks[i] = torch.cat([fbanks[i], torch.zeros(fbanks[i].shape[0], max_length-fbanks[i].shape[1], fbanks[i].shape[2])], 1)
            video_mel = torch.cat(fbanks, 0)
            video_mel_len = torch.Tensor(fbank_lens).to(torch.int32)
            #print("video_mel", video_mel.shape if video_mel is not None else None, len(video_captions) if video_captions is not None else 0, video_mel_len)
            if video_mel is not None:
                if mel.shape[1] < video_mel.shape[1]:
                    mel = torch.cat([mel, torch.zeros(mel.shape[0], video_mel.shape[1]-mel.shape[1], mel.shape[2])], 1)
                elif video_mel.shape[1] < mel.shape[1]:
                    video_mel = torch.cat([video_mel, torch.zeros(video_mel.shape[0], mel.shape[1]-video_mel.shape[1], video_mel.shape[2])], 1)
                #video_mel = video_mel.unsqueeze(1)
                mel = torch.cat([mel, video_mel], 0)
                text_selected += video_captions
                mel_len = torch.cat([mel_len, video_mel_len], 0)
                video_paths.extend(audios)
            #print("mel_final", mel.shape if mel is not None else None, len(text_selected) if text_selected is not None else 0, mel_len, video_paths)
        
        if self.instruments and (not val):
            fbanks = []
            fbank_lens = []
            audios = []
            video_captions = []
            indices = random.sample([self.indices_instruments[i] for i in range(self.device_id, len(self.indices_instruments), self.num_processes)], self.instruments*10)
            indices_featured = []
            indices_nonfeatured = []
            for i in indices:
                if i in self.bad_ids_instruments:
                    continue
                if self.audios_instruments[i].startswith("/ailab-train/speech/zhanghaomin/VGGSound/"):
                    if self.video_encoder == "clip_vit":
                        feature_path = self.audios_instruments[i].replace("/video/", "/feature/").replace(".mp4", ".npz")
                    elif self.video_encoder == "clip_vit2":
                        feature_path = self.audios_instruments[i].replace("/video/", "/feature_clip_vit2/").replace(".mp4", ".npz")
                    elif self.video_encoder == "clip_convnext":
                        feature_path = self.audios_instruments[i].replace("/video/", "/feature_clip_convnext/").replace(".mp4", ".npz")
                    elif self.video_encoder == "dinov2":
                        feature_path = self.audios_instruments[i].replace("/video/", "/feature_dinov2/").replace(".mp4", ".npz")
                    elif self.video_encoder == "mixed":
                        feature_path = self.audios_instruments[i].replace("/video/", "/feature_mixed/").replace(".mp4", ".npz")
                    else:
                        raise Exception("Invalid video_encoder " + self.video_encoder)
                else:
                    if self.video_encoder == "clip_vit":
                        feature_path = self.audios_instruments[i].replace(".mp4", ".generated.npz")
                    elif self.video_encoder == "clip_vit2":
                        feature_path = self.audios_instruments[i].replace(".mp4", ".generated.clip_vit2.npz")
                    elif self.video_encoder == "clip_convnext":
                        feature_path = self.audios_instruments[i].replace(".mp4", ".generated.clip_convnext.npz")
                    elif self.video_encoder == "dinov2":
                        feature_path = self.audios_instruments[i].replace(".mp4", ".generated.dinov2.npz")
                    elif self.video_encoder == "mixed":
                        feature_path = self.audios_instruments[i].replace(".mp4", ".generated.mixed.npz")
                    else:
                        raise Exception("Invalid video_encoder " + self.video_encoder)
                if os.path.exists(feature_path):
                    indices_featured.append(i)
                else:
                    indices_nonfeatured.append(i)
                    if len(indices_nonfeatured) >= self.instruments:
                        break
            #print(self.device_id, self.bad_ids_instruments, indices, indices_featured, indices_nonfeatured)
            indices = indices_nonfeatured[:self.instruments]
            if len(indices) < self.instruments:
                indices.extend(indices_featured[:self.instruments-len(indices)])
            #### dpo
            ####a_indices = []
            ####b_indices = []
            ####for i in indices[:len(indices)//2]:
            ####    path, name = self.audios_instruments[i].rsplit("/", 1)
            ####    assert(name[0] in ["a", "b"])
            ####    for j, p in enumerate(self.audios_instruments):
            ####        if p == path + "/a" + name[1:]:
            ####            a_indices.append(j)
            ####        elif p == path + "/b" + name[1:]:
            ####            b_indices.append(j)
            ####indices = a_indices + b_indices
            #### dpo
            for i in indices:
                waveform = self.read_audio_from_video(self.audios_instruments[i])
                if waveform is None:
                    print("Error audio in video instruments", i, self.audios_instruments[i], self.bad_ids_instruments)
                    self.bad_ids_instruments[i] = 1
                    continue
                length = random.randint(torch_tools.MIN_TARGET_LEN, torch_tools.MAX_TARGET_LEN)
                waveform = waveform[:, :length*torch_tools.hop_size]
                fbank = self.stft(waveform).transpose(-1,-2)
                fbanks.append(fbank)
                fbank_lens.append(fbank.shape[1])
                audios.append(self.audios_instruments[i])
                video_captions.append(self.inputs_instruments[i])
                #print("stft", waveform.shape, fbank.shape)
            max_length = max(fbank_lens)
            for i in range(len(fbanks)):
                if fbanks[i].shape[1] < max_length:
                    fbanks[i] = torch.cat([fbanks[i], torch.zeros(fbanks[i].shape[0], max_length-fbanks[i].shape[1], fbanks[i].shape[2])], 1)
            video_mel = torch.cat(fbanks, 0)
            video_mel_len = torch.Tensor(fbank_lens).to(torch.int32)
            #print("video_mel", video_mel.shape if video_mel is not None else None, len(video_captions) if video_captions is not None else 0, video_mel_len)
            if video_mel is not None:
                if mel.shape[1] < video_mel.shape[1]:
                    mel = torch.cat([mel, torch.zeros(mel.shape[0], video_mel.shape[1]-mel.shape[1], mel.shape[2])], 1)
                elif video_mel.shape[1] < mel.shape[1]:
                    video_mel = torch.cat([video_mel, torch.zeros(video_mel.shape[0], mel.shape[1]-video_mel.shape[1], video_mel.shape[2])], 1)
                #video_mel = video_mel.unsqueeze(1)
                mel = torch.cat([mel, video_mel], 0)
                text_selected += video_captions
                mel_len = torch.cat([mel_len, video_mel_len], 0)
                video_paths.extend(audios)
            #print("mel_final", mel.shape if mel is not None else None, len(text_selected) if text_selected is not None else 0, mel_len, video_paths)
        
        ####
        text_selected = [" ".join(text.split(" ")[:20]) for text in text_selected]
        
        ####
        if not val:
            ####T = 8
            T = 5
            ####T = 2
            text_selected = text_selected[-T:]
            mel = mel[-T:,...]
            video_paths = video_paths[-T:]
            mel_len = mel_len[-T:,...]
            self.video_drop_prompt = self.video_drop_prompt[-T:]
            self.audio_drop_prompt = self.audio_drop_prompt[-T:] if self.audio_drop_prompt is not None else None
        
        frames, midis = E2TTS.encode_video_frames(video_paths, mel.shape[1])
        return [text_selected, mel, video_paths, mel_len, self.video_drop_prompt, self.audio_drop_prompt, frames, midis]


class Text2SpeechDataset(Dataset):
    def __init__(self, samples=-1, stft=None, val=False):
        self.inputs = []
        self.audios = []
        self.indices = []

        train_scp = "/ckptstorage/zhanghaomin/docker/ximalaya/ximalaya_process/data_scp/train.json"
        test_scp = "/ckptstorage/zhanghaomin/docker/ximalaya/ximalaya_process/data_scp/test.json"
        scp = train_scp if not val else test_scp
        index = 0
        with open(scp, "r") as fr:
            for line in fr.readlines():
                data = json.loads(line.strip())
                wav = data["wav"]
                text = data["text"]
                if len(text) < 2:
                    continue
                self.inputs.append(text)
                self.audios.append(wav)
                self.indices.append(index)
                index += 1
        print("data size", len(self.inputs), val)
        self.samples = samples
        self.stft = stft
        self.sample_rate = 24000
        self.multi = 8
        self.val = val

    def __len__(self):
        return len(self.inputs)

    def get_num_instances(self):
        return len(self.inputs)

    def __getitem__(self, index):
        s1, s2, s3 = self.inputs[index], self.audios[index], self.indices[index]
        return s1, s2, s3

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        texts, audios, indices = [dat[i].tolist() for i in dat]
        
        fbanks = []
        fbank_lens = []
        text_selected = []
        for text, audio in zip(texts, audios):
            waveform, sr = torchaudio.load(audio)
            waveform = waveform[0:1, :]
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)
            waveform = torch_tools.normalize_wav(waveform)
            fbank = self.stft(waveform).transpose(-1,-2)
            #print("stft", waveform.shape, fbank.shape)
            if self.val:
                if waveform.shape[1] / float(self.sample_rate) < 2.0 or waveform.shape[1] / float(self.sample_rate) > 15.0:
                    continue
            else:
                if waveform.shape[1] / float(self.sample_rate) < 1.0 or waveform.shape[1] / float(self.sample_rate) > 20.0:
                    continue
            fbanks.append(fbank)
            fbank_lens.append(fbank.shape[1])
            text_selected.append(text)
            if self.samples > 0 and len(text_selected) >= self.samples:
                break
        if self.samples > 0 and len(text_selected) > self.samples:
            fbanks = fbanks[:self.samples]
            fbank_lens = fbank_lens[:self.samples]
            text_selected = text_selected[:self.samples]
        max_length = max(fbank_lens)
        for i in range(len(fbanks)):
            if fbanks[i].shape[1] < max_length:
                fbanks[i] = torch.cat([fbanks[i], torch.zeros(fbanks[i].shape[0], max_length-fbanks[i].shape[1], fbanks[i].shape[2])], 1)
        mel = torch.cat(fbanks, 0)
        mel_len = torch.Tensor(fbank_lens).to(torch.int32)
        return [text_selected, mel, None, mel_len, None]

