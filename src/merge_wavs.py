import os


if False:
    prefix = "__ailab-train__speech__zhanghaomin__scps__instruments__piano_2h__piano_2h_cropped2_cuts__"
    
    
    #path = "/ailab-train/speech/zhanghaomin/codes3/e2-tts-pytorch-main/outputs2t_4_4000/"
    #new_path = "/ailab-train/speech/zhanghaomin/codes3/e2-tts-pytorch-main/outputs2t_4_4000_20s/"
    #new_path2 = "/ailab-train/speech/zhanghaomin/codes3/e2-tts-pytorch-main/outputs2t_4_4000_10s/"
    
    
    #path = "/ailab-train/speech/zhanghaomin/codes3/e2-tts-pytorch-main/outputs2t_4_2_8000/"
    #new_path = "/ailab-train/speech/zhanghaomin/codes3/e2-tts-pytorch-main/outputs2t_4_2_8000_20s/"
    #new_path2 = "/ailab-train/speech/zhanghaomin/codes3/e2-tts-pytorch-main/outputs2t_4_2_8000_10s/"
    
    
    path = "/ailab-train/speech/zhanghaomin/codes3/e2-tts-pytorch-main/outputs2t_ab/"
    new_path = "/ailab-train/speech/zhanghaomin/codes3/e2-tts-pytorch-main/outputs2t_ab_20s/"
    new_path2 = "/ailab-train/speech/zhanghaomin/codes3/e2-tts-pytorch-main/outputs2t_ab_10s/"
    
    
    os.mkdir(new_path)
    os.mkdir(new_path2)
    
    
    utt = "nwwHuxHMIpc"
    
    for index in range(13):
        os.system("sox %s %s %s"%(path+prefix+utt+"."+str(index*2).zfill(8)+".wav", path+prefix+utt+"."+str(index*2+1).zfill(8)+".wav", new_path+utt+"."+str(index+1)+".wav"))
    
    for index in range(27):
        os.system("cp %s %s"%(path+prefix+utt+"."+str(index).zfill(8)+".wav", new_path2+utt+"."+str(index+1)+".wav"))
    
    
    utt = "ra1jf2nzJPg"
    
    for index in range(12):
        os.system("sox %s %s %s"%(path+prefix+utt+"."+str(index*2).zfill(8)+".wav", path+prefix+utt+"."+str(index*2+1).zfill(8)+".wav", new_path+utt+"."+str(index+1)+".wav"))
    
    for index in range(24):
        os.system("cp %s %s"%(path+prefix+utt+"."+str(index).zfill(8)+".wav", new_path2+utt+"."+str(index+1)+".wav"))
    
    
    utt = "u5nBBJndN3I"
    
    for index in range(12):
        os.system("sox %s %s %s"%(path+prefix+utt+"."+str(index*2).zfill(8)+".wav", path+prefix+utt+"."+str(index*2+1).zfill(8)+".wav", new_path+utt+"."+str(index+1)+".wav"))
    
    for index in range(25):
        os.system("cp %s %s"%(path+prefix+utt+"."+str(index).zfill(8)+".wav", new_path2+utt+"."+str(index+1)+".wav"))




if True:
    from moviepy.editor import VideoFileClip, AudioFileClip
    
    
    prefix = "__ailab-train__speech__zhanghaomin__scps__instruments__piano_20h__v2a_giant_piano2__"


    #path = "/ailab-train/speech/zhanghaomin/codes3/e2-tts-pytorch-main/outputs2t_20h_base/"
    #new_path2 = "/ailab-train/speech/zhanghaomin/codes3/e2-tts-pytorch-main/outputs2t_20h_base_10s/"
    
    #path = "/ailab-train/speech/zhanghaomin/codes3/e2-tts-pytorch-main/outputs2t_20h_vel/"
    #new_path2 = "/ailab-train/speech/zhanghaomin/codes3/e2-tts-pytorch-main/outputs2t_20h_vel_10s/"
    
    path = "/ailab-train/speech/zhanghaomin/codes3/e2-tts-pytorch-main/outputs2t_20h_dpo/"
    new_path2 = "/ailab-train/speech/zhanghaomin/codes3/e2-tts-pytorch-main/outputs2t_20h_dpo_10s/"
    
    #path_gt = "/ailab-train/speech/zhanghaomin/codes3/e2-tts-pytorch-main/gt_20h_cut10/"


    os.mkdir(new_path2)
    
    #os.mkdir(path_gt)


    utt = "amazing_grace__aamazing_arace"

    for index in range(8):
        if not os.path.exists(path+prefix+utt+"_"+str(index+1)+".wav"):
            continue
        os.system("cp %s %s"%(path+prefix+utt+"_"+str(index+1)+".wav", new_path2+utt+"."+str(index+1)+".wav"))
        
        #audio = AudioFileClip(path+prefix+utt+"_"+str(index+1)+".mp4")
        #if audio.duration > 10.0:
        #    audio = audio.subclip(0, 10.0)
        #audio.write_audiofile(path_gt+utt+"."+str(index+1)+".wav", fps=24000, ffmpeg_params=["-ac", "1"])


    utt = "Mad_World__amad_world"

    for index in range(24):
        if not os.path.exists(path+prefix+utt+"_"+str(index+1)+".wav"):
            continue
        os.system("cp %s %s"%(path+prefix+utt+"_"+str(index+1)+".wav", new_path2+utt+"."+str(index+1)+".wav"))
        
        #audio = AudioFileClip(path+prefix+utt+"_"+str(index+1)+".mp4")
        #if audio.duration > 10.0:
        #    audio = audio.subclip(0, 10.0)
        #audio.write_audiofile(path_gt+utt+"."+str(index+1)+".wav", fps=24000, ffmpeg_params=["-ac", "1"])


    utt = "Minuet_In_G_Minor__aminuet_in_g_minor"

    for index in range(14):
        if not os.path.exists(path+prefix+utt+"_"+str(index+1)+".wav"):
            continue
        os.system("cp %s %s"%(path+prefix+utt+"_"+str(index+1)+".wav", new_path2+utt+"."+str(index+1)+".wav"))
        
        #audio = AudioFileClip(path+prefix+utt+"_"+str(index+1)+".mp4")
        #if audio.duration > 10.0:
        #    audio = audio.subclip(0, 10.0)
        #audio.write_audiofile(path_gt+utt+"."+str(index+1)+".wav", fps=24000, ffmpeg_params=["-ac", "1"])



