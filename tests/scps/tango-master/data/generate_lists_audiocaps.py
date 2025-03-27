import os
import json


wav_path = "/zhanghaomin/datas/audiocaps/dataset/audioset/"

train_file = "/zhanghaomin/datas/audiocaps/dataset/metadata/audiocaps/datafiles/audiocaps_train_label.json"
test_file = "/zhanghaomin/datas/audiocaps/dataset/metadata/audiocaps/testset_subset/audiocaps_test_nonrepeat_subset_0.json"


wavs = {}
for fpath, dirs, fs in os.walk(wav_path):
    for f in fs:
        if f.endswith(".wav") or f.endswith(".WAV"):
            utt = f.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            assert(utt not in wavs)
            wavs[utt] = os.path.join(fpath, f)
print("wavs", len(wavs))


datas = {}
with open(train_file, "r") as fr:
    jsondata = json.loads(fr.read())
    print("train_file", len(jsondata["data"]))
    for data in jsondata["data"]:
        wav = data["wav"]
        caption = data["caption"]
        
        utt = wav.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        wav = wavs[utt]
        datas[utt] = (utt, wav, caption)

with open(test_file, "r") as fr:
    jsondata = json.loads(fr.read())
    print("test_file", len(jsondata["data"]))
    for data in jsondata["data"]:
        wav = data["wav"]
        caption = data["caption"]
        
        utt = wav.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        wav = wavs[utt]
        datas[utt] = (utt, wav, caption)


test_final = "/zhanghaomin/codes2/tango-master/data/test_audiocaps_subset.json"
test_utts = {}
test_utts_list = []
with open(test_final, "r") as fr:
    for line in fr.readlines():
        wav = json.loads(line.strip())["location"]
        utt = wav.rsplit("/", 1)[-1].rsplit("_", 1)[0]
        utt = "Y"+utt
        assert(utt not in test_utts)
        test_utts[utt] = 1
        test_utts_list.append(utt)


with open("/zhanghaomin/codes2/tango-master/data/audiocaps/train_audiocaps.json", "w") as fw:
    for utt, (utt, wav, caption) in datas.items():
        if utt not in test_utts:
            fw.write(json.dumps({"id": utt, "caption": caption, "audio": wav}) + "\n")


with open("/zhanghaomin/codes2/tango-master/data/audiocaps/test_audiocaps.json", "w") as fw:
    for utt, (utt, wav, caption) in datas.items():
        if utt in test_utts:
            fw.write(json.dumps({"id": utt, "caption": caption, "audio": wav}) + "\n")


os.system("mkdir audiocaps/test_wavs/")
for index, utt in enumerate(test_utts_list):
    utt, wav, caption = datas[utt]
    os.system("cp %s audiocaps/test_wavs/output_%s.wav" % (wav, str(index)))

