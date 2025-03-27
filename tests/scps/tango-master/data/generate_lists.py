import json
import random


#jsonfile = "/zhanghaomin/datas/audioset_sl/as_final.json"
#jsonfile = "/radiostorage/WavCaps/json_files/BBC_Sound_Effects/bbc_final.json"
#jsonfile = "/radiostorage/WavCaps/json_files/FreeSound/fsd_final.json"
jsonfile = "/radiostorage/WavCaps/json_files/SoundBible/sb_final.json"


with open(jsonfile, "r") as fr:
    jsondata = json.loads(fr.read())
    print(len(jsondata["data"]))
    print(jsondata["data"][0])
    
num_all = len(jsondata["data"])

datas = list(range(num_all))

random.seed(0)
random.shuffle(datas)

#val_data = datas[:1000]
#test_data = datas[:1000]
#train_data = datas[1000:]
train_data = datas


#{"id": "Yb0RFKhbpFJA.wav", "caption": "Wind and a man speaking are heard, accompanied by buzzing and ticking.", "audio": "wav_path", "duration": 10.0}
#{"description": "Greengrocer Cicada (Cyclochila Virens) - close-up stridulation, almost reaching full burst. Dog in distance. Recorded in captivity.", "category": "['Nature']", "caption": "A bug is stridulating. A dog is barking in the distance.", "id": "NHU05051026", "duration": 88.84353125, "audio": "wav_path", "download_link": "https://sound-effects-media.bbcrewind.co.uk/zip/NHU05051026.wav.zip"}
#{"id": "180913", "file_name": "UK Mello.wav", "href": "/people/Tempouser/sounds/180913/", "tags": ["Standard", "ringtone", "basic", "traditional"], "description": "Standard traditional basic ringtone, in mello tone.", "author": "Tempouser", "duration": 3.204375, "download_link": "https://freesound.org/people/Tempouser/sounds/180913/download/180913__tempouser__uk-mello.wav", "caption": "A traditional ringtone is playing.", "audio": "wav_path"}
#{"title": "Airplane Landing Airport", "description": "Large commercial airplane landing at an airport runway.", "author": "Daniel Simion", "href": "2219-Airplane-Landing-Airport.html", "caption": "An airplane is landing.", "id": "2219", "duration": 14.1424375, "audio": "wav_path", "download_link": "http://soundbible.com/grab.php?id=2219&type=wav"}


#with open("val_audioset_sl.json", "w") as fw:
#    for index in val_data:
#        fw.write(json.dumps(jsondata["data"][index]) + "\n")

#with open("test_audioset_sl.json", "w") as fw:
#    for index in test_data:
#        fw.write(json.dumps(jsondata["data"][index]) + "\n")

with open("train_soundbible.json", "w") as fw:
    for index in train_data:
        fw.write(json.dumps(jsondata["data"][index]) + "\n")

