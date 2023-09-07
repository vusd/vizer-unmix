from pathlib import Path
import torch
import torchaudio
import numpy as np
import scipy
import librosa
import os
from openunmix import predict
from openunmix import data

import sys

if len(sys.argv) < 2:
    print("usage: vizer.py input.mp3 [outdir]")
    sys.exit(1)

infile = sys.argv[1]

if len(sys.argv) > 2:
    basename = sys.argv[2]
else:
    basename = os.path.splitext(os.path.basename(infile))[0]

if os.path.exists(basename):
    print("oops, output directory {} already exists - delete this first or change outdir".format(basename))
    sys.exit(1)

os.makedirs(basename)

# https://www.youtube.com/watch?v=DQLUygS0IAQ
# https://www.youtube.com/watch?v=J9gKyRmic20

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# audio,rate = librosa.load(infile, sr=44100, mono=False)
audio, rate = data.load_audio(infile)
print("Rate is ", rate)

estimates = predict.separate(audio=audio, targets=['vocals', 'drums', 'bass', 'other'], device=device, rate=rate)

# estimates['vocals'].shape = (11953152, 2)
# audio.shape = (2, 11954040)


# import soundfile as sf
# sf.available_formats()


# librosa.output.write_wav('out_west.wav', audio, rate)
# librosa.output.write_wav('out_west.mp3', audio, rate)

for target, estimate in estimates.items():
    target_path = str(basename / Path(target).with_suffix(".mp3"))
    torchaudio.save(
        target_path,
        torch.squeeze(estimate).to("cpu"),
        sample_rate=rate,
    )

volume_lists = [None] * 5

volume_lists[0] = 100 + librosa.core.amplitude_to_db(audio,top_db=100.0)
volume_lists[1] = 100 + librosa.core.amplitude_to_db(estimates['vocals'][0][0],top_db=100.0)
volume_lists[2] = 100 + librosa.core.amplitude_to_db(estimates['drums'][0][0],top_db=100.0)
volume_lists[3] = 100 + librosa.core.amplitude_to_db(estimates['bass'][0][0],top_db=100.0)
volume_lists[4] = 100 + librosa.core.amplitude_to_db(estimates['other'][0][0],top_db=100.0)

print("Estimates[vocals]")
print(estimates['vocals'].shape);

# 44100 to 60 -> 735
down_sample = rate / 60

import math
raw_len = volume_lists[1].shape[0]
num_windows = math.ceil(raw_len / down_sample)
# volume_list = [None] * num_windows
volumes = np.zeros([num_windows, 5])

print("NUM WIND is ", num_windows, " from ", raw_len)
print(volume_lists[1].shape)
print(estimates['vocals'].shape)

for i in range(num_windows):
	start_point = i * 735
	stop_point = (i+1) * 735
	window = volume_lists[1][start_point:stop_point]
	volumes[i][0] = np.mean(window)
	for j in range(1,5):
		window = volume_lists[j][start_point:stop_point]
		# print(i,j,start_point,stop_point,window)
		volumes[i][j] = np.mean(window)

with open('{}/volumes.csv'.format(basename), 'w') as f:
    for item in volumes:
        f.write("{:4.2f},{:4.2f},{:4.2f},{:4.2f},{:4.2f}\n".format(item[0],item[1],item[2],item[3],item[4]))
