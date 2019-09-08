import torch
import numpy as np
import scipy
import librosa
import test
import os

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

audio,rate = librosa.load(infile, sr=44100, mono=False)

estimates = test.separate(audio=audio.T, targets=['vocals', 'drums', 'bass', 'other'], device=device, residual_model=False)

# estimates['vocals'].shape = (11953152, 2)
# audio.shape = (2, 11954040)


# import soundfile as sf
# sf.available_formats()


# librosa.output.write_wav('out_west.wav', audio, rate)
# librosa.output.write_wav('out_west.mp3', audio, rate)


librosa.output.write_wav('{}/out0_all.wav'.format(basename), audio, rate)
librosa.output.write_wav('{}/out1_vocals.wav'.format(basename), estimates['vocals'], rate)
librosa.output.write_wav('{}/out2_drums.wav'.format(basename), estimates['drums'], rate)
librosa.output.write_wav('{}/out3_bass.wav'.format(basename), estimates['bass'], rate)
librosa.output.write_wav('{}/out4_other.wav'.format(basename), estimates['other'], rate)

# https://stackoverflow.com/questions/3255674/convert-audio-files-to-mp3-using-ffmpeg
for file in ["out0_all", "out1_vocals", "out2_drums", "out3_bass", "out4_other"]:	
    command = "ffmpeg -i {0}/{1}.wav -vn -ar 44100 -ac 2 -b:a 192k {0}/{1}.mp3".format(basename, file)
    # command = "ffmpeg -i input.wav -vn -ar 44100 -ac 2 -b:a 192k output.mp3".format(bars_dir,bars_dir)
    os.system(command)

command = "rm {}/*.wav".format(basename)
os.system(command)

volume_lists = [None] * 5

volume_lists[0] = 100 + librosa.core.amplitude_to_db(audio,top_db=100.0)
volume_lists[1] = 100 + librosa.core.amplitude_to_db(estimates['vocals'],top_db=100.0)
volume_lists[2] = 100 + librosa.core.amplitude_to_db(estimates['drums'],top_db=100.0)
volume_lists[3] = 100 + librosa.core.amplitude_to_db(estimates['bass'],top_db=100.0)
volume_lists[4] = 100 + librosa.core.amplitude_to_db(estimates['other'],top_db=100.0)

# 44100 to 60 -> 735

import math
raw_len = volume_lists[1].shape[0]
num_windows = math.ceil(raw_len / 735)
# volume_list = [None] * num_windows
volumes = np.zeros([num_windows, 5])

for i in range(num_windows):
	start_point = i * 735
	stop_point = (i+1) * 735
	window = volume_lists[0][:,start_point:stop_point]
	volumes[i][0] = np.mean(window)
	for j in range(1,5):
		window = volume_lists[j][start_point:stop_point,:]
		# print(i,j,start_point,stop_point,window)
		volumes[i][j] = np.mean(window)

with open('{}/volumes.csv'.format(basename), 'w') as f:
    for item in volumes:
        f.write("{:4.2f},{:4.2f},{:4.2f},{:4.2f},{:4.2f}\n".format(item[0],item[1],item[2],item[3],item[4]))
