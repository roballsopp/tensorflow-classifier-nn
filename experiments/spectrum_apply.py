import logging
import time
from argparse import ArgumentParser
import os

import tensorflow as tf
import nn
from matplotlib import pyplot as plt
import numpy as np

from load import Wave, WaveTF
from scipy.signal import get_window

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def build_fft_window(name, length, padded_len=None):
	padded_len = padded_len if padded_len is not None else length

	fft_window = tf.convert_to_tensor(get_window(name, length), dtype=tf.float32)
	# make it the same length as the signal
	fft_window = tf.pad(fft_window, [[0, padded_len - length]])
	# add channel dim to window
	fft_window = tf.expand_dims(fft_window, axis=0)
	return fft_window

parser = ArgumentParser()

parser.add_argument('--input-file', help="Specify input file path", required=True)
parser.add_argument('--spectrum-file', help="Specify spectrum file path", required=True)
parser.add_argument('--job-dir', type=str, default='./tmp/wav_out')
parser.add_argument('--job-name', type=str, default=time.strftime('%Y-%m-%d_%H-%M-%S'))


def get_fft_length(input_length):
	next_power_2 = int(np.ceil(np.log2(input_length)))
	return np.power(2, next_power_2)


def pad_to_length(x, out_length):
	x_len = x.shape[-1].value
	return tf.pad(x, [[0, 0], [0, out_length - x_len]])


def polar_to_rect(magnitude, phase):
	real = magnitude * tf.cos(phase)
	imag = magnitude * tf.sin(phase)

	return real, imag


def polar_to_complex(magnitude, phase):
	real, imag = polar_to_rect(magnitude, phase)
	return tf.complex(real, imag)

args = parser.parse_args()

input_filepath = args.input_file
spectrum_filepath = args.spectrum_file
job_dir = args.job_dir
job_name = args.job_name

logging.info('Loading input file ' + input_filepath + '...')
audio_wav = WaveTF.from_file(input_filepath)
spectrum_wav = WaveTF.from_file(spectrum_filepath)

audio_data = audio_wav.get_data()
audio_data_sr = audio_wav.sample_rate

spectrum_audio_data = spectrum_wav.get_data()
spectrum_sr = spectrum_wav.sample_rate

# linkin park start: 24, len: 10, no vox
# veil of maya start: 2, len: 10, vox
# awaiting winter: start: 10, len: 10, no nox

audio_len = audio_data.shape[-1].value
audio_fft_len = get_fft_length(audio_len)
audio_fft_in = pad_to_length(audio_data, audio_fft_len)
audio_fft = tf.spectral.rfft(audio_fft_in, fft_length=[audio_fft_len])

# slice out a good piece of the spectrum
spec_fft_in = spectrum_audio_data[:, spectrum_sr*24:spectrum_sr*34]
# pad it so we have an interpolated spectrum as large as the audio file one
spec_fft_in = pad_to_length(spec_fft_in, audio_fft_len)
spec_fft = tf.spectral.rfft(spec_fft_in, fft_length=[audio_fft_len])

spec_mag = tf.abs(spec_fft)
audio_mag = tf.abs(audio_fft)
audio_phase = tf.angle(audio_fft)

# halfway between the original spectrum and the desired spectrum
mag_out = (spec_mag * 0.5) + (audio_mag * 0.5)

out_complex = polar_to_complex(mag_out, audio_phase)

sig_out = tf.spectral.irfft(out_complex, fft_length=[audio_fft_len])
sig_out = nn.normalize(sig_out)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	sig_out = sess.run(sig_out)

	wav_out = Wave(sig_out, sample_rate=audio_data_sr)
	wav_out.to_file(os.path.join(job_dir, job_name + '_out.wav'))
