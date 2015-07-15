"""
=========================================

      Beat and downbeat estimation 
       based on pattern tracking

=========================================
"""

import ra
import os
import csv
import argparse
import scipy as sp
import numpy as np


# ============================================================================
#                         M A I N   P R O G R A M 
# ============================================================================

print(__doc__)

# ----------------------------------------------------------------------------
#                            ARGUMENTS PARSING 
# ----------------------------------------------------------------------------

# parser and arguments
parser = argparse.ArgumentParser(description='Extracts location of beats and downbeats using a rhythmic pattern tracking approach.')
parser.add_argument('input_file', metavar='in_file', nargs=1,
                   help='file name of audio file to be analyzed')
parser.add_argument('-v','--verbose', action="store_true", 
                   help='Verbose mode')
np.seterr(divide='ignore')
# parsing arguments
args = parser.parse_args()


# ----------------------------------------------------------------------------
#                             SET PARAMETERS  
# ----------------------------------------------------------------------------

# ---------------------------------------
#           Features parameters 
# ---------------------------------------
window_length = 20e-3 # window length
hop = 10e-3           # hop size
mel_flag = True       # If MEL filters should be used
nfilts = 160          # Number of MEL filters
log = False           # If LOG should be taken before taking differentiation
sum_flag = False      # Indicates if the subbands should be summed
freqs = [0, 200]      # cut-off frequency for summing low frequency band

# ---------------------------------------
#        Beat tracking parameters 
# ---------------------------------------
sigma_c = 2           # tolerance
sigma_o = 0.5         # std of observation model.
# pattern1 - piano pattern as in score 
pattern = [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0]
if args.verbose:
    print 'Selecting pattern: '
    print pattern

# ---------------------------------------
#        Input and output files 
# ---------------------------------------
audio_file = args.input_file[0]                       # audio filename
file_noext  = os.path.splitext(audio_file)[0]         # without extension
beats_output_file = file_noext + '_beats.csv'         # beats output file
downbeats_output_file = file_noext + '_downbeats.csv' # downbeats output file


# ----------------------------------------------------------------------------
#                          LOAD AUDIO SIGNAL
# ----------------------------------------------------------------------------
    
if args.verbose:
    print 'Loading audio signal ...'

signal = ra.data.AudioSignal(filename=audio_file, mix_opt='dm')

# ----------------------------------------------------------------------------
#                         COMPUTE FEATURES    
# ----------------------------------------------------------------------------

if args.verbose:
    print 'Computing features ...'

# features computation 
feature = ra.features.spectralFlux(signal, mel=mel_flag, log=log, sum=sum_flag,
                                   window_length=window_length, hop=hop, nfilts=nfilts)


# ----------------------------------------------------------------------------
#                         TEMPO ESTIMATION
# ----------------------------------------------------------------------------
    
if args.verbose:
    print 'Computing tempo ...'

# compute feature for tempo estimation
feature_tempo = ra.features.sumFeatures(feature)
# periodicity computation for estimation of period
per = ra.similarity.lm_dtft(feature_tempo, max_lag=0.80, min_lag=0.30, normalize=True)
# weighting function
per = ra.tempo.apply_weigth(per, weight_fun=ra.tempo.resonance_weight)
# compute period and beat per minute
lag_ind = per.data.argmax()
period = per.lag[lag_ind]
# compute beats per minute
beat_per = 60. / period 
    	
if args.verbose:
    print 'Estimated tempo is {:3.0f} BPM.'.format(beat_per)

# ----------------------------------------------------------------------------
#                      BEAT AND DOWNBEAT TRACKING 
# ----------------------------------------------------------------------------
  
# mel band indexes for summing low frequency band
lowidx = (np.abs(feature.feature_index - freqs[0])).argmin()
higidx = (np.abs(feature.feature_index - freqs[1])).argmin()
# sum within mel band indexes
feature_band = feature.data[range(lowidx,higidx),:].sum(axis=0)

# compute tatum period to normalize feature
period = period / 4	# added
period = int(round(period / hop)) - 1

# normalize feature
feature.data = ra.beat_track.normalize_features(feature_band, period * 8, p=8)

# create model
if args.verbose:
    print 'Creating model ...'
model = ra.beat_track.create_simple_pattern_model(len(pattern), period, sigma_c)
model.update()
model.likelihood_foo = ra.beat_track.gen_obs_model_simple_pattern(pattern, sigma_o)

# set uniform prior 
model.prior_foo = ra.beat_track.get_uniform_prior(model)

# run inference algorithm    
if args.verbose:
    print 'Running inference algorithm (please wait) ...'
[path, p] = ra.bayes_net.viterbi(feature.data, model)

# extract beat and downbeat position
if args.verbose:
    print 'Extracting beat and downbeat positions ...'
tatum_ind = [model.state_list[int(p)][0] == 0 for p in path]      # Places where the tatum was found
tt_ind = [(model.state_list[int(p)][1] % 4) == 0 for p in path]   # Pattern index
downbeat_ind = [(model.state_list[int(p)][1]) == 0 for p in path] # Pattern index
count_val = [model.state_list[int(p)][0] for p in path]           # Places where the tatum was found
pat_val = [model.state_list[int(p)][1] for p in path]             # Places where the tatum was found
tt_ind = sp.array(tt_ind)
downbeat_ind = sp.array(downbeat_ind)
beat_ind = (tt_ind * tatum_ind).nonzero()[0]
downbeat_ind = (tatum_ind * downbeat_ind).nonzero()[0]
beat_time =  (beat_ind / feature.fs) + feature.time_index[0]
downbeat_time = (downbeat_ind / feature.fs) + feature.time_index[0]


# ----------------------------------------------------------------------------
#                            SAVE RESULTS              
# ----------------------------------------------------------------------------

if args.verbose:
    print 'Saving results ...'
# save beats to output file
with open(beats_output_file, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
    srt_beat = ["{:2.3f}".format(num) for num in beat_time.tolist()]
    for beat in srt_beat:
	writer.writerow([beat])

# save downbeats to output file
with open(downbeats_output_file, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
    srt_downbeat = ["{:2.3f}".format(num) for num in downbeat_time.tolist()]
    for downbeat in srt_downbeat:
	writer.writerow([downbeat])


