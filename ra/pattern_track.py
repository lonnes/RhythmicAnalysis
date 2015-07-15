import argparse
import scipy as sp
import ra
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Extracts beat information using rhythmic template tracking.')
parser.add_argument('input_file', metavar='in_file', nargs=1,
                   help='file name of audio file to be analyzed')
parser.add_argument('output_file', metavar='out_file', nargs=1,
                   help='file name of file where the beat positions are to be saved')
parser.add_argument('tatum_period', metavar='period', nargs=1, type=float,
                   help='tatum period in seconds')
parser.add_argument('-p', '--pattern', action="store", default='candombe',
                   help='{candombe, chico, clave, or piano} pattern, default is candombe')
parser.add_argument('--plot', action="store_true", 
                   help='True if the results are to be plotted.')
parser.add_argument('--uniform', action="store_true", 
                   help='If used,  uniform prior is used.')
parser.add_argument('-v','--verbose', action="store_true", 
                   help='Verbose mode')
# Parsing arguments
args = parser.parse_args()
# Loading signal
if args.verbose:
    print 'Loading signal.'
signal = ra.data.AudioSignal(filename=args.input_file[0])
if args.verbose:
    print 'Selecting pattern: '+args.pattern
if args.pattern == 'candombe':
    pattern = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
elif args.pattern == 'chico':
    pattern = [0, 1, 1, 1]
elif args.pattern == 'clave':
    pattern = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0]
elif args.pattern == 'piano':
    pattern = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
else:
    print 'Invalid pattern chose! Exiting.'
    exit()
if args.verbose:
    print 'Calculating features.'
# Features parameters
window_length = 40e-3
hop = 20e-3
mel_flag = True # If MEL filters should be used.
nfilts = 40 # Number of MEL filters
log = False # If LOG should be taken before taking differentiation
sum_flag = True # Indicates if the subbands should be summed
period = int(round(args.tatum_period[0] / hop))
# Calculating features
feature = ra.features.spectralFlux(signal, mel=mel_flag, log=log, sum=sum_flag,
                                   window_length=window_length, hop=hop, nfilts=nfilts)
# Normalizing features
feature.data = ra.beat_track.normalize_features(feature.data, period * 5, p=8)
# Creating model
if args.verbose:
    print 'Creating model.'
sigma_c = 1 # Tolerance of approximately 60 ms.
sigma_o = 0.25 # Std of observation model.
model = ra.beat_track.create_simple_pattern_model(len(pattern), period, sigma_c)
model.update()
model.likelihood_foo = ra.beat_track.gen_obs_model_simple_pattern(pattern, sigma_o)
if args.uniform:
    model.prior_foo = ra.beat_track.get_uniform_prior(model)
else:
    model.prior_foo = ra.beat_track.get_pattern_prior(model)
if args.verbose:
    print 'Running inference algorithm.'
[post, alpha, beta] = ra.bayes_net.forward_backward_algorithm(feature.data, model)
# Extracting beat position
if args.verbose:
    print 'Extracting beat positions and saving results.'
prob = [p.max() for p in post[1:]] # Probability of MAP estimate
tatum_ind = [model.state_list[p.argmax()][0] == 0 for p in post[1:]] # Places where the tatum was found
tt_ind = [(model.state_list[p.argmax()][1] % 4) == 0 for p in post[1:]] # Pattern index
tt_ind = sp.array(tt_ind)
beat_ind = (tt_ind * tatum_ind).nonzero()[0]
beat_time =  (beat_ind / feature.fs) + feature.time_index[0]
# Saving output file
sp.savetxt(args.output_file[0], beat_time, fmt='%2.3F', newline='\n')
# Plotting results if desired
if args.plot:
    plt.figure(1)
    plt.plot(feature.data)
    plt.plot(tatum_ind, 'g')
    plt.plot(beat_ind, sp.ones(beat_ind.shape), 'rx')
    plt.legend(['Norm. Features', 'Tatum', 'Beat'])
    plt.title('Detected tatum and beat')
    plt.xlabel('Frames (k)')
    plt.ylabel('Normalized Feature Value')
    plt.figure(2)
    plt.plot(prob)
    plt.xlabel('Frames')
    plt.ylabel('Probability')
    plt.title('Probability of chosen states')
    plt.show()

