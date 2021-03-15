import numpy as np
import joblib
import glob
import matplotlib.pyplot as plt
import json
import os
import librosa as lbr

def mps_extract(filename, sr = 44100, n_fft = 882, hop_length = 882, mps_n_fft = 100, mps_hop_length = 100, n_mels = 64, plot_mps = True, **kwargs):
    '''                
    Parameters
    ----------
    
    filename:       str, path to wav files to be converted
    sr:             int, sampling rate for wav file (Default: 44100 Hz)
    n_fft:          int, window size for mel spectrogram extraction (Default: 882)
    hop_length:     int, step size for mel spectrogram extraction (Default: 882)
    mps_n_fft:      int, window size for mps extraction (Default: 100)
    mps_hop_length: int, step size for mps extraction (Default: 100)
    n_mels:         int, numbers of mels used (Default: 64)
    plot_mps:       bool, if true the Mel spectrogram for the first window and according mps will be plotted (Default: False)
    kwargs:         additional keyword arguments that will be transferred to librosa's melspectrogram function
    
    Returns
    -------
    
    tuple of a feature representation (2-dimensional array: samples x feature)
    sampling rate
    names of all features (list of strings of mod/s for each mod/Hz)        
    
    '''
    wav, _ = lbr.load(filename, sr=sr) 
    mel_spec = lbr.feature.melspectrogram(y=wav, sr=sr, hop_length=hop_length, **kwargs) 
    mel_spec = mel_spec.T
     
    if mps_n_fft >= mel_spec.shape[0]: 
        raise ValueError("The mps window size exceeds the Mel spectrogram. Please enter a smaller integer.")
    if mps_hop_length >= mel_spec.shape[0]: 
        raise ValueError("The mps step size exceeds the Mel spectrogram. Please enter a smaller integer.")
     
    mps_all = []
    mps_plot = []
    n_hops = int((mel_spec.shape[0]/mps_hop_length)+1)
    nyquist_mps = int(np.ceil(mel_spec.shape[1]/2))
    for i in range(1,n_hops):
        mps = np.fft.fft2(mel_spec[mps_n_fft*(i-1):mps_n_fft*i,:])
        mps = np.abs(np.fft.fftshift(mps))
        mps = mps[int(mps_n_fft/2):,nyquist_mps:]
        mps_plot.append(mps)
        mps = np.reshape(mps,(1,np.size(mps)))
        mps_all.append(mps)
        
    mps_all = np.array(mps_all)
    mps_plot = np.array(mps_plot)
    mps_all = np.concatenate(mps_all)
   
    fs_spectrogram = sr/hop_length
    fs_mps = fs_spectrogram/mps_hop_length
    
    mel_freqs = lbr.mel_frequencies(n_mels = n_mels, **{param: kwargs[param] for param in ['n_mels', 'fmin', 'fmax', 'htk'] if param in kwargs})
    freq_step_log = np.log(mel_freqs[2]) - np.log(mel_freqs[1])
   
    mps_freqs = np.fft.fftshift(np.fft.fftfreq(mel_spec.shape[1], d = freq_step_log)) 
    mps_times = np.fft.fftshift(np.fft.fftfreq(mps_n_fft, d = 1. /fs_spectrogram)) 
   
    if plot_mps:
       fig, (ax1,ax2)= plt.subplots(1, 2, figsize=(20, 10)) 
       first_mel = mel_spec[0:mps_n_fft,:]
       time = np.arange(0,mps_n_fft)*fs_spectrogram
       frequency = np.arange(0,mel_spec.shape[1])*fs_mps
       image1 = ax1.imshow(first_mel.T, origin = 'lower', aspect = 'auto')
       ax1.set_xticks(np.arange(0,mps_n_fft,20))
       ax1.set_yticks(np.arange(0,first_mel.shape[1],10))
       x1= ax1.get_xticks()
       y1= ax1.get_yticks()
       ax1.set_xticklabels(['{:.0f}'.format(xtick) for xtick in time[x1]])
       ax1.set_yticklabels(['{:.2f}'.format(ytick) for ytick in frequency[y1]])
       ax1.set_title('Mel Spectrogram 1st window')
       ax1.set_ylabel('Frequencyband (Hz)')
       ax1.set_xlabel('Time (s)')
       cbar = fig.colorbar(image1, ax = ax1, format='%+2.0f dB')
       cbar.set_label('dB')
       image2 = ax2.imshow(np.log(mps_plot[0,:,:].T), origin = 'lower', aspect = 'auto')
       mps_freqs2 = mps_freqs[nyquist_mps:,]
       mps_times2 = mps_times[int(mps_n_fft/2):,]
       ax2.set_xticks(np.arange(0,len(mps_times2),20))
       ax2.set_yticks(np.arange(0,len(mps_freqs2),8))
       x2= ax2.get_xticks()
       y2= ax2.get_yticks()
       ax2.set_xticklabels(['{:.0f}'.format(xtick2) for xtick2 in mps_times[x2]])
       ax2.set_yticklabels(['{:.2f}'.format(ytick2) for ytick2 in mps_freqs2[y2]])
       ax2.set_title(' MPS for Mel Spectrogram (1st window)')
       ax2.set_xlabel('Temporal Modulation (mod/s)')
       ax2.set_ylabel('Spectral Modulation (cyc/oct)')
       cbar = fig.colorbar(image2, ax=ax2)
       cbar.set_label('(log) MPS')
   
   	# Extracting feature names                     
    names_features = ['{0:.2f} mod/s {1:.2f} cyc/oct)'.format(mps_time, mps_freq) for mps_time in mps_times2 for mps_freq in mps_freqs2]
   
   	# Determine MPS repitition time 
    stim_TR = fs_mps
                       
    return mps_all, stim_TR, names_features


def get_mel_spectrogram(filename, log=True, sr=44100, hop_length=512, **kwargs):
    '''Returns the (log) Mel spectrogram of a given wav file, the sampling rate of that spectrogram and names of the frequencies in the Mel spectrogram

    Parameters
    ----------
    filename : str, path to wav file to be converted
    sr : int, sampling rate for wav file
         if this differs from actual sampling rate in wav it will be resampled
    log : bool, indicates if log mel spectrogram will be returned
    kwargs : additional keyword arguments that will be
             transferred to librosa's melspectrogram function

    Returns
    -------
    a tuple consisting of the Melspectrogram of shape (time, mels), the repetition time in seconds, and the frequencies of the Mel filters in Hertz 
    '''
    wav, _ = lbr.load(filename, sr=sr)
    melspecgrams = lbr.feature.melspectrogram(y=wav, sr=sr, hop_length=hop_length, **kwargs)
    if log:
        melspecgrams[np.isclose(melspecgrams, 0)] = np.finfo(melspecgrams.dtype).eps
        melspecgrams = np.log(melspecgrams)
    log_dict = {True: 'Log ', False: ''}
    freqs = lbr.core.mel_frequencies(
            **{param: kwargs[param] for param in ['n_mels', 'fmin', 'fmax', 'htk']
                if param in kwargs})
    freqs = ['{0:.0f} Hz ({1}Mel)'.format(freq, log_dict[log]) for freq in freqs]
    return melspecgrams.T, sr / hop_length, freqs


if __name__ == '__main__':
    import argparse
    from itertools import cycle
    parser = argparse.ArgumentParser(description='Wav2bids stim converter.')
    parser.add_argument('file', help='Name of file or space separated list of files or glob expression for wav files to be converted.', nargs='+')
    parser.add_argument('-c' ,'--config', help='Path to json file that contains the parameters to librosa\'s melspectrogram function.')
    parser.add_argument('-o', '--output', help='Path to folder where to save tsv and json files, if missing uses current folder.')
    parser.add_argument('-t', '--start-time', help='Start time in seconds relative to first data sample.'
            ' Either a single float (same starting time for all runs) or a list of floats.', nargs='+', type=float, default=0.)
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as fl:
            config = json.load(fl)
    else:
        config = dict()
    if isinstance(args.file, str):
        args.file = [args.file]
    if len(args.file) == 1 and '*' in args.file[0]:
        args.file = glob.glob(args.file[0])
    if isinstance(args.start_time, float):
        args.start_time = [args.start_time]
    if len(args.start_time) > 1 and len(args.start_time) != len(args.file):
        raise ValueError('Number of files and number of start times are unequal. Start time has to be either one element or the same number as number of files.')
    for wav_file, start_time in zip(args.file, cycle(args.start_time)):
        melspec, sr_spec, freqs = mps_extract(wav_file, **config)
        tsv_file = os.path.basename(wav_file).split('.')[0] + '.tsv.gz'
        json_file = os.path.basename(wav_file).split('.')[0] + '.json'
        if args.output:
            tsv_file = os.path.join(args.output, tsv_file)
            json_file = os.path.join(args.output, json_file)
        np.savetxt(tsv_file, melspec, delimiter='\t')
        metadata = {'SamplingFrequency': sr_spec, 'StartTime': start_time,
                    'Columns': freqs}
        with open(json_file, 'w+') as fp:
            json.dump(metadata, fp)

