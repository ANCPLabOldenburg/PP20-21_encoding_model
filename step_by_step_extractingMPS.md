# MPS Feature Extractor
***

### This is a function to extract the Modulation Power Spectrum based on the MEL Spectrogram with a 2D Fourier Transform from wav files. It is implemented in the script ![wav_files_to_bids_tsv_2.py](https://github.com/jannenold/practical_project_2020/blob/main/wav_files_to_bids_tsv_2.py) and should only be used within that script. This step-by-step Markdown aims to explain the individual steps only of the function mps_extract inside that script. The calculations are based on the paper by Elliot & Theunissen (2009) doi:10.1371/journal.pcbi.1000302
***

The output is stored in BIDS format. 



```python
def mps_extract(filename, sr = 44100, n_fft= 882, hop_length = 882, mps_n_fft = 100, 
                mps_hop_length = 100, n_mels = 64, plot_mps = True, **kwargs) 
```
    
### Input

- filename:        str, path to wav files to be converted
- sr:              int, sampling rate for wav file (*Default*: 44100 Hz)
- n_fft:           int, window size for mel spectrogram extraction (*Default*:882)
- hop_length:      int, step size for mel spectrogram extraction (*Default*: 882)
- mps_n_fft:       int, window size for mps extraction (*Default*: 100)
- mps_hop_length:  int, step size for mps extraction (*Default*: 100)
- n_mels:          int, number of mels used (*Default*: 64)
- plot_mps:        bool, plotting the mel spectrogram and mps forthe first window side by side (*Default*: True)
- kwargs:          additional keyword arguments that will be transferred to librosa's melspectrogram function

### Output

- tuple of a feature representation (2-dimensional array: samples x feature)
- repitition time in seconds: 
- names of all features (list of strings of mod/s for each mod/Hz):   
   
*Note*: Default settings are set so the windows for the extraction of the Mel spectrogram and the MPS each are non-overlapping. The values of 882 /100 for the window length and hop length were chose as a multiple of the sampling rates (Audio 44100 /Mel 100) to minimise rounding errors. 

**Load Packages**


```python
import numpy as np
import matplotlib.pyplot as plt
import librosa as lbr
import json
import os
import warnings            
import pandas as pd 
```

**Step 1.**

Extract wav files from directory


```python
wav, _ = lbr.load(filename, sr=sr) 
```

**Step 2.**

Extract MEL Spectogram from wav files


```python
mel_spec = lbr.feature.melspectrogram(y=wav, sr=sr, hop_length=hop_length,
                                              **kwargs)
                                                                                          
# Transpose Mel spectrogram for further analyses and compatibility
mel_spec = mel_spec.T
```
![mel](https://user-images.githubusercontent.com/73650127/110319807-af5e7500-800f-11eb-8f97-bc5bc1e399c6.png)

**Step 3.**

Check input parameters


```python
if mps_n_fft >= mel_spec.shape[0]:
    raise ValueError("The mps window size exceeds the Mel spectrogram. Please enter a smaller integer.")

if mps_hop_length >= mel_spec.shape[0]:
    raise ValueError("The mps step size exceeds the Mel spectrogram. Please enter a smaller integer.")
```

**Step 4.**

Extract MPS by looping through spectrogram with pre-set window size (mps_n_fft) and pre-set hop_length (mps_hop_length). Also extracting the Nyquist Frequency. mps_all will be converted to a numpy array. 


```python
mps_all = []
mps_plot = []
nyquist_mps = np.ceil(mel_spec.shape[1]/2)



for i in range(1,101):
    
    #Extract mps for predefined window
    mps = np.fft.fft2(mel_spec[mps_n_fft*(i-1):mps_n_fft*i,:])
   
    # use absoulte and shifted frequencies
    mps = np.abs(np.fft.fftshift(mps))
    
    # only take quarter of complete MPS (due to mirroring)
    mps = mps[int(mps_n_fft/2):,nyquist_mps:]
    
    # Define variable for later plotting
    mps_plot.append(mps)
   
    # Flattening the mps to a vector
    mps = np.reshape(mps,(1,np.size(mps)))
    
    # Append mps to mps all
    mps_all.append(mps)
    
# Convert mps_all into an array outside the loop
mps_all = np.array(mps_all)

# Convert mps_plot into an array outside loop
mps_plot = np.array(mps_plot)

# Concatinating the MPS row-wise
mps_all = np.concatenate(mps_all)
```

**Step 5.**

Extract Axes Labels 



```python

# Calculate the raw signal length
raw_length_sec = (len(wav)/sr)
raw_length_min = raw_length_sec/60

# Sampling Rate in Mel Spectrogram
fs_spectrogram = round(len(mel_spec)/(raw_length_sec))#if i roiund it the fs_spec will be 0 

# Sampling rate in MPS 
fs_mps = round(mps_n_fft/(raw_length_min))

# Extract Axes units for plotting 
# Calculate step sizes for MPS based on the logarithmic frequencies
mel_freqs = lbr.mel_frequencies(n_mels = n_mels, **{param: kwargs[param] for param in ['n_mels', 'fmin', 'fmax', 'htk'] if param in kwargs})
freq_step_log = np.log(mel_freqs[2]) - np.log(mel_freqs[1])

# Calculate labels for X and Y axes
mps_freqs = np.fft.fftshift(np.fft.fftfreq(mel_spec.shape[1], d = freq_step_log)) # returns fourier transformed freuqencies which are already shifted (lower freq in center))
mps_times = np.fft.fftshift(np.fft.fftfreq(mps_n_fft, d = 1. /fs_spectrogram)) 
```

**Step 6.**

Plot Mel Spectrogram of first window and according MPS next to each other

```python

if plot_mps:
       fig, (ax1,ax2)= plt.subplots(1, 2, figsize=(20, 10))
       
       # use only first window of Mel spectrogram to plot
       first_mel = mel_spec[0:mps_n_fft,:]
       
       #extract time and frequency axes
       time = np.arange(0,mps_n_fft)*fs_spectrogram
       frequency = np.arange(0,mel_spec.shape[1])*fs_mps
       
       # define first plot (Mel spectrgram)
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
       
       # define second plot (MPS for Mel spectrogram first window)
       image2 = ax2.imshow(np.log(mps_plot[0,:,:].T), origin = 'lower', aspect = 'auto')
       
       # use only half of the frequqecies (up to Niquist so the MPS is not mirrored)
       mps_freqs2 = mps_freqs[nyquist_mps:,]
       
       # use only the right side off the mirrored Y axis 
       mps_times2 = mps_times[int(mps_n_fft/2):,]
       
       ax2.set_xticks(np.arange(0,len(mps_times2),20))
       ax2.set_yticks(np.arange(0,len(mps_freqs2),8))
       x2= ax2.get_xticks()
       y2= ax2.get_yticks()
       ax2.set_xticklabels(['{:.0f}'.format(xtick2) for xtick2 in mps_times2[x2]])
       ax2.set_yticklabels(['{:.2f}'.format(ytick2) for ytick2 in mps_freqs2[y2]])
       ax2.set_title(' MPS for Mel Spectrogram (1st window)')
       ax2.set_xlabel('Temporal Modulation (mod/s)')
       ax2.set_ylabel('Spectral Modulation (cyc/oct)')
       cbar = fig.colorbar(image2, ax=ax2)
       cbar.set_label('(log) MPS')
    
```
![mps](https://user-images.githubusercontent.com/73650127/110319812-b08fa200-800f-11eb-8e60-8353a82b8e13.png)





**Step 6.**

Extract names of the features in the MPS


```python
names_features = ['{0:.2f} mod/s {1:.2f} cyc/oct)'.format(mps_time, mps_freq) 
                  for mps_time in mps_times for mps_freq in mps_freqs]
```

**Step 7.**

Determine the repitition time between two mps.

```python
stim_TR = fs_mps
```

**Step 8.**
Declare output. 

```python
return mps_all, stim_TR, names_features
```
