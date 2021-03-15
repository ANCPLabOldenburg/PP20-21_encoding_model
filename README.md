# PP20-21_encoding_model

The code was created as part of a Practical Project at the Applied Neurocognitive Psychology Lab/University of Oldenburg in 2020 - 2021 under the supervision of Moritz Boos and Dr. Arkan Al-Zubaidi.  The code provides an encoding model to predict fMRI data stored in BIDS format from natural speech. 

1) **Extracting** the Modulation Power Spectrum: This function extracts the Modulation Power Spectrum from an auditory stimulus in a BIDS compliant format.
The function is based on the MEL spectrogram and the 2D Fourier Transform. 
Use with `python wav_files_to_bids_tsv_2.py path/to/your/wavfiles/*.wav -c path/to/your/config.json` [Extracting MPS](https://github.com/jannenold/practical_project_2020/blob/main/wav_files_to_bids_tsv_2.py). By default extracts the MPS. 
For a fully commented script and step by step explanation please refer to the [step-by-step MPS Extraction](https://github.com/jannenold/practical_project_2020/blob/main/step_by_step_extractingMPS.md) here.

2) **Validate** the model: Use the Voxelwiseencoding App by Moritz Boos (for the script, please refer to [Validating Voxelwiseencoding Model](https://github.com/jannenold/practical_project_2020/blob/main/script_validating_VWE_MPS.py), for a step by step application, see [step-by-step Validating VWE](https://github.com/jannenold/practical_project_2020/blob/main/step_by_step_validating_VWE.md))

# Usage for extracting the MPS 
<pre> 
positional arguments:
-------
  filename :      str, path to wav files to be converted. Can be used with wildcard *.wav. 

keyword arguments:
------
  sr:             int, sampling rate of auditory files (samples per second: 44100 Hz by default)
  n_fft:          int, window length of spectrogram (default 882)
  mps_n_fft:      int, window length for extracting the MPS (default 100)
  hop_length:     int, step size for extracting MEL spectrogram (default 882)
  mps_hop_length: int, step size for extracting MPS (default 100)
  n_mels:         int, number of mels used (default 64)
  
  
optional arguments:
------
  plot_mps:       bool, plotting the mel spectrogram and mps forthe first window side by side (by default set to True)


</pre>

# Output

The function returns three outputs:

1. a representation of a feature matrix of shape (samples x features)
2. Stimulus Repetition Time (int)
3. the names of the features (as list of strings)

**Optional**

The function can return the plotted MPS. By default, this is set to True and has to be indicated if needed otherwise.

*Note*: Default settings are set so the windows for the extraction of the Mel spectrogram and the MPS each are non-overlapping.
