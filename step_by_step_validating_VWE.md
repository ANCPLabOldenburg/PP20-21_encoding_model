# Validating the Voxelwiseencoding Model
***

### This step-by-step guide provides guidance for running the ![voxelwiseencoding app](https://mjboos.github.io/voxelwiseencoding/)  by ![Moritz Boos](https://mjboos.github.io/) for the Modulation Power Spectrum and different parameters for stimulus length used and offset of the fMRI data. The Voxelwissencoding app allows to train voxelwiseendocing models on naturalistic stimuli.The output after this validation are Correlation Scores (Pearson Correlation) corrected for multiple correlation and thresholded to > 0 (to account for over-correction of the model). 

***
**Installing the Voxelwiseencoing app (install in Console, not in sypder/anaconda)**

pip install -e /your_datapath/voxelwiseencoding


# Requirements:  Naturalistic auditory data (speech) in BIDS compliant format. 

**Load Packages**

``` python
import json
from nilearn.masking import unmask
import nilearn
from nilearn.masking import apply_mask
from nilearn.plotting import plot_stat_map
from nilearn.image import mean_img
from nilearn import plotting 
import nibabel
import numpy as np
from nilearn.image import threshold_stats_img
from nilearn.input_data import MultiNiftiMasker
from nilearn.masking import compute_brain_mask
```

**Defining the paramters used in extracting the MPS**

```python
mel_params = {'n_mels': 64, 'sr': 44100, 'hop_length': 882, 'n_fft': 882, 'fmax': 8000,'mps_hop_length': 100, 'mps_n_fft':100}
with open('config.json', 'w+') as fl:
    json.dump(mel_params, fl)
```    

**Extracting Modulation Power Spectrum**

```python
!python audio2bidsstim/wav_files_to_bids_tsv_2.py /data/fg_audio/*.wav -c config.json
```

**Importing the model validation from the voxelwissencoding app**

```python
from voxelwiseencoding.process_bids import run_model_for_subject
```

**Define parameters which should be applied to BOLD data (these are the parameters used for preprocessing the BOLD fMRI files)**

```python
bold_prep_params = {'standardize': 'zscore', 'detrend': True}
```
# Important:
**This is the part where you can get "creative" and change the lag_time (stimulus length used to predict BOLD data in seconds) and offset_stim (offset of the fMRI data in relation to the stimulus presentation)**

![lag](https://user-images.githubusercontent.com/73650127/110309513-b8e0e080-8001-11eb-8bb9-5ce71bf4fc6d.png)

```python
lagging_params = {'lag_time': 6, 'offset_stim': 2}
```

**Defining the CV parameters(these are the parameters for sklearn's Ridge estimator, cv = number of folds used for CV)**

```python
ridge_params = {'alphas': [1e-1, 1, 100, 1000],'cv': 2, 'normalize': True}
```

**Run the validation and define the subject (here '01') and where it can be found (here '/data/your_path/aligned_mps/')**

```python
ridges, scores, computed_mask = run_model_for_subject('01', '/data/your_path/aligned_mps/',
                                                      task='aomovie', mask='epi', bold_prep_kwargs=bold_prep_params,
                                                      preprocess_kwargs=lagging_params, encoding_kwargs=ridge_params)
                                                      
```
![sub1](https://user-images.githubusercontent.com/73650127/110309543-c5653900-8001-11eb-8898-22b74bee4c92.png)

**(Optional) The output will be the Raw Correlation Scores which can be saved for later analyses**

```python
mps_scores01_6lag_2off = scores
```

**Correcting Scores for multiple correlation using bonferroni method**
