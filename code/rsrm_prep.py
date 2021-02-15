#!/usr/bin/env python
# coding: utf-8
# %%

# %%


# Import modules
import numpy as np
import mne
from wget import download
from os import remove
from pyprep.find_noisy_channels import NoisyChannels

# Dataset class that specifies file structure for different datasets
class Dataset:
    
    # Init method and get_file_format method need to be more general
    # Write a config file to ingest dataset specific information
    
    # Constructor assigns name, location, and size
    def __init__(self, name):
        '''Placeholder docstring (need to make more general)'''
        self.name = name
        
        # Data dictionary for all datasets: directory, subjects, tasks
        data_dict = {'motormovement_imagine': 
                         {'base_dir': 'https://github.com/VIXSoh/SRM/raw/master/data/',
                          'n_subj': 109,
                          'n_task': 14
                         }
                     # Add datasets here
                     # Write congig for different datasets
                    }
        # Assign attributes to dataset instance
        self.base_dir = data_dict[self.name]['base_dir']
        self.n_subj = data_dict[self.name]['n_subj']
        self.n_task = data_dict[self.name]['n_task']
    
    # Generates appropriate file paths
    def get_file_format(self, subj, task):
        '''Placeholder docstring (need to make more general)'''
        
        # Checks if name follows this file structure
        if self.name in ['motormovement_imagine']:
            
            # Get all file paths
            subj_num = f"S{str(subj).rjust(3, '0')}"
            task_num = f"R{str(task).rjust(2, '0')}"
            self.file_path = f'{self.base_dir}{subj_num}/{subj_num}{task_num}.edf'

    # Get data from internet using wget
    def wget_raw_edf(self):
        '''Reads an edf file from the internet (need to make more general for multiple file formats)

        Returns
        -------
        raw: mne raw object
            Raw dataset
        '''

        wg = download(self.file_path) # download the data locally (temporarily)
        print(wg)
        raw = mne.io.read_raw_edf(wg, preload=True) # read data as an MNE object
        remove(wg) # delete the file locally
        return raw

    # Specifies which files to iterate through
    def gen_iter(self, param, n_params):
        '''Helper method that allows for specific subject/task combinations or simply all of them

        Returns
        -------
        param_iter: list
            The iterator objects useful for subjects and tasks
        '''

        if param != 'all': # for subset of the data
            if type(param) != list:
                param_iter = [param]
            else:
                param_iter = param
        else: # for all of the data
            param_iter = range(1, n_params+1)
        return param_iter
    
    # Reads multiple EEG files with a single call
    def preproc(self, event_dict, baseline_start, stim_dur, montage, subjects='all', tasks='all', 
                hp_cutoff=1, lp_cutoff='none', line_noise=60, seed=42, eog_chan='none'):
        '''Preprocesses a single EEG file. Assigns a list of epoched data to Dataset instance,
        where each entry in the list is a subject with concatenated task data. Here is the basic 
        structure of the preprocessing workflow:
        
            - Set the montage
            - Band-pass filter (high-pass filter by default)
            - Automatically detect bad channels
            - Notch filter out line-noise
            - Reference data to average of all EEG channels
            - Automated removal of eye-related artifacts using ICA
            - Spherical interpolation of detected bad channels
            - Extract events and epoch the data accordingly
            - Align the events based on type (still need to implement this!)
            - Create a list of epoched data, with subject as the element concatenated across tasks
        
        Parameters
        ----------
        event_dict: dict
            Maps integers to semantic labels for events within the experiment
            
        baseline_start: int or float
            Specify start of the baseline period (in seconds)
            
        stim_dur: int or float
            Stimulus duration (in seconds)
                Note: may need to make more general to allow various durations
                
        montage: mne.channels.montage.DigMontage
            Maps sensor locations to coordinates
            
        subjects: list or 'all'
            Specify which subjects to iterate through
            
        tasks: list or 'all'
            Specify which tasks to iterate through
            
        hp_cutoff: int or float
            The low frequency bound for the highpass filter in Hz
            
        line_noise: int or float
            The frequency of electrical noise to filter out in Hz
            
        seed: int
            Set the seed for replicable results
            
        eog_chan: str
            If there are no EOG channels present, select an EEG channel
            near the eyes for eye-related artifact detection
        '''

        missing = [] # initialize missing file list
        subj_iter = self.gen_iter(subjects, self.n_subj) # get subject iterator
        task_iter = self.gen_iter(tasks, self.n_task) # get task iterator

        # Iterate through subjects (initialize subject epoch list)
        epochs_subj = []
        for subj in subj_iter:

            # Iterate through tasks (initialize within-subject task epoch list)
            epochs_task = []
            for task in task_iter:
                # Specify the file format
                self.get_file_format(subj, task)

                try: # Handles missing files
                    raw = self.wget_raw_edf() # read
                except:
                    print(f'---\nThis file does not exist: {self.file_path}\n---')
                    # Need to write the missing file list out
                    missing.append(self.file_path)
                    break

                # Set montage and seed
                raw.set_montage(montage=mont)
                np.random.seed(seed)

                # Apply high-pass filter
                print(f'Apply high-pass filter at {hp_cutoff} Hz')
                raw.filter(l_freq=hp_cutoff, h_freq=None, picks=['eeg', 'eog'], verbose=False)

                # Instantiate NoisyChannels object
                noise_chans = NoisyChannels(raw, do_detrend=False)

                # Detect bad channels through multiple methods
                noise_chans.find_bad_by_nan_flat()
                noise_chans.find_bad_by_deviation()
                noise_chans.find_bad_by_SNR()

                # Set the bad channels in the raw object
                raw.info['bads'] = noise_chans.get_bads()
                print(f'Bad channels detected: {noise_chans.get_bads()}')

                # Define the frequencies for the notch filter (60Hz and its harmonics)
                notch_filt = np.arange(line_noise, raw.info['sfreq'] // 2, line_noise)

                # Apply notch filter
                print(f'Apply notch filter at {line_noise} Hz and its harmonics')
                raw.notch_filter(notch_filt, picks=['eeg', 'eog'], verbose=False)

                # Reference to the average of all the good channels 
                # Automatically excludes raw.info['bads']
                raw.set_eeg_reference(ref_channels='average')

                # Instantiate ICA object
                ica = mne.preprocessing.ICA()
                # Run ICA
                ica.fit(raw)

                # Find which ICs match the EOG pattern
                if eog_chan == 'none':
                    eog_indices, eog_scores = ica.find_bads_eog(raw, verbose=False)
                else:
                    eog_indices, eog_scores = ica.find_bads_eog(raw, eog_chan, verbose=False)

                # Apply the IC blink removals (if any)
                ica.apply(raw, exclude=eog_indices)
                print(f'Removed IC index {eog_indices}')

                # Interpolate bad channels
                raw.interpolate_bads()

                # Find events
                events = mne.events_from_annotations(raw)[0]

                # Epoch the data
                preproc_epoch = mne.Epochs(raw, events, tmin=baseline_start, tmax=stim_dur, 
                                    event_id=event_dict, event_repeated='error', 
                                    on_missing='ignore', preload=True)
                # Add to epoch list
                epochs_task.append(preproc_epoch)

            # Assuming some data exists for a subject
            # Concatenate epochs within subject
            concat_epoch = mne.concatenate_epochs(epochs_task)
            epochs_subj.append(concat_epoch)
        # Attaches a list with each entry corresponding to epochs for a subject
        self.epoch_list = epochs_subj
            
    def feature_engineer(self, step_size, freq_min, freq_max, num_freq_fams):
        
        '''Computes features and flattens epoch list into a matrix. Assings a list
        of matrices where each element is a subject and the matrix dimensionality is
        num_features x num_trials. Here is the feature engineering workflow:
        
            -Extract power using Morlet wavelets
        
        Parameters
        ----------
        step_size: int or float
            The size of the time-window within each epoch to compute features
            
        freq_min: int or float
            The minimum frequency to compute features for
            
        freq_max: int or float
            The maximum frequency to compute features for
                
        num_freq_fams: int
            The number of frequency families to compute
        '''

        # Define frequencies of interest (log-spaced)
        freqs = np.logspace(*np.log10([freq_min, freq_max]), num=num_freq_fams)
        # Get different number of cycles for each frequency
        n_cycles = freqs / 2.

        # Iterate through subjects
        subj_mats=[]
        for idx, subj_data in enumerate(self.epoch_list):
            print(f'----- Feature engineering subject {idx} -----')
            # Compute power with Morlet wavelets
            power = mne.time_frequency.tfr_morlet(subj_data, freqs=freqs, 
                               n_cycles=n_cycles, use_fft=True,
                               return_itc=False, n_jobs=1, average=False)

            # Iterate through time-windows
            for i in np.arange(0, max(subj_data.times), step_size):
                # Create a copy of the power object to crop at time-window boundaries
                cropped = power.copy()
                # Crop the data for time-window of interest
                step_data = cropped.crop(tmin=i, tmax=i+step_size)

                # Retain shape after averaging (add 3rd axis back)
                step_avg = np.expand_dims(
                    # Average along the samples within the time-window
                    np.average(step_data.data, axis=3), 
                        axis=3)
                # Stack the time-windowed arrays along the 3rd axis
                if i == 0:
                    step_stack = step_avg
                else:
                    step_stack = np.concatenate((step_stack, step_avg), axis=3)

            shp = step_stack.shape
            # Not sure if this is the exact reshaping we want
            arr2mat = np.reshape(step_stack, (shp[0], shp[1]*shp[2]*shp[3]))
            subj_mats.append(arr2mat.T)
            print(f'Completed subject {idx}')

        self.feature_mats = subj_mats 

# This code isn't sufficient for building the montage:
dic_var = {
    "Fp1.":[-29.4367, 83.9171, -6.9900],
    "Fpz.":[0.1123, 88.2470, -1.7130],
    "Fp2.":[29.8723, 84.8959, -7.0800],
    "Af7.":[-54.8397, 68.5722, -10.5900],
    "Af3.":[-33.7007, 76.8371, 21.2270],
    "Afz.":[0.2313, 80.7710, 35.4170],
    "Af4.":[35.7123, 77.7259, 21.9560],
    "Af8.":[55.7433, 69.6568, -10.7550],
    "F7..":[-70.2629, 42.4743, -11.4200],
    "F5..":[-64.4658, 48.0353, 16.9210],
    "F3..":[-50.2438, 53.1112, 42.1920],
    "F1..":[-27.4958, 56.9311, 60.3420],
    "Fz..":[0.3122, 58.5120, 66.4620],
    "F2..":[29.5142, 57.6019, 59.5400],
    "F4..":[51.8362, 54.3048, 40.8140],
    "F6..":[67.9142, 49.8297, 16.3670],
    "F8..":[73.0431, 44.4217, -12.0000],
    "Ft7.":[-80.7750, 14.1203, -11.1350],
    "Fc5.":[-77.2149, 18.6433, 24.4600],
    "Fc3.":[-60.1819, 22.7162, 55.5440],
    "Fc1.":[-34.0619, 26.0111, 79.9870],
    "Fcz.":[0.3761, 27.3900, 88.6680],
    "Fc2.":[34.7841, 26.4379, 78.8080],
    "Fc4.":[62.2931, 23.7228, 55.6300],
    "Fc6.":[79.5341, 19.9357, 24.4380],
    "Ft8.":[81.8151, 15.4167, -11.3300],
    "T9..":[-85.8941, -15.8287, -48.283],
    "T7..":[-84.1611, -16.0187, -9.346],
    "C5..":[-80.2801, -13.7597, 29.1600],
    "C3..":[-65.3581, -11.6317, 64.3580],
    "C1..":[-36.1580, -9.9839, 89.7520],
    "Cz..":[0.4009, -9.1670, 100.2440],
    "C2..":[37.6720, -9.6241, 88.4120],
    "C4..":[67.1179, -10.9003, 63.5800],
    "C6..":[83.4559, -12.7763, 29.2080],
    "T8..":[85.0799, -15.0203, -9.4900],
    "T10.":[85.5599, -16.3613, -48.2710],
    "Tp7.":[-84.8302, -46.0217, -7.056],
    "Cp5.":[-79.5922, -46.5507, 30.9490],
    "Cp3.":[-63.5562, -47.0088, 65.6240],
    "Cp1.":[-35.5131, -47.2919, 91.3150],
    "Cpz.":[0.3858, -47.3180, 99.4320],
    "Cp2.":[38.3838, -47.0731, 90.6950],
    "Cp4.":[66.6118, -46.6372, 65.5800],
    "Cp6.":[83.3218, -46.1013, 31.2060],
    "Tp8.":[85.5488, -45.5453, -7.1300],
    "P7..":[-72.4343, -73.4527, -2.487],
    "P5..":[-67.2723, -76.2907, 28.3820],
    "P3..":[-53.0073, -78.7878, 55.9400],
    "P1..":[-28.6203, -80.5249, 75.4360],
    "Pz..":[0.3247, -81.1150, 82.6150],
    "P2..":[31.9197, -80.4871, 76.7160],
    "P4..":[55.6667, -78.5602, 56.5610],
    "P6..":[67.8877, -75.9043, 28.0910],
    "P8..":[73.0557, -73.0683, -2.5400],
    "Po7.":[-54.8404, -97.5279, 2.7920],
    "Po3.":[-36.5114, -100.8529, 37.1670],
    "Poz.":[0.2156, -102.1780, 50.6080],
    "Po4.":[36.7816, -100.8491, 36.3970],
    "Po8.":[55.6666, -97.6251, 2.7300],
    "O1..":[-29.4134, -112.4490, 8.8390],
    "Oz..":[0.1076, -114.8920, 14.6570],
    "O2..":[29.8426, -112.1560, 8.8000],
    "Iz..":[0.0045, -118.5650, -23.0780],
}

nsn = [0,88,-52]
l = [-85.8941, -15.8287, -52]
r = [85.5599, -16.3613, -52]
mont = mne.channels.make_dig_montage(ch_pos=dic_var, nasion=nsn, lpa=l, rpa=r)
mont

