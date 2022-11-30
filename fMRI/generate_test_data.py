import argparse
import pickle
from pathlib import Path

import nibabel
import numpy as np
import scipy.ndimage as ndimage
import scipy.spatial.distance as sp_distance
import scipy.stats as stats
import sklearn.manifold as manifold
from brainiak.utils import fmrisim

num_class = 3

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='IGL xCI MNIST batch model')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed')
    parser.add_argument('--noise_weight', type=float, default=0, metavar='N',
                        help='noise weight')
    args = parser.parse_args()

    np.random.seed(args.seed)
    noise_weight = args.noise_weight

    """## Set parameters"""

    # load patterns

    with open('data/noise_dict.pkl', 'rb') as f:
        noise_dict = pickle.load(f)

    patterns_imagination = np.load('data/patterns_imagination.npy')


    # load human data

    # nii = nibabel.load('./drive/MyDrive/data/Corr_MVPA_Data_dataspace/Participant_01_rest_run01.nii')
    nii = nibabel.load('data/Participant_01_rest_run01.nii')
    volume = nii.get_fdata()

    # load some basic properties of human data? (size of the volume and the resolution of the voxels)

    dim = volume.shape  # What is the size of the volume
    dimsize = nii.header.get_zooms()  # Get voxel dimensions from the nifti header
    tr = dimsize[3]
    if tr > 100:  # If high then these values are likely in ms and so fix it
        tr /= 1000


    """1.3 Generate an activity template and a mask"""

    mask, template = fmrisim.mask_brain(volume=volume,
                                        mask_self=True,
                                        )

    """## Generate noise

    generate noise seperately for imagination, preception, and judgement

    noise of imagination is greater than preception
    """

    print('Calculating noise...')
    # Calculate the noise given the parameters
    noise_imagination = fmrisim.generate_noise(dimensions=dim[0:3],
                                tr_duration=int(tr),
                                stimfunction_tr=[0] * dim[3],
                                mask=mask,
                                template=template,
                                noise_dict=noise_dict,
                                )
    print('Noise calculation done...')


    # Create the different types of noise
    total_time = 500
    timepoints = list(range(0, total_time, int(tr)))

    drift = fmrisim._generate_noise_temporal_drift(total_time,
                                                int(tr),
                                                )

    mini_dim = np.array([2, 2, 2])
    autoreg = fmrisim._generate_noise_temporal_autoregression(timepoints,
                                                            noise_dict,
                                                            mini_dim,
                                                            np.ones(mini_dim),
                                                            )
                
    phys = fmrisim._generate_noise_temporal_phys(timepoints,
                                                )

    stimfunc = np.zeros((int(total_time / tr), 1))
    stimfunc[np.random.randint(0, int(total_time / tr), 50)] = 1
    task = fmrisim._generate_noise_temporal_task(stimfunc,
                                                )

    """## Generate signal

    the frontoparietal network processes visual attention, as well as mismatch negativity, and the posterior parietal cortex processes numbers

    That is: preception and judgement are both in posterior parietal cortex.
    """

    # Create the region of activity where signal will appear
    coordinates = np.array([[21, 21, 21]])  # Where in the brain is the signal
    feature_size = 4  # How big, in voxels, is the size of the ROI
    signal_volume = fmrisim.generate_signal(dimensions=dim[0:3],
                                            feature_type=['cube'],
                                            feature_coordinates=coordinates,
                                            feature_size=[feature_size],
                                            signal_magnitude=[1],
                                            )


    # Create a pattern for each voxel in our signal ROI
    voxels = feature_size ** 3


    """### Generate event time course

    at each event, we have the following events


    1. human umagine a number
    2. human precept a number from agent
    3. humen make a judgement about match and mismatch
    """

    # Set up stimulus event time course parameters # unit is second
    event_duration = 2  # How long is each event
    isi = 7  # What is the time between each event
    burn_in = 1  # How long before the first event
    temporal_res = 10.0 # How many timepoints per second of the stim function are to be generated?

    total_time = int(dim[3] * tr) + burn_in  # How long is the total event time course
    events = int((total_time - ((event_duration + isi) * 2))  / ((event_duration + isi) * 2)) * 2  # How many events are there?
    onsets_all = np.linspace(burn_in, events * (event_duration + isi), events)  # Space the events out

    # a list w/ length of num_class. element i contains the time stamps of when the nunber i is imagined/precepted
    onsets_imagination = [[] for _ in range(num_class)]
    length = len(onsets_all)
    data_context = []
    for i in range(length):
        context = np.random.randint(num_class)
        onsets_imagination[context].append(onsets_all[i])
        data_context.append(context)

    for i in range(num_class):
        onsets_imagination[i] = np.array(onsets_imagination[i])


    onsets_imagination_all = onsets_all.copy()

    def create_multiple_events(onsets_list):
        stimfunc_list = []
        for onsets in onsets_list:
            stimfunc = fmrisim.generate_stimfunction(onsets=onsets,
                                                    event_durations=[event_duration],
                                                    total_time=total_time,
                                                    temporal_resolution=temporal_res,
                                                    )
            stimfunc_list.append(stimfunc)
        return stimfunc_list

    # # Create a time course of events

    stimfunc_list_imagination = create_multiple_events(onsets_imagination)

    """### Estimate the voxel weight for each event
    """

    def compute_weighted_stimfunc(stimfunc_list,pattern_list):
        weights_list = []
        for i in range(len(stimfunc_list)):
            weights = np.matlib.repmat(stimfunc_list[i], 1, voxels).transpose() * pattern_list[i]
            weights_list.append(weights)
        stimfunc_weighted = np.zeros_like(weights)
        for weights in weights_list:
            stimfunc_weighted += weights
        return stimfunc_weighted.transpose()

    stimfunc_weighted_imagination = compute_weighted_stimfunc(stimfunc_list_imagination,patterns_imagination)


    """what is signal function?"""

    signal_func_imagination = fmrisim.convolve_hrf(stimfunction=stimfunc_weighted_imagination,
                                    tr_duration=tr,
                                    temporal_resolution=temporal_res,
                                    scale_function=1,
                                    )


    # Specify the parameters for signal
    signal_method = 'CNR_Amp/Noise-SD'
    signal_magnitude = [0.5]

    # Where in the brain are there stimulus evoked voxels
    signal_idxs = np.where(signal_volume == 1)

    # Pull out the voxels corresponding to the noise volume
    noise_func_imagination = noise_imagination[signal_idxs[0], signal_idxs[1], signal_idxs[2], :].T

    # Compute the signal appropriate scaled
    signal_func_scaled_imagination = fmrisim.compute_signal_change(signal_func_imagination,
                                                    noise_func_imagination,
                                                    noise_dict,
                                                    magnitude=signal_magnitude,
                                                    method=signal_method,
                                                    )

    signal_imagination = fmrisim.apply_signal(signal_func_scaled_imagination,
                                signal_volume,
                                )

    """Tune the SNR here. Remember noise_imagination is greater than noise_preception"""

    brain_imagination = signal_imagination + noise_imagination * noise_weight

    """## Analyse data"""

    hrf_lag = 4  # Assumed time from stimulus onset to HRF peak

    # Get the lower and upper bounds of the ROI
    lb = (coordinates - ((feature_size - 1) / 2)).astype('int')[0]
    ub = (coordinates + ((feature_size - 1) / 2) + 1).astype('int')[0]

    data_imagination = brain_imagination[lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2], ((onsets_imagination_all + hrf_lag) / tr).astype('int')]
    data_imagination= data_imagination.reshape((voxels, data_imagination.shape[3]))

    np.save('data/data_imagination_test_seed_' + str(args.seed) + '_noise_level_' + str(args.noise_weight) + '.npy', data_imagination.T)
    np.save('data/data_context_test_seed_' + str(args.seed) + '_noise_level_' + str(args.noise_weight) + '.npy', data_context)


if __name__ == '__main__':
    main()
