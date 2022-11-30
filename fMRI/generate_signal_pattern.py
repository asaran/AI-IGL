import pickle

import nibabel
import numpy as np
from brainiak.utils import fmrisim

num_class = 3

# load human data
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

"""compute noise parameters"""

print('Calculating noise parameters...')
# Calculate the noise parameters from the data. Set it up to be matched.
noise_dict = {'voxel_size': [dimsize[0], dimsize[1], dimsize[2]], 'matched': 1}
noise_dict = fmrisim.calc_noise(volume=volume,
                                mask=mask,
                                template=template,
                                noise_dict=noise_dict,
                                )
print('Noise parameters computation done...')


feature_size = 4  # How big, in voxels, is the size of the ROI
voxels = feature_size ** 3

patterns_perception = [np.random.rand(voxels).reshape((voxels, 1)) for _ in range(num_class)]
# patterns_imagination = [np.random.rand(voxels).reshape((voxels, 1)) for _ in range(num_class)]
patterns_imagination = [noise + np.random.rand(voxels).reshape((voxels, 1)) * 0.5 for noise in patterns_perception]
patterns_judgment = [np.random.rand(voxels).reshape((voxels, 1)) for _ in range(2)]


with open('data/noise_dict.pkl', 'wb') as f:
    pickle.dump(noise_dict, f)

np.save('data/patterns_perception.npy', patterns_perception)
np.save('data/patterns_imagination.npy', patterns_imagination)
np.save('data/patterns_judgment.npy', patterns_judgment)