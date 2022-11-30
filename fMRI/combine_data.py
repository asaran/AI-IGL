import numpy as np

num_seeds = 10

noise_weight = 0.05


data_imagination = np.empty((0,64),float)
data_action = np.empty((0),float)
data_feedback = np.empty((0,64),float)
data_reward = np.empty((0),float)

data_imagination_test = np.empty((0,64),float)
data_context_test = np.empty((0),float)


for seed in range(1,51):
    data_imagination = np.concatenate((data_imagination,np.load('data/data_imagination_seed_' + str(seed) + '_noise_level_' + str(noise_weight) +  '.npy')), axis=0)
    data_action = np.concatenate((data_action,np.load('data/data_action_seed_' + str(seed) + '_noise_level_' + str(noise_weight) + '.npy')),axis=0)
    data_feedback = np.concatenate((data_feedback,np.load('data/data_feedback_seed_' + str(seed) + '_noise_level_' + str(noise_weight) + '.npy')), axis=0)
    data_reward = np.concatenate((data_reward, np.load('data/data_reward_seed_' + str(seed) + '_noise_level_' + str(noise_weight) + '.npy')), axis=0)
    data_imagination_test = np.concatenate((data_imagination_test,np.load('data/data_imagination_test_seed_' + str(seed) + '_noise_level_' + str(noise_weight) + '.npy')), axis=0)
    data_context_test = np.concatenate((data_context_test, np.load('data/data_context_test_seed_' + str(seed) + '_noise_level_' + str(noise_weight) + '.npy')), axis=0)

noise_weight = str(noise_weight)

np.save('data/data_imagination_combined_noise_level_'+noise_weight,data_imagination)
np.save('data/data_action_combined_noise_level_'+noise_weight,data_action)
np.save('data/data_feedback_combined_noise_level_'+noise_weight,data_feedback)
np.save('data/data_reward_combined_noise_level_'+noise_weight,data_reward)
np.save('data/data_imagination_test_combined_noise_level_'+noise_weight,data_imagination_test)
np.save('data/data_context_test_combined_noise_level_'+noise_weight,data_context_test)