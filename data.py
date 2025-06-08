from scipy.io import loadmat

tau = loadmat('project_files_&_data/tau.mat')['tau']
samples = loadmat('project_files_&_data/samples.mat')['y_sampled'][0,:]
