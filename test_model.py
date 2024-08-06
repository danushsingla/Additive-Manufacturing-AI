import numpy as np
import h5py
from model import VPPM
import torch
import torch.nn as nn
from multiprocessing import Pool
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split


path_files = [r'D:\2021-04-16 TCR Phase 1 Build 2.hdf5', r'D:\2021-04-28 TCR Phase 1 Build 3.hdf5']
probe_layers_count = 15 # 3561 # 23 is the problem


# Calculate the super voxel size with a 1.0 mm x 1.0 mm x 3.5 mm voxel size for an 1842 x 1842 pixel image or 245 mm x 245 mm image
# However, the y dimension is front to back so that means we only want a 1.0 mm x 3.5 mm voxel size for the sliding matrix
# These are the dimensions of the sliding matrix calculated for length (1.0 * 1842/245 then rounded) and for width (3.5 * 1842/245 then rounded)
# Was 8 x 26
voxel_size_length = 8
voxel_size_width = 26

# Original
def mean(values):
    return torch.mean(values).item()

def rmse(values):
    # Flatten the values from 2D tensor to 1D tensor
    flattened_values = values.flatten()
    # Calculate RMSE
    return torch.sqrt(torch.mean(flattened_values**2)).item()

# What if I doubled the features and added this to the feature list?
def std(values):
    return torch.std(values).item()


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# Define the L1 regularization function
def l1_regularization(model, lambda_l1):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return lambda_l1 * l1_norm

# Training loop with L1 regularization
lambda_l1 = 1e-5

def create_feature(build, path, probe_layer):
     # Extract array
    array = build[path][probe_layer, ...]

    # Check if this is an array or a single float. If array, do super voxels
    if(type(array) == np.ndarray):      
        # Convert to torch tensor and move to GPU
        array = torch.tensor(array, dtype=torch.float32).cuda()

        # Normalize array values
        max_val = torch.max(array)

        # For True and False values
        if max_val == False:
            max_val = 0
        elif max_val == True:
            max_val = 1

        # Just in case max_val is 0 and it can skip 1
        if max_val != 0:
            array = array / max_val
        
        # # Clamp in case there are negative values (shouldn't exist for this data)
        # array = torch.clamp(array, min=0)

        # Get dimensions of array
        length, width = array.shape # 1842, 1842

        # Iterate through the array and calculate the mean of each super voxel
        super_voxels = []
        for i in range(0, length, voxel_size_length):
            for j in range(0, width, voxel_size_width):
                super_voxel = array[i:i+voxel_size_length, j:j+voxel_size_width]
                # if torch.sum(super_voxel) != 0:  # Ignore if the super voxel contains only zeros
                super_voxels.append(mean(super_voxel))
                super_voxels.append(std(super_voxel))
        
        # Now take care of the overlap
        super_voxel = array[-voxel_size_length:, -voxel_size_width:]
        # if torch.sum(super_voxel) != 0:  # Ignore if the super voxel contains only zeros
        super_voxels.append(mean(super_voxel))
        super_voxels.append(std(super_voxel))

        return super_voxels
    else:
        max_val = np.max(array)

        # For True and False values
        if max_val == False:
            max_val = 0
        elif max_val == True:
            max_val = 1

        # Just in case max_val is 0
        if max_val != 0:
            array = array / max_val

        # Find mean of array and return it
        return np.mean(array)

def extract_features(probe_layer):
    features = []
    ground_truths = []

    for path_file in path_files:
        try:
            with h5py.File(path_file, 'r') as build:
                paths = [
                    'slices/segmentation_results/0',
                    'slices/segmentation_results/1',
                    'slices/segmentation_results/3',
                    'slices/segmentation_results/5',
                    'slices/segmentation_results/6',
                    'slices/segmentation_results/7',
                    'slices/segmentation_results/8',
                    'slices/segmentation_results/10',
                    'temporal/layer_times',
                    'temporal/top_flow_rate',
                    'temporal/bottom_flow_rate',
                    'temporal/module_oxygen',
                    'temporal/build_plate_temperature',
                    'temporal/bottom_flow_temperature',
                    'temporal/actual_ventilator_flow_rate',
                ]

                # Adjust probe layer for the second file because it has less layers
                if path_file == r'D:\2021-04-28 TCR Phase 1 Build 3.hdf5':
                    probe_layer = (int)(probe_layer/3.75)

                feature = [create_feature(build, path, probe_layer) for path in paths]

                # Check if a list is empty (all zeros)
                # skip_extraction = False
                # for item in feature:
                #     if type(item) == list and len(item) == 0:
                #         skip_extraction = True
                # if skip_extraction:
                #     continue

                # Check nan values
                for values in feature:
                    if type(values) == list:
                        for value in values:
                            if np.isnan(value):
                                raise ValueError("NaN values found in features")
                    else:
                        if np.isnan(values):
                            raise ValueError("NaN values found in features")
                
                # Assuming that feature[0] is an array (may change in the future)
                # Divide by 2 since you are adding std with the mean

                for index in range((int)(len(feature[0])/2)):
                    # Want to create a list for each voxel
                    temp_list = []
                    for lists in feature:
                        if type(lists) == list:
                            # Only append for each index
                            # Append mean
                            temp_list.append(lists[index])
                            # Append std
                            temp_list.append(lists[index+1])
                        else:
                            # Since lists is a scalar we want to append all scalar values
                            temp_list.append(lists)
                    features.append(temp_list)
                # Get ground truth (we are just doing UTS for now)
                ground_truth = build['samples/test_results/ultimate_tensile_strength'][probe_layer, ...]
                if np.isnan(ground_truth).any():
                    raise KeyError
                
                # Append same amount of ground_truths as features
                for i in range((int)(len(feature[0])/2)):
                    ground_truths.append(ground_truth)
        except (KeyError, IndexError) as e:
            print(e)
            exit()
            # Handle cases where probe_layer is not defined or index is out of range
            return None, None
    if len(features) != 0 and len(ground_truths) != 0:
        return features, ground_truths
    else:
        return None, None

if __name__ == '__main__':
    results = None
    # Extract features and ground truth using parallel processing
    probe_layers = range(probe_layers_count)  # 3561

    all_features = []
    all_ground_truth = []

    for probe_layer in probe_layers:
        features, ground_truths = extract_features(probe_layer)
        if features and ground_truths:
            all_features.extend(features)
            # Need to add the targets as many times as there are features
            all_ground_truth.extend(ground_truths)
        if probe_layer % 100 == 0:
            print(f'Finished extracting features for probe layer {probe_layer}')

    print('Finished extracting features')

    # If statement as a secondary measure if results is still None
    if all_features and all_ground_truth:
        # Convert lists to tensors
        all_features_tensor = torch.tensor(all_features, dtype=torch.float32).cuda()
        all_ground_truth_tensor = torch.tensor(all_ground_truth, dtype=torch.float32).cuda()

        # Create DataLoader for batch training
        dataset = TensorDataset(all_features_tensor, all_ground_truth_tensor)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        # Create the model
        model = VPPM(n_features=23).cuda()
        model.load_state_dict(torch.load(r"models/model_weights.pth")['model_state_dict'])

        # Create cross entropy loss and optimizer functions
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)


        for epoch in range(200):
            train_loss = 0
            for inputs, targets in dataloader:
                inputs = inputs.cuda()
                inputs = inputs.squeeze()

                # Forward pass
                outputs = model(inputs)
                targets = targets.unsqueeze(1)

                # Calculate the loss (sqrt because we want RMS not MSE)
                loss = criterion(outputs, targets)

                # Add L1 regularization
                l1_penalty = l1_regularization(model, lambda_l1)
                loss += l1_penalty

                # Find RMSE
                if(loss < 0):
                    print(f"Loss: {loss}")
                else:
                    loss = torch.sqrt(loss)
                
                train_loss += loss.item()

            print(f'Epoch: {epoch}, Loss: {train_loss/len(dataloader)}')