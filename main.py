import numpy as np
import h5py
from model import VPPM
import torch
import torch.nn as nn
from multiprocessing import Pool
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


path_files = [r'D:\2021-04-16 TCR Phase 1 Build 2.hdf5', r'D:\2021-04-28 TCR Phase 1 Build 3.hdf5']        # , r'D:\2021-04-28 TCR Phase 1 Build 3.hdf5'
probe_layers_count = 25 # 3561


# Calculate the super voxel size with a 1.0 mm x 1.0 mm x 3.5 mm voxel size for an 1842 x 1842 pixel image or 245 mm x 245 mm image
# However, the y dimension is front to back so that means we only want a 1.0 mm x 3.5 mm voxel size for the sliding matrix
# These are the dimensions of the sliding matrix calculated for length (1.0 * 1842/245 then rounded) and for width (3.5 * 1842/245 then rounded)
# Was 8 x 26
voxel_size_length = 8
voxel_size_width = 26

def compute_feature_importance(features, ground_truths):
    # Convert lists to numpy arrays
    features_np = np.array(features)
    ground_truths_np = np.array(ground_truths).ravel()  # Ensure ground_truths is a 1D array

    # Standardize features (optional but often helps)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_np)

    # Train a Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(features_scaled, ground_truths_np)

    # Get feature importances
    importances = rf.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    return importances, indices

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

    # Feature 1: Importance 0.4785019450771155 std matters a lot it appears
    # Feature 0: Importance 0.3670543755169024
    # Feature 2: Importance 0.08569122012454419
    # Feature 3: Importance 0.04652552923420717
    # Feature 12: Importance 0.004655201017576254
    # Feature 13: Importance 0.004120085084887399
    # Feature 5: Importance 0.003901564976410069
    # Feature 4: Importance 0.003353579841165944
    # Feature 7: Importance 0.003241224076512906
    # Feature 6: Importance 0.0028384535322772252
    # Feature 10: Importance 6.0885748954704845e-05
    # Feature 11: Importance 5.5935769446120465e-05

    for path_file in path_files:
        try:
            with h5py.File(path_file, 'r') as build:
                paths = [
                    'slices/segmentation_results/0',  # 0 1
                    'slices/segmentation_results/1',  # 2 3
                    'slices/segmentation_results/3',  # 4 5
                    'slices/segmentation_results/5',  # 6 7
                    'slices/segmentation_results/8',  # 8 9
                ]

                # 'slices/segmentation_results/2',  # 10 11
                # 'slices/segmentation_results/7',  # 10 11
                # 'slices/segmentation_results/6',  # 8 9
                # 'slices/segmentation_results/10', # 14 15
                #     'temporal/layer_times',           # 16
                #     'temporal/top_flow_rate',         # 17
                #     'temporal/bottom_flow_rate',      # 18
                #     'temporal/module_oxygen',         # 19
                #     'temporal/build_plate_temperature', # 20
                #     'temporal/bottom_flow_temperature', # 21
                #     'temporal/actual_ventilator_flow_rate', # 22
                # -------------------------------
                # 'slices/segmentation_results/4',  # 12 13
                # 'slices/segmentation_results/9',  # 14 15

                # Adjust probe layer for the second file because it has less layers
                if path_file == r'D:\2021-04-28 TCR Phase 1 Build 3.hdf5':
                    probe_layer = (int)(probe_layer/3.75)

                feature = [create_feature(build, path, probe_layer) for path in paths]


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

    # increment_counter = 0

    # for probe_layer in probe_layers:
    #     if probe_layer+increment_counter < 3000:
    #         probe_layer += increment_counter
    #     else:
    #         probe_layer += 3000
    #     increment_counter += 500
    #     print(f'Extracting features for probe layer {probe_layer}')
    #     features, ground_truths = extract_features(probe_layer)
    #     if features and ground_truths:
    #         all_features.extend(features)
    #         # Need to add the targets as many times as there are features
    #         all_ground_truth.extend(ground_truths)
    #     if probe_layer % 100 == 0:
    #         print(f'Finished extracting features for probe layer {probe_layer}')
    #     if len(all_features) != len(all_ground_truth):
    #         print('Length of features and ground truth do not match')
    #         print(f'Length of features: {len(all_features)}')
    #         print(f'Length of ground truth: {len(all_ground_truth)}')
    #         print(f'Probe layer: {probe_layer}')

    # probe_layers = range(15)
    probe_layers = range(3000, 3200, 25)
    with Pool(processes=8) as pool:
        results = pool.map(extract_features, probe_layers)

    all_features = []
    all_ground_truth = []

    for result in results:
        features, ground_truths = result
        if features and ground_truths:
            all_features.extend(features)
            all_ground_truth.extend(ground_truths)

    print('Finished extracting features')

    # If statement as a secondary measure if results is still None
    if all_features and all_ground_truth:
        # importances, indices = compute_feature_importance(all_features, all_ground_truth)

        # # Print feature importance
        # for i in indices:
        #     print(f"Feature {i}: Importance {importances[i]}")

        # # Optionally, visualize feature importances
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 6))
        # plt.title('Feature Importances')
        # plt.bar(range(len(importances)), importances[indices], align='center')
        # plt.xticks(range(len(importances)), indices)
        # plt.xlabel('Feature Index')
        # plt.ylabel('Importance')
        # plt.show()
        # exit()
                
        # Convert lists to tensors
        all_features_tensor = torch.tensor(all_features, dtype=torch.float32).cuda()
        all_ground_truth_tensor = torch.tensor(all_ground_truth, dtype=torch.float32).cuda()

        # Split data into training and validation sets
        try:
            train_features, val_features, train_targets, val_targets = train_test_split(
                all_features_tensor, all_ground_truth_tensor, test_size=0.2, random_state=42
            )
        except:
            print('Error splitting data into training and validation sets')
            print(f'Length of features: {len(all_features_tensor)}')
            print(f'Length of targets: {len(all_ground_truth_tensor)}')


        # Create DataLoader for batch training
        train_dataset = TensorDataset(train_features, train_targets)
        val_dataset = TensorDataset(val_features, val_targets)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

        # Create the model
        model = VPPM(n_features=10).cuda()

        # Load model
        # model.load_state_dict(torch.load('models/model_weights.pth')['model_state_dict'])
        # model.apply(weights_init)

        # Create cross entropy loss and optimizer functions
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        for epoch in range(400):                
            train_loss = 0
            model.train()
            # 3561 comes from the number of layers
            for inputs, targets in train_loader:    # 3561
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

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.cuda()
                    targets = targets.cuda().unsqueeze(1)
                    outputs = model(inputs)

                    loss = criterion(outputs, targets)
                    l1_penalty = l1_regularization(model, lambda_l1)
                    loss += l1_penalty
                    loss = torch.sqrt(loss)

                    val_loss += loss.item()
            scheduler.step(val_loss)

            print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}')

            if epoch % 10 == 0:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss / len(train_loader),
                'val_loss': val_loss / len(val_loader)}, 
                f'models/model_weights.pth')
    else:
        print('No valid results found')