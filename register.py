# #!/usr/bin/env python

#!/usr/bin/env python

#!/usr/bin/env python

import os
import glob
import argparse
import torch
import numpy as np
import cv2
from tqdm import tqdm

# Configure environment
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm

def find_latest_model(model_dir='models'):
    """Find the latest model checkpoint in the models directory."""
    model_files = glob.glob(os.path.join(model_dir, '*.pt'))
    if not model_files:
        raise ValueError(f"No model checkpoints found in {model_dir}")
    
    # Extract epoch numbers and find the latest
    epochs = [int(os.path.splitext(os.path.basename(f))[0]) for f in model_files]
    latest_idx = np.argmax(epochs)
    return model_files[latest_idx]

def natural_sort_key(s):
    """Key function for natural sorting of filenames."""
    import re
    # Split the string into numeric and non-numeric parts
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def calculate_error(deformed, target):
    """
    Calculate pixelwise error between deformed source and target tensor.
    
    Parameters:
    deformed (torch.Tensor): The deformed source image from the model
    target (torch.Tensor): The actual target image
    
    Returns:
    torch.Tensor: Pixelwise absolute error
    """
    #return torch.abs(deformed - target)
    return torch.mean((deformed - target) ** 2)

def process_scalar_field(scalar_field, threshold=1e-6):
    """Process scalar field using colormap to highlight different value ranges."""
    # Convert to numpy and take absolute values
    field = np.abs(scalar_field.cpu().numpy().squeeze())
    
    # Create the processed field
    field_processed = np.zeros_like(field)
    non_zero_mask = field > threshold
    
    if non_zero_mask.any():
        # Apply power scaling for better small value visibility
        field_processed[non_zero_mask] = np.power(field[non_zero_mask], 0.5)
        
        # Normalize to [0, 1] for colormap
        field_processed = (field_processed - field_processed.min()) / (field_processed.max() - field_processed.min())
        
        # Create custom colormap
        colors = np.array([
            [0, 0, 0],        # 0: Black for zero values
            [0, 0, 128],      # Very small values: Navy
            [0, 0, 255],      # Small values: Blue
            [0, 255, 255],    # Small-medium values: Cyan
            [0, 255, 0],      # Medium values: Green
            [255, 255, 0],    # Medium-high values: Yellow
            [255, 128, 0],    # High values: Orange
            [255, 0, 0]       # Very high values: Red
        ]) / 255.0
        
        # Create lookup table for colormap
        n_bins = 256
        lookup_table = np.zeros((n_bins, 3))
        positions = np.linspace(0, 1, len(colors))
        
        for i in range(3):  # RGB channels
            lookup_table[:, i] = np.interp(
                np.linspace(0, 1, n_bins),
                positions,
                colors[:, i]
            )
        
        # Apply colormap
        indices = (field_processed * (n_bins - 1)).astype(int)
        colored_field = lookup_table[indices]
        
        # Convert to uint8
        colored_field = (colored_field * 255).astype(np.uint8)
        
        return colored_field
    
    return np.zeros((*field.shape, 3), dtype=np.uint8)

def save_scalar_field(scalar_field, output_path, frame_num):
    """Save scalar field as both raw numpy array and visualization."""
    try:
        # Save raw data
        np_path = os.path.join(output_path, 'raw', f'frame_{frame_num:04d}.npy')
        os.makedirs(os.path.dirname(np_path), exist_ok=True)
        np.save(np_path, scalar_field.cpu().numpy().squeeze())
        
        # Save visualization
        vis_path = os.path.join(output_path, 'vis', f'frame_{frame_num:04d}.png')
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        
        # Process and save visualization
        field_processed = process_scalar_field(scalar_field)
        cv2.imwrite(vis_path, cv2.cvtColor(field_processed, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"Error saving scalar field for frame {frame_num}: {str(e)}")

def downsample_scalar_field(scalar_field):
    """Downsample scalar field by half using average pooling."""
    return torch.nn.functional.avg_pool2d(scalar_field, kernel_size=2, stride=2)

def save_progression():
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model_path = find_latest_model('models')
    print(f"Loading model from {model_path}")
    model = vxm.networks.VxmDense.load(model_path, device)
    model.to(device)
    model.eval()
    
    # Load initial data
    data = np.load('voxel_data/train_0_cine_1.npy')
    source = data[..., 0]
    target = data[..., 1]
    
    # Initialize video writer
    frame_size = (source.shape[1], source.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('progression.mp4', fourcc, 2.0, frame_size)
    
    # Save initial frame
    source_norm = ((source - source.min()) / (source.max() - source.min()) * 255).astype(np.uint8)
    source_rgb = cv2.cvtColor(source_norm, cv2.COLOR_GRAY2RGB)
    video_writer.write(source_rgb)
    
    # Prepare initial tensors
    current_source = torch.from_numpy(source).unsqueeze(0).unsqueeze(0).float().to(device)
    target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).float().to(device)
    
    print("Processing iterative deformations...")
    with torch.no_grad():
        for i in range(20):
            # Normalize tensors
            # Normalization
            # source_min = current_source.min()
            # source_max = current_source.max()
            # target_min = target_tensor.min()
            # target_max = target_tensor.max()

            # if source_max > source_min:
            #     current_source = (current_source - source_min) / (source_max - source_min)
            # else:
            #     current_source = source_tensor - source_min

            # if target_max > target_min:
            #     target_tensor = (target_tensor - target_min) / (target_max - target_min)
            # else:
            #     target_tensor = target_tensor - target_min
            
            # Get model prediction
            inputs = [current_source, target_tensor]
            deformed_source, _, _ = model(*inputs)
            
            # Convert deformed source to video frame
            deformed = deformed_source.cpu().numpy().squeeze()
            deformed_norm = ((deformed - deformed.min()) / (deformed.max() - deformed.min()) * 255).astype(np.uint8)
            deformed_rgb = cv2.cvtColor(deformed_norm, cv2.COLOR_GRAY2RGB)
            
            # Write frame
            video_writer.write(deformed_rgb)
            
            # Update source for next iteration
            current_source = deformed_source
            
            print(f"Completed iteration {i+1}/20")
    
    # Release video writer
    video_writer.release()
    print("Progression video saved as progression.mp4")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='voxel_data',
                        help='directory containing preprocessed .npy files')
    parser.add_argument('--model-dir', default='models',
                        help='directory containing model checkpoints')
    parser.add_argument('--output-source', default='source_video.mp4',
                        help='output video file name for source images')
    parser.add_argument('--output-deformed', default='deformed_video.mp4',
                        help='output video file name for deformed images')
    parser.add_argument('--scalar-field-dir', default='scalar_field',
                        help='directory for saving scalar field data')
    parser.add_argument('--gpu', default='0',
                        help='GPU ID number(s) (default: 0)')
    args = parser.parse_args()

    # Device handling
    device = 'cuda' if torch.cuda.is_available() and args.gpu != '-1' else 'cpu'
    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Create scalar field directories
    original_dir = os.path.join(args.scalar_field_dir, 'original')
    downsampled_dir = os.path.join(args.scalar_field_dir, 'downsampled')
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(downsampled_dir, exist_ok=True)

    # Find and load the latest model
    model_path = find_latest_model(args.model_dir)
    print(f"Loading model from {model_path}")
    model = vxm.networks.VxmDense.load(model_path, device)
    model.to(device)
    model.eval()

    # Get all data files
    data_files = sorted(glob.glob(os.path.join(args.data_dir, '*.npy')))
    if not data_files:
        raise ValueError(f"No .npy files found in {args.data_dir}")
    
    print(f"Found {len(data_files)} input files")
    processed_count = 0
    failed_files = []

    data_files.sort(key=natural_sort_key)

    # Get image dimensions from first file
    try:
        sample_data = np.load(data_files[0])
        if sample_data.ndim != 3 or sample_data.shape[-1] != 2:
            raise ValueError(f"Invalid data shape: {sample_data.shape}")
        frame_size = (sample_data.shape[1], sample_data.shape[0])
    except Exception as e:
        print(f"Error loading first file: {str(e)}")
        return

    # Initialize video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    source_writer = cv2.VideoWriter(args.output_source, fourcc, 2.0, frame_size)
    deformed_writer = cv2.VideoWriter(args.output_deformed, fourcc, 2.0, frame_size)

    # Process each pair of images
    print("Processing image pairs...")
    with torch.no_grad():
        for frame_num, data_file in enumerate(tqdm(data_files)):
            try:
                # Extract frame numbers from filename
                base_name = os.path.basename(data_file)
                frame_nums = base_name.split('.')[0]  # Remove file extension
                #train_num, cine_num = frame_nums.split('_')  # Assuming filename format: "train_X_cine_Y.npy"
                train_num = frame_nums.split('_')[1]
                cine_num = frame_nums.split('_')[-1]
                # Load and prepare data
                data = np.load(data_file)
                if data.ndim != 3 or data.shape[-1] != 2:
                    print(f"Error: Invalid data shape in file {data_file}")
                    failed_files.append(data_file)
                    continue

                source = data[..., 0]
                target = data[..., 1]
                
                # Prepare input tensors
                source_tensor = torch.from_numpy(source).unsqueeze(0).unsqueeze(0).float().to(device)
                target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).float().to(device)
                
                # # Normalization
                # source_min = source_tensor.min()
                # source_max = source_tensor.max()
                # target_min = target_tensor.min()
                # target_max = target_tensor.max()

                # if source_max > source_min:
                #     source_tensor = (source_tensor - source_min) / (source_max - source_min)
                # else:
                #     source_tensor = source_tensor - source_min

                # if target_max > target_min:
                #     target_tensor = (target_tensor - target_min) / (target_max - target_min)
                # else:
                #     target_tensor = target_tensor - target_min

                # Get model prediction
                inputs = [source_tensor, target_tensor]
                deformed_source, phi, scalar_field = model(*inputs)

                # Calculate pixelwise error
                error_def = calculate_error(deformed_source, target_tensor)
                mean_error = error_def.mean().item()
                
                # Print error with specified format
                print(f"After deformation train_{train_num}_cine_{cine_num} has error: {mean_error:.4f}")

                # Calculate pixelwise error
                error_wo = calculate_error(source_tensor, target_tensor)
                mean_error = error_wo.mean().item()
                
                # Print error with specified format
                print(f"Without deformation train_{train_num}_cine_{cine_num} has error: {mean_error:.4f}")
                
                # Save scalar fields
                save_scalar_field(scalar_field, original_dir, frame_num)
                save_scalar_field(downsample_scalar_field(scalar_field), downsampled_dir, frame_num)
                
                # Process source image
                source_norm = ((source - source.min()) / (source.max() - source.min()) * 255).astype(np.uint8)
                source_rgb = cv2.cvtColor(source_norm, cv2.COLOR_GRAY2RGB)
                
                # Process deformed image
                deformed = deformed_source.cpu().numpy().squeeze()
                deformed_norm = ((deformed - deformed.min()) / (deformed.max() - deformed.min()) * 255).astype(np.uint8)
                deformed_rgb = cv2.cvtColor(deformed_norm, cv2.COLOR_GRAY2RGB)
                
                # Write frames to videos
                source_writer.write(source_rgb)
                deformed_writer.write(deformed_rgb)
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing file {data_file}: {str(e)}")
                failed_files.append(data_file)

    # Release video writers
    source_writer.release()
    deformed_writer.release()

    # Print summary
    print("\nProcessing Summary:")
    print(f"Total input files: {len(data_files)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed files: {len(failed_files)}")
    
    if failed_files:
        print("\nFailed files:")
        for file_path in failed_files:
            print(f"- {os.path.basename(file_path)}")

    print(f"\nSource video saved to {args.output_source}")
    print(f"Deformed video saved to {args.output_deformed}")
    print(f"Scalar fields saved to:")
    print(f"  Original: {original_dir}")
    print(f"  Downsampled: {downsampled_dir}")

if __name__ == '__main__':
    main()
    #save_progression()





# import os
# import glob
# import argparse
# import torch
# import numpy as np
# import cv2
# from tqdm import tqdm

# # Configure environment
# os.environ['NEURITE_BACKEND'] = 'pytorch'
# os.environ['VXM_BACKEND'] = 'pytorch'
# import voxelmorph as vxm

# def find_latest_model(model_dir='models'):
#     """Find the latest model checkpoint in the models directory."""
#     model_files = glob.glob(os.path.join(model_dir, '*.pt'))
#     if not model_files:
#         raise ValueError(f"No model checkpoints found in {model_dir}")
    
#     # Extract epoch numbers and find the latest
#     epochs = [int(os.path.splitext(os.path.basename(f))[0]) for f in model_files]
#     latest_idx = np.argmax(epochs)
#     return model_files[latest_idx]

# def main():
#     # Parse arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data-dir', default='voxel_data',
#                         help='directory containing preprocessed .npy files')
#     parser.add_argument('--model-dir', default='models',
#                         help='directory containing model checkpoints')
#     parser.add_argument('--output-source', default='source_video.mp4',
#                         help='output video file name for source images')
#     parser.add_argument('--output-deformed', default='deformed_video.mp4',
#                         help='output video file name for deformed images')
#     parser.add_argument('--gpu', default='0',
#                         help='GPU ID number(s) (default: 0)')
#     args = parser.parse_args()

#     # Device handling
#     device = 'cuda' if torch.cuda.is_available() and args.gpu != '-1' else 'cpu'
#     if args.gpu != '-1':
#         os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

#     # Find and load the latest model
#     model_path = find_latest_model(args.model_dir)
#     print(f"Loading model from {model_path}")
#     model = vxm.networks.VxmDense.load(model_path, device)
#     model.to(device)
#     model.eval()

#     # Get all data files
#     data_files = sorted(glob.glob(os.path.join(args.data_dir, '*.npy')))
#     if not data_files:
#         raise ValueError(f"No .npy files found in {args.data_dir}")

#     # Get image dimensions from first file
#     sample_data = np.load(data_files[0])
#     if sample_data.ndim != 3 or sample_data.shape[-1] != 2:
#         raise ValueError(f"Unexpected data shape: {sample_data.shape}")

#     # Initialize video writers
#     frame_size = (sample_data.shape[1], sample_data.shape[0])  # original image size
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     source_writer = cv2.VideoWriter(args.output_source, fourcc, 2.0, frame_size)
#     deformed_writer = cv2.VideoWriter(args.output_deformed, fourcc, 2.0, frame_size)

#     # Process each pair of images
#     print("Processing image pairs...")
#     with torch.no_grad():
#         for data_file in tqdm(data_files):
#             # Load and prepare data
#             data = np.load(data_file)
#             source = data[..., 0]
#             target = data[..., 1]
            
#             # Prepare input tensors
#             source_tensor = torch.from_numpy(source).unsqueeze(0).unsqueeze(0).float().to(device)
#             target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).float().to(device)
            
#             # Get model prediction
#             inputs = [source_tensor, target_tensor]
#             deformed_source, phi, scalar_field = model(*inputs)
            
#             # Process source image
#             source_norm = ((source - source.min()) / (source.max() - source.min()) * 255).astype(np.uint8)
#             source_rgb = cv2.cvtColor(source_norm, cv2.COLOR_GRAY2RGB)
            
#             # Process deformed image
#             deformed = deformed_source.cpu().numpy().squeeze()
#             deformed_norm = ((deformed - deformed.min()) / (deformed.max() - deformed.min()) * 255).astype(np.uint8)
#             deformed_rgb = cv2.cvtColor(deformed_norm, cv2.COLOR_GRAY2RGB)
            
#             # Write frames to videos
#             source_writer.write(source_rgb)
#             deformed_writer.write(deformed_rgb)

#     # Release video writers
#     source_writer.release()
#     deformed_writer.release()
#     print(f"Source video saved to {args.output_source}")
#     print(f"Deformed video saved to {args.output_deformed}")

# if __name__ == '__main__':
#     main()
    
# import os
# import glob
# import argparse
# import torch
# import numpy as np
# import cv2
# from tqdm import tqdm

# # Configure environment
# os.environ['NEURITE_BACKEND'] = 'pytorch'
# os.environ['VXM_BACKEND'] = 'pytorch'
# import voxelmorph as vxm

# def find_latest_model(model_dir='models'):
#     """Find the latest model checkpoint in the models directory."""
#     model_files = glob.glob(os.path.join(model_dir, '*.pt'))
#     if not model_files:
#         raise ValueError(f"No model checkpoints found in {model_dir}")
    
#     # Extract epoch numbers and find the latest
#     epochs = [int(os.path.splitext(os.path.basename(f))[0]) for f in model_files]
#     latest_idx = np.argmin(epochs)  #np.argmax(epochs)
#     return model_files[latest_idx]

# def create_video_writer(output_path, frame_size):
#     """Create a video writer with specified settings."""
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     return cv2.VideoWriter(output_path, fourcc, 2.0, frame_size)

# def process_batch(batch_data, model, device, source_path, deformed_path, frame_size, batch_num):
#     """Process a batch of images and save to video."""
#     # Create video writers for this batch
#     source_writer = create_video_writer(
#         os.path.join(source_path, f'batch_{batch_num:03d}.mp4'),
#         frame_size
#     )
#     deformed_writer = create_video_writer(
#         os.path.join(deformed_path, f'batch_{batch_num:03d}.mp4'),
#         frame_size
#     )
    
#     with torch.no_grad():
#         for data in batch_data:
#             # Load and prepare data
#             source = data[..., 0]
#             target = data[..., 1]
            
#             # Prepare input tensors
#             source_tensor = torch.from_numpy(source).unsqueeze(0).unsqueeze(0).float().to(device)
#             target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).float().to(device)
            
#             # Get model prediction
#             inputs = [source_tensor, target_tensor]
#             deformed_source, _ = model(*inputs)
            
#             # Process source image
#             source_norm = ((source - source.min()) / (source.max() - source.min()) * 255).astype(np.uint8)
#             source_rgb = cv2.cvtColor(source_norm, cv2.COLOR_GRAY2RGB)
            
#             # Process deformed image
#             deformed = deformed_source.cpu().numpy().squeeze()
#             deformed_norm = ((deformed - deformed.min()) / (deformed.max() - deformed.min()) * 255).astype(np.uint8)
#             deformed_rgb = cv2.cvtColor(deformed_norm, cv2.COLOR_GRAY2RGB)
            
#             # Write frames to videos
#             source_writer.write(source_rgb)
#             deformed_writer.write(deformed_rgb)
    
#     # Release writers
#     source_writer.release()
#     deformed_writer.release()

# def main():
#     # Parse arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data-dir', default='voxel_data',
#                         help='directory containing preprocessed .npy files')
#     parser.add_argument('--model-dir', default='models',
#                         help='directory containing model checkpoints')
#     parser.add_argument('--output-dir', default='output_videos',
#                         help='base output directory for videos')
#     parser.add_argument('--batch-size', type=int, default=20,
#                         help='number of frames per video')
#     parser.add_argument('--gpu', default='0',
#                         help='GPU ID number(s) (default: 0)')
#     args = parser.parse_args()

#     # Create output directories
#     source_dir = os.path.join(args.output_dir, 'source_videos')
#     deformed_dir = os.path.join(args.output_dir, 'deformed_videos')
#     os.makedirs(source_dir, exist_ok=True)
#     os.makedirs(deformed_dir, exist_ok=True)

#     # Device handling
#     device = 'cuda' if torch.cuda.is_available() and args.gpu != '-1' else 'cpu'
#     if args.gpu != '-1':
#         os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

#     # Find and load the latest model
#     model_path = find_latest_model(args.model_dir)
#     print(f"Loading model from {model_path}")
#     model = vxm.networks.VxmDense.load(model_path, device)
#     model.to(device)
#     model.eval()

#     # Get all data files
#     data_files = sorted(glob.glob(os.path.join(args.data_dir, '*.npy')))
#     if not data_files:
#         raise ValueError(f"No .npy files found in {args.data_dir}")

#     # Get image dimensions from first file
#     sample_data = np.load(data_files[0])
#     if sample_data.ndim != 3 or sample_data.shape[-1] != 2:
#         raise ValueError(f"Unexpected data shape: {sample_data.shape}")

#     frame_size = (sample_data.shape[1], sample_data.shape[0])

#     # Process files in batches
#     batch_data = []
#     batch_count = 0
    
#     print("Processing image pairs...")
#     for data_file in tqdm(data_files):
#         # Load data
#         data = np.load(data_file)
#         batch_data.append(data)
        
#         # Process batch when it reaches the desired size or at the end
#         if len(batch_data) == args.batch_size or data_file == data_files[-1]:
#             process_batch(
#                 batch_data,
#                 model,
#                 device,
#                 source_dir,
#                 deformed_dir,
#                 frame_size,
#                 batch_count
#             )
#             batch_data = []  # Clear batch
#             batch_count += 1

#     print(f"Videos saved to:")
#     print(f"  Source videos: {source_dir}")
#     print(f"  Deformed videos: {deformed_dir}")

# if __name__ == '__main__':
#     main()