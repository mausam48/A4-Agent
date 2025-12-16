import os
import scipy.io
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
import shutil

def preprocess_umd_dataset(dataset_path, output_path):
    """
    Preprocesses the UMD Part-Affordance dataset by converting .mat label files
    to affordance mask images.
    """
    affordance_map = {
        'knife': ['cut', 'grasp'],
        'saw': ['cut', 'grasp'],
        'scissors': ['cut', 'grasp'],
        'shears': ['cut', 'grasp'],
        'scoop': ['scoop', 'grasp'],
        'spoon': ['scoop', 'grasp'],
        'trowel': ['scoop', 'grasp'],
        'bowl': ['contain', 'grasp'],
        'cup': ['contain', 'grasp'],
        'ladle': ['contain', 'grasp'],
        'mug': ['contain', 'grasp'],
        'pot': ['contain', 'grasp'],
        'shovel': ['support', 'grasp'],
        'turner': ['support', 'grasp'],
        'hammer': ['pound', 'grasp'],
        'mallet': ['pound', 'grasp'],
        'tenderizer': ['pound', 'grasp']
    }
    afford_dict_name_to_num = {
        'grasp': 1,
        'cut': 2,
        'scoop': 3,
        'contain': 4,
        'pound': 5,
        'support': 6,
        'wrap-grasp': 7
    }

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    tool_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    tool_dirs.sort()

    for tool_dir in tqdm(tool_dirs, desc="Processing tool categories"):
        category = tool_dir.split('_')[0]
        if category not in affordance_map:
            continue

        actions = affordance_map[category]
        tool_dir_path = os.path.join(dataset_path, tool_dir)
        
        output_tool_dir = os.path.join(output_path, tool_dir)
        if not os.path.exists(output_tool_dir):
            os.makedirs(output_tool_dir)

        label_rank_files = [file for file in sorted(os.listdir(tool_dir_path)) if file.endswith('_label_rank.mat')]
        rgb_files = [file for file in sorted(os.listdir(tool_dir_path)) if file.endswith('.jpg')]

        print(len(label_rank_files), len(rgb_files))
        assert len(label_rank_files) == len(rgb_files)

        for i in range(len(label_rank_files)):
            if i % 30 != 2:
                continue
            label_rank_file = label_rank_files[i]
            rgb_file = rgb_files[i]
            # rgb_file_path = os.path.join(tool_dir_path, rgb_file)
            # rgb_image = Image.open(rgb_file_path)
            mat_file_path = os.path.join(tool_dir_path, label_rank_file)
            mat_data = scipy.io.loadmat(mat_file_path)
            gt_mat = mat_data['gt_label']
            # import pdb; pdb.set_trace()

            h, w = gt_mat.shape[:2]
            
            for action in actions:
                if action not in afford_dict_name_to_num:
                    continue

                action_index = afford_dict_name_to_num[action] - 1
                gt_mask = np.zeros((h, w))
                
                # check if the 3rd dimension has enough channels
                if gt_mat.shape[2] <= action_index:
                    # print(f"Warning: action_index {action_index} out of bounds for {mat_file_path}")
                    continue
                # import pdb; pdb.set_trace()
                max_val = np.max(gt_mat)

                if max_val == 0: # Avoid division by zero for empty masks
                    gt_mask_uint8 = (gt_mask * 255).astype(np.uint8)
                    mask_image = Image.fromarray(gt_mask_uint8)
                    output_filename = label_rank_file.replace('_label_rank.mat', f'_{action}_gt_mask.png')
                    output_filepath = os.path.join(output_tool_dir, output_filename)
                    mask_image.save(output_filepath)
                    continue

                # for i in range(h):
                #     for j in range(w):
                #         rank = gt_mat[i, j, action_index]
                #         if rank != 0:
                #             gt_mask[i, j] = 1 - rank / max_val
                
                # Vectorize the mask creation to avoid slow pixel-wise loops
                rank_channel = gt_mat[:, :, action_index] # here
                # print(rank_channel.max())
                # non_zero_mask = rank_channel != 0
                # gt_mask[non_zero_mask] = 1 - rank_channel[non_zero_mask] / max_val
                equal_one_mask = rank_channel == 1
                gt_mask[equal_one_mask] = 1
                
                gt_mask_uint8 = (gt_mask * 255).astype(np.uint8)
                mask_image = Image.fromarray(gt_mask_uint8)
                
                output_filename = label_rank_file.replace('_label_rank.mat', f'_{action}_gt_mask.png')
                output_filepath = os.path.join(output_tool_dir, output_filename)
                mask_image.save(output_filepath)
                
                # copy rgb image to output directory
                # output_filename = rgb_file.replace('.jpg', f'_{action}_rgb.jpg')
                output_rgb_filename = rgb_file
                output_rgb_filepath = os.path.join(output_tool_dir, output_rgb_filename)
                shutil.copy(os.path.join(tool_dir_path, rgb_file), output_rgb_filepath)
                # continue

if __name__ == "__main__":

    dataset_path = '/hpc2hdd/home/zzhang300/zixin_workspace/CVPR26/Dataset/part-affordance-dataset/tools'
    output_path = '/hpc2hdd/home/zzhang300/zixin_workspace/CVPR26/Dataset/part-affordance-dataset/preprocessed-testset-11-4'

    preprocess_umd_dataset(dataset_path, output_path)
    print("Preprocessing complete.")
