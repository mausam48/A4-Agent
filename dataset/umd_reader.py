import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np

class UmdDataset(Dataset):
    """
    PyTorch Dataset class for reading the preprocessed UMD Part-Affordance dataset.
    This version loads images and masks and converts them to tensors without resizing or normalization.
    """
    
    # AFFORDANCE_MAP = {
    #     'grasp': 1,
    #     'cut': 2,
    #     'scoop': 3,
    #     'contain': 4,
    #     'pound': 5,
    #     'support': 6,
    #     'wrap-grasp': 7
    # }

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the preprocessed images and masks.
        """
        self.root_dir = root_dir
        self.image_files = []
        self.mask_files = []
        self.dataset_type = "UMD"
        
        if not os.path.exists(root_dir):
            raise RuntimeError(f"Root directory not found: {root_dir}")

        for tool_dir in sorted(os.listdir(self.root_dir)):
            tool_dir_path = os.path.join(self.root_dir, tool_dir)
            if os.path.isdir(tool_dir_path):
                for file_name in sorted(os.listdir(tool_dir_path)):
                    if file_name.endswith('_rgb.jpg'):
                        self.image_files.append(os.path.join(tool_dir_path, file_name))
                    if file_name.endswith('_gt_mask.png'):
                        self.mask_files.append(os.path.join(tool_dir_path, file_name))
                        
    def __len__(self):
        # return len(self.image_files)
        return len(self.mask_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_path = self.image_files[idx]
        mask_path = self.mask_files[idx] # ladle_02_00000003_grasp_gt_mask.png
        # Extract affordance type from mask filename
        # e.g., ladle_02_00000003_grasp_gt_mask.png -> grasp
        mask_filename = os.path.basename(mask_path)
        affordance_type = mask_filename.split('_gt_mask.png')[0].split('_')[-1]
        # Extract base filename and construct image path
        # e.g., ladle_02_00000003_grasp_gt_mask.png -> ladle_02_00000003_rgb.jpg
        base_filename = '_'.join(mask_filename.split('_')[:-3])  # Remove 'grasp_gt_mask.png' part
        img_path = os.path.join(os.path.dirname(mask_path), base_filename + '_rgb.jpg')
        # import pdb; pdb.set_trace()
        image = Image.open(img_path).convert('RGB')
        
        sample = {'image': image, 'mask_path': mask_path, 'image_path': img_path, 'affordance_type': affordance_type}

        return sample


if __name__ == '__main__':
    # Example usage:
    # NOTE: You need to change this path to your preprocessed dataset path.
    dataset_path = ''
    
    # Check if the dataset path exists
    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print(f"Error: Dataset not found or directory is empty at '{dataset_path}'.")
        print("Please run the preprocess_umd_new.py script first or check the path.")
    else:
        print(f"Loading dataset from: {dataset_path}")
        
        umd_dataset = UmdDataset(root_dir=dataset_path)
        
        print(f"Dataset size: {len(umd_dataset)}")

        if len(umd_dataset) > 0:
            # Let's check the first sample
            sample = umd_dataset[0]
            image_tensor, mask_path, image_path, affordance_type = sample['image'], sample['mask_path'], sample['image_path'], sample['affordance_type']
            
            print(f"\n--- First Sample ---")
            print(f"Image tensor shape: {image_tensor.shape}")
            print(f"Mask path: {mask_path}")
            print(f"Image path: {image_path}")
            print(f"Affordance type: {affordance_type}")
            print(f"Image path: {image_path}")
            # You can still use a DataLoader
            from torch.utils.data import DataLoader
            
            # Note: If images have different sizes, you can't use batch_size > 1 without a custom collate_fn
            try:
                dataloader = DataLoader(umd_dataset, batch_size=1, shuffle=True, num_workers=0)
                
                # Get one batch
                batch_sample = next(iter(dataloader))
                print(f"\n--- DataLoader Batch (batch_size=1) ---")
                print(f"Batch image tensor shape: {batch_sample['image'].shape}")
                print(f"Batch mask path: {batch_sample['mask_path']}")
                print(f"Batch image path: {batch_sample['image_path']}")
                print(f"Batch affordance type: {batch_sample['affordance_type']}")
            except Exception as e:
                print(f"\nCould not create a batch with the DataLoader. This might be due to images of different sizes.")
                print(f"Error: {e}")
            import pdb; pdb.set_trace()
