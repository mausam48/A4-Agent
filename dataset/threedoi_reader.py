import os
import pickle
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ThreeDOIReasoningDataset(Dataset):
    """
    PyTorch Dataset class for reading the 3DOI reasoning affordance dataset.
    
    This reader loads the pickle files that contain:
    - frame_path: path to the RGB image
    - mask_path: path to the affordance mask
    - task_object_class: the object class name
    - question: the reasoning question
    - answer: the ground truth answer
    """

    dataset_type = "3DOI"

    def __init__(self, base_dir, difficulty='easy', split='val'):
        """
        Args:
            base_dir (string): Base directory containing the pickle files.
            difficulty (string): 'easy' or 'hard' (default: 'easy')
            split (string): 'train' or 'val' (default: 'val')
        """
        self.base_dir = base_dir
        self.difficulty = difficulty
        self.split = split
        self.dataset_type = "3DOI"
        
        # Construct the pickle file path
        dataset_name = f'3doi_{difficulty}_reasoning'
        pkl_path = os.path.join(base_dir, f'{dataset_name}_{split}.pkl')
        
        if not os.path.exists(pkl_path):
            raise RuntimeError(f"Pickle file not found: {pkl_path}")
        
        # Load the pickle file
        print(f"Loading 3DOI {difficulty} reasoning {split} data from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            self.data_list = pickle.load(f)
        
        # Filter out broken images
        filtered_data = []
        for data_item in self.data_list:
            # Skip the broken image mentioned in AffordanceNet code
            if 'EK_frame_0000040462.jpg' in data_item['frame_path']:
                print(f"Skipping broken image: {data_item['frame_path']}")
                continue
            filtered_data.append(data_item)
        
        self.data_list = filtered_data
        
        print(f"Loaded {len(self.data_list)} samples from 3DOI {difficulty} reasoning {split} set")
        
        # Organize data by task_object_class (similar to reasoning_aff_reader.py)
        self.images_by_class = {}
        self.labels_by_class = {}
        self.questions_by_class = {}
        self.answers_by_class = {}
        self.indices_by_class = {}
        
        for idx, data_item in enumerate(self.data_list):
            class_name = data_item['task_object_class']
            if class_name not in self.images_by_class:
                self.images_by_class[class_name] = []
                self.labels_by_class[class_name] = []
                self.questions_by_class[class_name] = []
                self.answers_by_class[class_name] = []
                self.indices_by_class[class_name] = []
            
            self.images_by_class[class_name].append(data_item['frame_path'])
            self.labels_by_class[class_name].append(data_item['mask_path'])
            self.questions_by_class[class_name].append(data_item['question'])
            self.answers_by_class[class_name].append(data_item['answer'])
            self.indices_by_class[class_name].append(idx)
        
        print(f"Categories: {list(self.images_by_class.keys())}")
        for class_name, images in self.images_by_class.items():
            print(f"  - {class_name}: {len(images)} samples")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data_item = self.data_list[idx]
        
        # Load image
        image_path = data_item['frame_path']
        image = Image.open(image_path).convert('RGB')
        
        # Load mask
        mask_path = data_item['mask_path']
        mask = Image.open(mask_path)
        mask = np.array(mask)
        
        sample = {
            'image': image,
            'mask': mask,
            'image_path': image_path,
            'mask_path': mask_path,
            'task_object_class': data_item['task_object_class'],
            'question': data_item['question'],
            'answer': data_item['answer']
        }
        
        return sample
    
    def get_class_names(self):
        """Return the list of object class names in the dataset."""
        return list(self.images_by_class.keys())
    
    def get_samples_by_class(self, class_name):
        """Get all samples for a specific class."""
        if class_name not in self.images_by_class:
            raise ValueError(f"Class {class_name} not found in dataset")
        
        indices = self.indices_by_class[class_name]
        return [self[idx] for idx in indices]


if __name__ == '__main__':
    base_dir = ''
    
    print("\n" + "=" * 80)
    print("Testing ThreeDOIReasoningDataset")
    print("=" * 80)
    
    try:
        # Test loading 3DOI easy reasoning validation set
        reasoning_dataset = ThreeDOIReasoningDataset(
            base_dir=base_dir,
            difficulty='easy',
            split='val'
        )
        
        print(f"\nDataset size: {len(reasoning_dataset)}")
        print(f"Class names: {reasoning_dataset.get_class_names()}")
        
        if len(reasoning_dataset) > 0:
            sample = reasoning_dataset[0]
            print(f"\n--- First Sample from Reasoning Dataset ---")
            print(f"Task object class: {sample['task_object_class']}")
            print(f"Question: {sample['question']}")
            print(f"Answer: {sample['answer']}")
            print(f"Image path: {sample['image_path']}")
            print(f"Mask path: {sample['mask_path']}")
            print(f"Image size: {sample['image'].size}")
            print(f"Mask shape: {sample['mask'].shape}")
            
    except Exception as e:
        print(f"Error loading 3DOI reasoning dataset: {e}")
        import traceback
        traceback.print_exc()

