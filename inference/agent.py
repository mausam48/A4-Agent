import argparse
import os
import sys
from contextlib import ExitStack
import json
from PIL import Image
import numpy as np
import torch
from diffusers import QwenImageEditPlusPipeline
from sam2.sam2_image_predictor import SAM2ImagePredictor
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.image_utils import save_image_with_points_and_box, save_image_with_mask
from utils.agent_utils import post_process
from utils.model_utils import generate_response, read_markdown_file
from dataset.umd_reader import UmdDataset
from dataset.threedoi_reader import ThreeDOIReasoningDataset
from tqdm import tqdm   
import shutil
from rex_omni import RexOmniVisualize, RexOmniWrapper


from dotenv import load_dotenv

load_dotenv()

QWEN_2_5_URL = os.getenv("QWEN_2_5_URL")
QWEN_3_URL = os.getenv("QWEN_3_URL")
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")
SYSTEM_PROMPT_PATH_THINKER_BASE = os.getenv("SYSTEM_PROMPT_THINKER_BASE")
SYSTEM_PROMPT_PATH_THINKER = os.getenv("SYSTEM_PROMPT_THINKER")
SYSTEM_PROMPT_PATH_DREAMER = os.getenv("SYSTEM_PROMPT_DREAMER")



def spotter_detection(rex_model, image, object_part, output_dir):
    """
    Run detection inference pipeline using Rex-Omni.
    """
    categories = [object_part]
    bboxes, points = [], []
    results_bbox = rex_model.inference(images=image, task="detection", categories=categories)

    result = results_bbox[0]
    predictions = result["extracted_predictions"]
    key_list = list(predictions)
    try:
        for i in range(len(predictions[key_list[0]])):
            bbox = predictions[key_list[0]][i]['coords']
            bboxes.append(bbox)
    except:
        bboxes = None

    results_pointing = rex_model.inference(images=image, task="pointing", categories=categories)
    result = results_pointing[0]
    predictions = result["extracted_predictions"]
    key_list = list(predictions)   
    try:
        for i in range(len(predictions[key_list[0]])):
            point = predictions[key_list[0]][i]['coords']
            points.append(point)
    except:
        points = None

    bboxes_rex, points_rex = bboxes, points
    if bboxes_rex == None or points_rex == None:
        print(f"Error: No bounding box or points received from Rex-Omni.")
        return None, None
    
    save_image_with_points_and_box(image, points_rex, bboxes_rex, save_prefix=f"{output_dir}/image_with_rex_grouding") # save image with rex predictions
    
    return bboxes_rex, points_rex

def spotter_segmentation(sam2_model, image, bboxes, points, output_dir, device):
    """
    Run segmentation inference pipeline using SAM2.
    """
    if bboxes is None or points is None:
        print("Skipping SAM2 inference due to missing bboxes or points.")
        return

    # Process with SAM2 locally
    print("Processing with SAM2...")
    try:
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
            image_tensor = np.array(image.convert("RGB"))
            sam2_model.set_image(image_tensor)
            
            mask_all = np.zeros((image_tensor.shape[0], image_tensor.shape[1]), dtype=bool)
            
            # If bboxes list length mismatches points, we will ignore bboxes (points-only)
            # Note: points is a list of points [[x,y], ...], bboxes might be a list of boxes [[x1,y1,x2,y2], ...]
            
            # Ensure inputs are lists
            current_points = points if points else []
            current_bboxes = bboxes if bboxes else []
            
            # If we have points but no bboxes, create None list for bboxes
            if len(current_points) > 0 and len(current_bboxes) != len(current_points):
                # If we have 1 bbox and multiple points, maybe we should reuse the bbox? 
                # Or just ignore bboxes if count doesn't match?
                # Based on sam2_server logic:
                use_zip = current_bboxes and len(current_bboxes) == len(current_points)
                iterable = zip(current_bboxes, current_points) if use_zip else zip([None] * len(current_points), current_points)
            elif len(current_points) > 0:
                iterable = zip(current_bboxes, current_points)
            else:
                # Fallback or handle no points case
                iterable = []

            for bbox_item, point in iterable:
                masks, scores, _ = sam2_model.predict(
                    point_coords=[point],
                    point_labels=[1],
                    box=bbox_item if bbox_item is not None else None,
                    multimask_output=True,
                )
                scores = torch.tensor(scores)
                best_mask_idx = torch.argmax(scores)
                best_mask = masks[best_mask_idx].squeeze()
                mask_all = np.logical_or(mask_all, best_mask)
        
        mask = mask_all
        
        # Save the raw mask as a black and white image
        mask_image_array = (mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_image_array)
        mask_save_path = f"{output_dir}/mask.png"
        mask_image.save(mask_save_path)
        print(f"Raw mask saved to {mask_save_path}")

        save_image_with_mask(
            mask, image, save_prefix=f"{output_dir}/image_with_mask", borders=False
        )
        print(f"Masked image saved to {output_dir}/image_with_mask.png")

    except Exception as e:
        print(f"Error running SAM2: {e}")
        import traceback
        traceback.print_exc()


def dreamer(edit_pipeline, image, prompt, output_dir):
    """
    Edit image using Qwen Image Edit Pipeline.
    """
    inputs = {
        "image": [image],
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }
    # with torch.inference_mode():
    output = edit_pipeline(**inputs)
    output_image = output.images[0]
    edit_image_path = os.path.join(output_dir, "edited_image.png")
    output_image.save(edit_image_path)
    print("image saved at", os.path.abspath(edit_image_path))
    return output_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grounding agent with Rex-Omni")
    parser.add_argument("--resume", action="store_true", help="Skip samples that already have results in output directory")
    parser.add_argument("--model-name", type=str, default="gpt-4o", help="Model name to use for inference")
    parser.add_argument("--dataset-type", type=str, required=True, choices=["UMD", "3DOI", "ReasonAff"], help="Type of dataset to use")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--api-url", type=str, default=None, help="API server URL (default: use API_BASE_URL from env)")
    parser.add_argument("--api-key", type=str, default=None, help="API key for authentication (default: use API_KEY from env)")
    parser.add_argument("--use-dreamer", action="store_true", help="Whether to use dreamer to generate imagined image")
    args = parser.parse_args()

    # Initialize dataset based on type
    if args.dataset_type == "UMD":
        dataset = UmdDataset(root_dir=args.dataset_path)
    elif args.dataset_type == "3DOI":
        dataset = ThreeDOIReasoningDataset(base_dir=args.dataset_path, split='val')
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    #### initialize Rex-Omni model ####
    model_path = "IDEA-Research/Rex-Omni"
    rex_model = RexOmniWrapper(
        model_path=model_path,
        backend="transformers",  # or "vllm" for faster inference
        max_tokens=4096,
        temperature=0.0,
        top_p=0.05,
        top_k=1,
        repetition_penalty=1.05,
    )

    #### initialize SAM2 model ####
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for SAM2: {device}")
    sam2_model = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")

    #### initialize Qwen Image Edit Pipeline ####
    if args.use_dreamer:
        edit_pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16, device_map="balanced")
        edit_pipeline.set_progress_bar_config(disable=None)
    else:
        edit_pipeline = None

    #### initialize MLLM kwargs ####
    model_name = args.model_name
    if 'gpt' in model_name:
        api_url = args.api_url if args.api_url else API_BASE_URL
        api_key = args.api_key if args.api_key else API_KEY
    else:
        api_url = args.api_url if args.api_url else QWEN_2_5_URL
        api_key = args.api_key if args.api_key else ''
    mllm_kwargs = {
        "model_name": model_name,
        "api_url": api_url,
        "api_key": api_key
    }

    if args.use_dreamer:
        thinker_system_prompt = read_markdown_file(SYSTEM_PROMPT_PATH_THINKER)
    else:
        thinker_system_prompt = read_markdown_file(SYSTEM_PROMPT_PATH_THINKER_BASE)
    dreamer_system_prompt = read_markdown_file(SYSTEM_PROMPT_PATH_DREAMER)

    #### start inference ####
    for sample in tqdm(dataset, desc="Predicting Affordance"):
        #### process dataset ####
        if dataset.dataset_type == "UMD":
            dataset_name = "UMD"
            image_path = sample['image_path']
            image = sample['image']
            mask_path = sample['mask_path']
            affordance_type = sample['affordance_type']
            image_paths = [image_path]
            images = [image]
            thinker_prompt = thinker_system_prompt.replace("TASK", "Find the part of the object in the center of the image that can " + affordance_type)
            dreamer_prompt = dreamer_system_prompt + "Find the part of the object that can \" " + affordance_type + "\"."
            save_post_fix = affordance_type

        elif dataset.dataset_type == "3DOI":
            dataset_name = "3DOI"
            image_path = sample['image_path']
            image = sample['image']
            mask_path = sample['mask_path']
            question = sample['question']
            answer = sample['answer']
            # import pdb; pdb.set_trace()
            image_paths = [image_path]
            images = [image]
            thinker_prompt = thinker_system_prompt.replace("TASK", question)
            dreamer_prompt = dreamer_system_prompt + question
            save_post_fix = ''
        
        
        #### get sample name ####
        sample_name = os.path.basename(image_paths[0]).split(".")[0]
        
        #### apply dataset-specific naming rules ####
        if dataset.dataset_type == "UMD" and save_post_fix:
            # UMD: {image_name}_{affordance_type}
            sample_dir_name = f"{sample_name}_{save_post_fix}"
        else:
            sample_dir_name = sample_name   
        
        output_dir = os.path.join("output", dataset_name, model_name, sample_dir_name)
        os.makedirs(output_dir, exist_ok=True)

        # Check if results already exist (resume mode)
        if args.resume:
            mask_file = os.path.join(output_dir, "mask.png")
            if os.path.exists(mask_file):
                print(f"Skipping {output_dir} - results already exist")
                continue

        ## copy gt_mask to output dir and rename to gt_mask.png
        if 'mask_path' in sample and sample['mask_path'] and os.path.exists(sample['mask_path']):
            shutil.copy(sample['mask_path'], os.path.join(output_dir, "gt_mask.png"))
        ## copy original image to output dir and rename to original_image.png
        shutil.copy(image_paths[0], os.path.join(output_dir, "original_image.png"))

        if args.use_dreamer:
            print("[1/3] Running Dreamer")
            dreamer_response = generate_response(
                **mllm_kwargs,
                prompt_text=dreamer_prompt,
                images=images
            )
            print(f"Dreamer response: {dreamer_response}")
            with open(os.path.join(output_dir, "dreamer.txt"), "w") as f:
                f.write(dreamer_response)
            imagined_image = dreamer(edit_pipeline, images[0], dreamer_response, output_dir)
            images.append(imagined_image)
        else:
            print("[1/3] Skipping Dreamer")


        #### generate thinker response ####
        print("[2/3] Running Thinker")
        thinker_response = generate_response(
            **mllm_kwargs,
            prompt_text=thinker_prompt,
            images=images
        )
        bboxes_mllm, points_mllm, object_part = post_process(thinker_response, images, output_dir)
        print(f"Object part: {object_part}")

        #### Spotter Inference ####
        print("[3/3] Running Spotter")
        bboxes, points = spotter_detection(rex_model, images[0], object_part, output_dir)
        spotter_segmentation(sam2_model, images[0], bboxes, points, output_dir, device)
