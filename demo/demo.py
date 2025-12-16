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
from rex_omni import RexOmniVisualize, RexOmniWrapper
from dotenv import load_dotenv

load_dotenv()

QWEN_2_5_URL = os.getenv("QWEN_2_5_URL")
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")

# System prompts paths
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
        return None

    # Process with SAM2 locally
    print("Processing with SAM2...")
    try:
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
            image_tensor = np.array(image.convert("RGB"))
            sam2_model.set_image(image_tensor)
            
            mask_all = np.zeros((image_tensor.shape[0], image_tensor.shape[1]), dtype=bool)
            
            # Ensure inputs are lists
            current_points = points if points else []
            current_bboxes = bboxes if bboxes else []
            
            # Logic to handle bbox/point matching from agent.py
            if len(current_points) > 0 and len(current_bboxes) != len(current_points):
                use_zip = current_bboxes and len(current_bboxes) == len(current_points)
                iterable = zip(current_bboxes, current_points) if use_zip else zip([None] * len(current_points), current_points)
            elif len(current_points) > 0:
                iterable = zip(current_bboxes, current_points)
            else:
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
        
        return mask_save_path

    except Exception as e:
        print(f"Error running SAM2: {e}")
        import traceback
        traceback.print_exc()
        return None


def dreamer(edit_pipeline, image, prompt, output_dir):
    """
    Edit image using Qwen Image Edit Pipeline.
    """
    if edit_pipeline is None:
        print("Dreamer pipeline not initialized.")
        return image

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


def run_affordance_detection(image_path, task_instruction, output_dir, 
                             model_name="gpt-4o", 
                             sam2_model=None,
                             rex_model=None,
                             edit_pipeline=None,
                             use_dreamer=False):
    """Main function to run affordance detection on a single image.
    
    Args:
        image_path: Path to input image
        task_instruction: Task description (e.g., "grasp the handle", "pour water")
        output_dir: Directory to save results
        model_name: LLM model to use
        sam2_model: Loaded SAM2 model
        rex_model: Loaded Rex-Omni model
        edit_pipeline: Loaded Qwen Image Edit pipeline
        use_dreamer: Whether to use dreamer to generate imagined image
    
    Returns:
        dict: Results containing mask path and other outputs
    """
    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    images = [image]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save a copy of the input image
    image.save(os.path.join(output_dir, "original_image.png"))

    # Load prompts
    if use_dreamer:
        if SYSTEM_PROMPT_PATH_THINKER and os.path.exists(SYSTEM_PROMPT_PATH_THINKER):
            thinker_system_prompt = read_markdown_file(SYSTEM_PROMPT_PATH_THINKER)
        else:
            print("Warning: Thinker prompt for dreamer not found, using base.")
            thinker_system_prompt = read_markdown_file(SYSTEM_PROMPT_PATH_THINKER_BASE)
            
        if SYSTEM_PROMPT_PATH_DREAMER and os.path.exists(SYSTEM_PROMPT_PATH_DREAMER):
            dreamer_system_prompt = read_markdown_file(SYSTEM_PROMPT_PATH_DREAMER)
        else:
            print("Warning: Dreamer prompt not found.")
            dreamer_system_prompt = ""
    else:
        if SYSTEM_PROMPT_PATH_THINKER_BASE and os.path.exists(SYSTEM_PROMPT_PATH_THINKER_BASE):
            thinker_system_prompt = read_markdown_file(SYSTEM_PROMPT_PATH_THINKER_BASE)
        else:
            # Fallback for demo if env vars not set, though ideally they should be
            print("Warning: Base Thinker prompt not found.")
            thinker_system_prompt = "You are an intelligent agent designed to find the specific part of an object..."

    # Prepare prompts
    thinker_prompt = thinker_system_prompt.replace("TASK", task_instruction)
    
    # MLLM args for dreamer and thinker
    api_url = API_BASE_URL if 'gpt' in model_name else QWEN_2_5_URL
    api_key = API_KEY if 'gpt' in model_name else ''
    mllm_kwargs = {
        "model_name": model_name,
        "api_url": api_url,
        "api_key": api_key,
    }

    if use_dreamer:
        dreamer_prompt = dreamer_system_prompt + task_instruction
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

    print("[2/3] Running Thinker")
    # Generate response from LLM    
    thinker_response = generate_response(
        **mllm_kwargs,
        prompt_text=thinker_prompt,
        images=images
    )
    print(f"Thinker Response:\n{thinker_response}\n")
    
    # Save raw response
    with open(os.path.join(output_dir, "thinker_response.txt"), "w") as f:
        f.write(thinker_response)
    
    # Post-process response
    try:
        bboxes_mllm, points_mllm, object_part = post_process(thinker_response, images, output_dir)
        print(f"Object part: {object_part}")
    except Exception as e:
        print(f"Error extracting grounding information: {e}")
        return {"error": str(e)}
    
    print("[3/3] Running Spotter")
    
    bboxes, points = spotter_detection(rex_model, images[0], object_part, output_dir)    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_path = spotter_segmentation(sam2_model, images[0], bboxes, points, output_dir, device)
    
    if mask_path:
        return {
            "success": True,
            "mask_path": mask_path,
            "visualization_path": f"{output_dir}/image_with_mask.png",
            "object_part": object_part,
            "bboxes": bboxes,
            "points": points
        }
    else:
        return {"error": "SAM2 failed to generate mask"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo: Affordance detection with custom image and task instruction",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--task", type=str, required=True, help="Task instruction")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model name")
    parser.add_argument("--use-dreamer", action="store_true", help="Use dreamer to generate imagined image")
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output is None:
        image_basename = os.path.splitext(os.path.basename(args.image))[0]
        args.output = os.path.join("output", "demo", image_basename)
    
    print("="*60)
    print("Affordance Detection Demo")
    print("="*60)
    
    # Initialize Models
    print("Initializing models...")
    
    # Rex-Omni
    print("Loading Rex-Omni...")
    model_path = "IDEA-Research/Rex-Omni"
    rex_model = RexOmniWrapper(
        model_path=model_path,
        backend="transformers",
        max_tokens=4096,
        temperature=0.0,
        top_p=0.05,
        top_k=1,
        repetition_penalty=1.05,
    )

    # SAM2
    print("Loading SAM2...")
    sam2_model = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")

    # Qwen Image Edit
    if args.use_dreamer:
        print("Loading Qwen Image Edit Pipeline...")
        edit_pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16, device_map="balanced")
        edit_pipeline.set_progress_bar_config(disable=None)
    else:
        edit_pipeline = None

    print("Models loaded.")
    print("="*60)

    # Run detection
    result = run_affordance_detection(
        image_path=args.image,
        task_instruction=args.task,
        output_dir=args.output,
        model_name=args.model,
        sam2_model=sam2_model,
        rex_model=rex_model,
        edit_pipeline=edit_pipeline,
        use_dreamer=args.use_dreamer
    )
    
    print()
    print("="*60)
    if result.get("success"):
        print("✓ Detection completed successfully!")
        print(f"  Mask: {result['mask_path']}")
        print(f"  Visualization: {result['visualization_path']}")
        print(f"  Detected: {result['object_part']}")
    else:
        print("✗ Detection failed!")
        print(f"  Error: {result.get('error', 'Unknown error')}")
    print("="*60)
