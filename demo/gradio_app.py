import gradio as gr
import os
import sys
import uuid
import shutil
from PIL import Image
import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Add parent directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import the core function from demo.py
try:
    from demo import run_affordance_detection
    from rex_omni import RexOmniWrapper
except ImportError:
    # If running from src directly, try relative import logic or adjust path
    sys.path.append(current_dir)
    from demo import run_affordance_detection
    from rex_omni import RexOmniWrapper

# Pre-load Models
print("Loading models...")

# Rex-Omni
global_rex_model = None
try:
    print("Loading Rex-Omni model...")
    rex_model_path = "IDEA-Research/Rex-Omni"
    global_rex_model = RexOmniWrapper(
        model_path=rex_model_path,
        backend="transformers",
        max_tokens=4096,
        temperature=0.0,
        top_p=0.05,
        top_k=1,
        repetition_penalty=1.05,
    )
    print("Rex-Omni model loaded successfully!")
except Exception as e:
    print(f"Error loading Rex-Omni model: {e}")

# SAM2
global_sam2_model = None
try:
    print("Loading SAM2 model...")
    global_sam2_model = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")
    print("SAM2 model loaded successfully!")
except Exception as e:
    print(f"Error loading SAM2 model: {e}")


def process_image(image_path, task_instruction):
    """
    Wrapper function for Gradio to run affordance detection.
    """
    if image_path is None:
        return None, None
        
    if not task_instruction:
        return None, None

    # Create a unique output directory for this request
    unique_id = str(uuid.uuid4())[:8]
    output_dir = os.path.join(parent_dir, "output", "gradio", unique_id)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"Processing image: {image_path} with task: {task_instruction}")
        
        if global_sam2_model is None:
            print("Error: SAM2 model is not loaded.")
            return None, None
            
        result = run_affordance_detection(
            image_path=image_path,
            task_instruction=task_instruction,
            output_dir=output_dir,
            model_name="gpt-4o", # Default model
            rex_model=global_rex_model,
            sam2_model=global_sam2_model,
            edit_pipeline=None # Not using dreamer in gradio for now to save memory/speed
        )
        
        if result.get("success"):
            # Return the mask and the visualization
            mask_path = result["mask_path"]
            viz_path = result["visualization_path"]
            
            return mask_path, viz_path
        else:
            print(f"Error: {result.get('error')}")
            return None, None
            
    except Exception as e:
        print(f"Exception occurred: {e}")
        return None, None

# Create Gradio Interface
with gr.Blocks(title="Affordance Prediction Demo") as demo:
    gr.Markdown("# Affordance Prediction Demo")
    gr.Markdown("Upload an image and provide a task instruction (e.g., 'grasp the handle', 'pour water').")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="filepath", label="Input Image")
            task_input = gr.Textbox(label="Task Instruction", placeholder="e.g., grasp the handle")
            submit_btn = gr.Button("Run Prediction", variant="primary")
            
        with gr.Column():
            output_mask = gr.Image(label="Generated Mask")
            output_viz = gr.Image(label="Visualization")
    
    submit_btn.click(
        fn=process_image,
        inputs=[input_image, task_input],
        outputs=[output_mask, output_viz]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
