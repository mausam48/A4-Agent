from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn
import io
import os
import json
import time
import numpy as np
from PIL import Image
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.image_utils import save_image_with_points_and_box, save_image_with_mask
import argparse


app = FastAPI(title="SAM2 Segmentation API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load SAM2 predictor once
segmentation_model = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")


def _parse_prompts(
    bbox_json: Optional[str],
    points_json: Optional[str],
):
    """Parse prompts from explicit bbox/points JSON strings.

    Returns:
        bboxes: List[List[float]]
        points: List[List[float]]
    """
    bboxes: List[List[float]] = []
    points: List[List[float]] = []

    # Override or complement with explicit params if provided
    if bbox_json:
        try:
            parsed_bbox = json.loads(bbox_json)
            if isinstance(parsed_bbox, list) and len(parsed_bbox) == 4:
                bboxes = [parsed_bbox]
            elif isinstance(parsed_bbox, list) and all(
                isinstance(x, list) and len(x) == 4 for x in parsed_bbox
            ):
                bboxes = parsed_bbox
        except json.JSONDecodeError:
            pass

    if points_json:
        try:
            parsed_points = json.loads(points_json)
            if (
                isinstance(parsed_points, list)
                and all(isinstance(x, list) and len(x) == 2 for x in parsed_points)
            ):
                points = parsed_points
        except json.JSONDecodeError:
            pass

    return bboxes, points


@app.post("/segment")
async def segment(
    image: UploadFile = File(...),
    bbox: Optional[str] = Form(None),  # JSON string: [x1,y1,x2,y2] or list of such
    points: Optional[str] = Form(None),  # JSON string: [[x,y], [x,y], ...]
    multimask_output: bool = Form(True),
):
    start_time = time.time()

    # Load image
    image_content = await image.read()
    pil_image = Image.open(io.BytesIO(image_content)).convert("RGB")

    # Parse explicit bbox/points
    bboxes, pts = _parse_prompts(bbox, points)
    if not pts:
        return {"error": "No points provided. Supply points JSON."}

    # Run segmentation
    with torch.inference_mode(), torch.autocast(
        "cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16
    ):
        image_tensor = np.array(pil_image)
        mask_all = np.zeros((image_tensor.shape[0], image_tensor.shape[1]), dtype=bool)

        segmentation_model.set_image(image_tensor)

        # If bboxes list length mismatches points, we will ignore bboxes (points-only)
        use_zip = bboxes and len(bboxes) == len(pts)
        iterable = zip(bboxes, pts) if use_zip else zip([None] * len(pts), pts)

        for bbox_item, point in iterable:
            masks, scores, _ = segmentation_model.predict(
                point_coords=[point],
                point_labels=[1],
                box=bbox_item if bbox_item is not None else None,
                multimask_output=multimask_output,
            )
            scores = torch.tensor(scores)
            best_mask_idx = torch.argmax(scores)
            best_mask = masks[best_mask_idx].squeeze()
            mask_all = np.logical_or(mask_all, best_mask)

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Request processed in {processing_time:.2f} seconds.")

    return {
        "mask": mask_all.tolist(),
        "processing_time_sec": round(processing_time, 3),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 Segmentation API Server")
    parser.add_argument("-p", "--port", type=int, default=8010, help="Port to run the server on (default: 8010)")
    parser.add_argument("-H", "--host", type=str, default="0.0.0.0", help="Host to run the server on (default: 0.0.0.0)")
    args = parser.parse_args()

    print(f"Starting SAM2 server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


