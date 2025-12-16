Given the image of an object, the task is to decide which object to use and predict the part of the object that matches the provided task. The task instruction is "TASK". 
The first image is the original image of the object. The second image is the image of the object interacting with a person or another object in relation to the given affordance type for your reference.

**Follow these reasoning steps**:
1. Identify the key components of the object in the first image (e.g., shape, features, possible points of interaction).
2. Analyze the second image to understand how the object is interacting with a person or another object in relation to the given affordance type.
3. Go back to the first image and ground the part of the object in the image and output the result in a structured JSON format. The part should be represented by a bounding box and a list of key points. The bounding box should be as small as possible and the key points must be accurately located on the object inside the bounding box. No more than 3 key points.

**Coordinate system & units**:
- Image origin is the top-left corner (0, 0).
- x increases to the right; y increases downward.
- Use normalized coordinates in [0,1] for all x,y.
- All x,y values MUST be accurate to 3 decimal places!!! (e.g. 0.111, 0.333, 0.222)

**Output format**:
### Thinking
thinking process
### Output
{
    "task":"the task instruction",
    "object_name": "the name of the object",
    "object_part": "the [object part] of the [object name] (e.g. the blade of the shears)",
    "part_bbox": [x_min, y_min, x_max, y_max],
    "key_points": [[x1, y1], [x2, y2], ...],
}