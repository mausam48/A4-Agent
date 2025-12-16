# Demo

## Using Open-source Qwen-2.5VL as the `Thinker`
python demo/demo.py --image /path/to/your/image --task "your task instruction" --model qwen2.5-vl-7b
## with Dreamer module
python demo/demo.py --image /path/to/your/image --task "your task instruction" --model qwen2.5-vl-7b --use-dreamer

## Using Proprietary MLLMs as the `Thinker`
python demo/demo.py --image /path/to/your/image --task "your task instruction" --model gpt-4o
## with Dreamer module
python demo/demo.py --image /path/to/your/image --task "your task instruction" --model gpt-4o --use-dreamer

# Gradio UI
python demo/gradio_app.py


# Inference on 3DOI dataset
python inference/agent.py \
  --dataset-type 3DOI \
  --dataset-path /path/to/3DOI/dataset \
  --model-name qwen2.5-vl-7b \
  --resume 
# with Dreamer module
python inference/agent.py \
  --dataset-type 3DOI \
  --dataset-path /path/to/3DOI/dataset \
  --model-name qwen2.5-vl-7b \
  --resume \
  --use-dreamer

