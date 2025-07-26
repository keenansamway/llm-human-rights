#!/bin/bash

# Array of model-task pairs
model_tasks=(
    "openai/gpt-3.5-turbo-0125 paragraph"
    "openai/gpt-4o-2024-11-20 paragraph"
    "meta-llama/llama-3.3-70b-instruct paragraph"
    "meta-llama/llama-4-maverick-17b-128e-instruct paragraph"
    "qwen/qwen-2.5-72b-instruct paragraph"
    "deepseek/deepseek-chat-v3-0324 paragraph"

    "openai/gpt-3.5-turbo-0125 likert"
    "openai/gpt-4o-2024-11-20 likert"
    "meta-llama/llama-3.3-70b-instruct likert"
    "meta-llama/llama-4-maverick-17b-128e-instruct likert"
    "qwen/qwen-2.5-72b-instruct likert"
    "deepseek/deepseek-chat-v3-0324 likert"
    "anthropic/claude-sonnet-4 likert"
)

# Run scenario_evaluation.py for each model-task pair
for model_task in "${model_tasks[@]}"; do
    read -r model task <<< "$model_task"
    echo "Running scenario_evaluation.py with model: $model and task: $task"
    python scenario_evaluation.py --model "$model" --task "$task" --samples 5 --languages en zh-cn ro
    echo "Completed $model"
    echo "---"
done

echo "All models completed!"
