# LRM-FactEval

A Python tool for running different model sampling and evaluations.

## Features

- Support for multiple evaluation tasks: SimpleQA, triviaqa
- Support for various model samplers: ChatCompletion, Claude, openai, etc.
- Automatic generation of HTML reports and JSON results
- Support for debug mode and custom sample sizes

## Requirements

```bash
pip install pandas argparse openai
```

## Usage

### Basic Usage

```bash
python simple_evals.py
```

### Command Line Arguments
- `--debug`: Enable debug mode (use fewer samples)
- `--examples N`: Specify the number of samples to use

### Examples

```bash
# Debug mode
python simple_evals.py --debug

# Specify sample count
python simple_evals.py --examples 100
```

## Configuration

### Model Configuration

```python
model_name = "Qwen2.5-32B"           # Model name
model_dir = "Qwen/Qwen2.5-32B"       # Model path
model_use_chat = False               # Whether to use chat mode
model_enable_thinking = False        # Whether to enable thinking mode
model_max_tokens = 30000             # Maximum token count
model_config = "qwen"                # Model configuration type
data_path = "trivia"                 # Data path
```

### Grader Configuration

```python
grading_sampler = ChatCompletionSampler(
    model="Qwen/Qwen3-32B",
    model_name=model_name+data_path,
    url=judge_url,
    use_chat=True,
    enable_thinking=True,
    max_tokens=30000
)
```


## Output Files

- **HTML Reports**: `/tmp/{eval_name}_{model_name}.html`
- **JSON Results**: `/tmp/{eval_name}_{model_name}.json`
- **Debug Mode**: Files will have `_DEBUG` suffix

## Result Format

The program generates results containing the following metrics:
- `score`: Overall score
- `f1_score`: F1 score (if applicable)
- Other evaluation-specific metrics

Finally, it outputs a summary table showing all models' performance across evaluation tasks.

## Important Notes

1. Ensure correct API keys and URLs are set
2. Debug mode uses fewer samples, suitable for quick testing
3. Full evaluations may take considerable time
4. Result files are saved in `/tmp` directory


## Troubleshooting

- If you encounter model not found errors, check if the model name is correct
- Ensure network connectivity to access model APIs
- Check if there's sufficient disk space to store result files
