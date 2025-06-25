# LRM-FactEval


## Features

- Support for multiple evaluation tasks: SimpleQA, triviaqa
- Support for various model samplers: ChatCompletion, Claude, openai, etc.
- Automatic generation of HTML reports and JSON results
- Support for debug mode and custom sample sizes

## Requirements

```bash
pip install pandas argparse openai
```
# Simple Evaluations

A Python tool for running different model sampling and evaluations.

## Features

- Support for multiple evaluation tasks: MMLU, Math, GPQA, MGSM, DROP, HumanEval, SimpleQA, BrowseComp
- Support for various model samplers: ChatCompletion, Claude, O-Chat, etc.
- Automatic generation of HTML reports and JSON results
- Support for debug mode and custom sample sizes

## Model Deployment

**Important**: Before running evaluations, you need to deploy open-source models as API servers using vLLM or SGLang with OpenAI-compatible format.

### Option 1: Using vLLM

```bash
# Install vLLM
pip install vllm
```

### Option 2: Using SGLang

```bash
# Install SGLang
pip install sglang[all]
```

### API Configuration

Once your model server is running, configure the URLs in your evaluation script:

```python
model_url = "http://localhost:8000/v1"  # Your model API endpoint
judge_url = "http://localhost:8001/v1"  # Your judge model API endpoint
```

## Environment Variables

```bash
export OPENAI_API_KEY="your-api-key"
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
# repetition judge prompt
```python
f"""You are an expert reviewer specialized in evaluating the quality of AI thinking processes. You need to analyze the AI thinking process provided below and identify whether there is SEVERE logical repetition.

Logical repetition refers to when an AI repeatedly states the same or very similar reasoning steps, arguments, or conclusions without making substantive progress. However, for this task, you should ONLY identify SEVERE cases of repetition.

SEVERE logical repetition is characterized by:
1. Extensive repetition where the same exact reasoning steps are repeated multiple times with minimal or no variation
2. Clear cases where the AI is stuck in a loop, repeating almost identical content for multiple paragraphs
3. Substantial portions (at least 30%+) of the thinking process that add no new information and merely restate previous points
4. Obvious circular reasoning that makes the thinking process inefficient and unnecessarily lengthy

Minor repetition, such as briefly restating a point for emphasis or clarity, summarizing previous steps, or revisiting an idea with substantial new insight should NOT be considered severe repetition.

Here's an example of SEVERE repetition that should be marked as "YES":

'''
To find the length of the hypotenuse, I need to use the Pythagorean theorem. The Pythagorean theorem states that a² + b² = c², where a and b are the legs of the right triangle and c is the hypotenuse.

Given that a = 3 and b = 4, I need to substitute these values into the formula.
a² + b² = c²
3² + 4² = c²
9 + 16 = c²
25 = c²
c = 5

To solve this problem, I need to apply the Pythagorean theorem. The Pythagorean theorem tells us that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides. So a² + b² = c².

Let me substitute the values a = 3 and b = 4 into the Pythagorean formula.
a² + b² = c²
3² + 4² = c²
9 + 16 = c²
25 = c²
So c = 5

Actually, let me double-check my work using the Pythagorean theorem again. The formula is a² + b² = c², where a and b are the legs, and c is the hypotenuse.
With a = 3 and b = 4:
3² + 4² = c²
9 + 16 = c²
25 = c²
c = 5

Therefore, the length of the hypotenuse is 5.
'''

Here's an example of a GOOD thinking process without severe repetition that should be marked as "NO":

'''
Okay, let's see. The user is asking about the IEEE Frank Rosenblatt Award in 2010. I need to figure out who received that award.

First, I should recall what the Frank Rosenblatt Award is. From what I remember, the IEEE has various awards named after notable figures in the field of electrical engineering and computer science. Frank Rosenblatt is known for his work on the perceptron, which was an early neural network model. The award is likely related to neural networks or machine learning.

Now, the year in question is 2010. I need to find out who was awarded that year. I don't have the exact information memorized, so I might need to think of notable people in AI or neural networks around that time. Maybe someone like Geoffrey Hinton? He was active in the field, but I'm not sure about the exact year. Or perhaps Yann LeCun? He's another prominent figure in deep learning. Alternatively, maybe someone from the robotics or control systems area, since IEEE covers a broad range.

Wait, the Frank Rosenblatt Award is specifically for contributions to neural networks and machine learning. Let me check my memory. In 2010, the award might have gone to someone who made significant contributions around that time. Another possibility is Andrew Ng, but I think he was more prominent in the mid-2000s. Maybe someone else. Alternatively, the award might have been given to a team or an organization, but usually, it's an individual.

Alternatively, I can think of the IEEE website or databases, but since I can't access external resources, I need to rely on my existing knowledge. I recall that the 2010 IEEE Frank Rosenblatt Award was given to Geoffrey Hinton. Wait, but I'm not entirely certain. Another name that comes to mind is Yann LeCun, but I think he received it in a different year. Let me think again. Alternatively, maybe the award was given to someone like David Rumelhart, but he passed away earlier. 

Wait, I think the 2010 recipient was actually Geoffrey Hinton. However, I'm not 100% sure. Alternatively, maybe the answer is someone else. If I'm not certain, I should state that I can't recall the exact name but provide the context about the award. However, the user is expecting an answer, so I need to be as accurate as possible. Another angle: sometimes the award is given to a group. But I think it's more likely an individual. Let me try to remember. In 2010, Geoffrey Hinton was working on deep learning and had some breakthroughs, so that might be the connection. Alternatively, maybe it was someone like Raj Reddy, but he's more associated with AI in general. 

Alternatively, I might be confusing it with the Turing Award. The IEEE Frank Rosenblatt Award is a more specific award. Another possibility is that the 2010 recipient was not a well-known name, but I can't recall. Given that I can't be certain, but based on the information I have, I'll go with Geoffrey Hinton as the most probable answer, even though I'm not 100% sure.
'''

Notice that in the GOOD example, although the AI considers multiple possibilities and revisits some names, it's progressively exploring different candidates, offering context, and providing new reasoning each time. The word "alternatively" appears multiple times, but each instance introduces a new possibility or perspective, not merely repeating the same points.

Please carefully analyze the following thinking process with these strict criteria in mind:

===THINKING PROCESS STARTS===
{thinking_process}
===THINKING PROCESS ENDS===

After analyzing the thinking process, simply respond with ONLY ONE of these answers:
- "YES" if there is SEVERE logical repetition as defined above and illustrated in the first example
- "NO" if there is no severe logical repetition (even if some minor repetition exists) as shown in the second example

Your response should contain only the word "YES" or "NO" with no other text."""

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
## Troubleshooting

- **Model Server Issues**: Ensure your vLLM/SGLang server is running and accessible or Ensure correct API keys and URLs are set
- **API Connection**: Verify the model URLs are correct and servers are responding
- If you encounter model not found errors, check if the model name is correct
- Ensure network connectivity to access model APIs
- Check if there's sufficient disk space to store result files
- Result files are saved in `/tmp` directory
- **Memory Issues**: If using vLLM, you may need to adjust `gpu_memory_utilization` or `max_model_len` parameters
