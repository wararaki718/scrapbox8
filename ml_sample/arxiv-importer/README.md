# arXiv Importer: PyTorch Model Generator

This tool analyzes machine learning papers from arXiv and automatically generates PyTorch implementation code for the described model architectures.

## Features

- **Paper Retrieval**: Downloads a paper's PDF directly from an arXiv URL or ID.
- **Architecture Analysis**: Leverages a Large Language Model (LLM) to identify and extract key components of a model architecture, such as layers, activation functions, and input/output dimensions from the "Method" or "Architecture" sections of the paper.
- **Code Generation**:
    - Produces a `torch.nn.Module` class definition based on the extracted architectural information.
    - Includes comments in the generated code to trace back to the corresponding descriptions in the paper.
    - Adds a docstring to the `forward` method, detailing the tensor shape transformations.

## Tech Stack

- **Language**: Python 3.9+
- **PDF Parsing**: `pypdf`
- **LLM**: Google Gemini 1.5 Pro (`google-generativeai`)
- **Deep Learning Framework**: `torch`
- **HTTP Requests**: `requests`

## Setup

1.  **Clone the repository and navigate to the project directory:**
    ```bash
    git clone <repository-url>
    cd <repository-path>/ml_sample/arxiv-importer
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Set up your Google API Key:**
    You need to set your Google Gemini API key as an environment variable. This is required for the code generation step.
    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY"
    ```

## How to Run

Execute the `main.py` script and provide the URL of the arXiv paper you want to process.

```bash
python main.py
```

The script will then prompt you to enter an arXiv URL:
```
Enter the arXiv paper URL (e.g., https://arxiv.org/abs/1706.03762): 
```

The tool will perform the following steps:
1.  Download the PDF of the paper.
2.  Extract the text content.
3.  Send the text to the Gemini API to generate the PyTorch model code.
4.  Validate the syntax of the generated code.
5.  Save the valid code to the `generated_models/` directory and print it to the console.

### Example

If you input the URL for the "Attention Is All You Need" paper (`https://arxiv.org/abs/1706.03762`), the tool will attempt to generate the PyTorch code for the Transformer model and save it as `generated_models/Model_1706_03762.py`.
