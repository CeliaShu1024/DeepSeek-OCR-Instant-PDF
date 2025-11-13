# DeepSeek-OCR-Instant-PDF
Quick python demo for processing pdf with DeepSeek-OCR and result normalization. For the `.mmd` files obtained from DeepSeek-OCR, you can use the `mmdNormalizer.py` to transform it to the following json structure:

```json
{
	"id": filename (without ".mmd"),
	"sections":{
		"section":{
			"sid": incremental int,
			"heading": "" | "heading content...",
			"heading_level": int,
			"contents": "" | [
				"paragraph 1",
				"paragraph 2",
				...
			]
		},
		...
	}
}
```

where `heading_level` refers to the last heading the program detects for this section. We also added an approach to enhance compatibility with multi-page pdf files. A sample input and output case is given:

**Input**
```markdown
# Chapter 1
Some text split by

<--- Page Split --->

page boundary.

## Section 1.1
More content here.
```

**Output JSON:**
```json
{
  "id": "document",
  "sections": [
    {
      "sid": 1,
      "heading": "Chapter 1",
      "heading_level": 1,
      "contents": ["Some text split by page boundary."]
    },
    {
      "sid": 2,
      "heading": "Chapter 1: Section 1.1",
      "heading_level": 2,
      "contents": ["More content here."]
    }
  ]
}
```

## Prerequisites
This is an application based on DeepSeek-OCR. Please follow the instructions on the official installation of [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR).

```bash
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd DeepSeek-OCR

# conda env 
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr

# install DeepSeek-OCR
wget https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation

# download DeepSeek-OCR model
hf download deepseek-ai/DeepSeek-OCR --local-dir model # change 'model' to your custom path
```

---

# Scripts Documentation

This directory contains various scripts for processing PDF documents using DeepSeek-OCR and normalizing the results.

## Overview

The scripts are organized into three main categories:
1. **PDF Processing** - Convert PDFs to Markdown using DeepSeek-OCR
2. **Result Normalization** - Clean and structure the markdown output
3. **Configuration** - Centralized configuration management

---

## Configuration

### `config.py`

Central configuration file for all processing scripts.

**Key Settings:**
```python
# Image Processing Modes
BASE_SIZE = 1024      # Base image size for global view
IMAGE_SIZE = 640      # Image size for local patches
CROP_MODE = True      # Enable dynamic cropping
MIN_CROPS = 2         # Minimum crop tiles
MAX_CROPS = 6         # Maximum crop tiles (reduce if low GPU memory)

# Processing Settings
MAX_CONCURRENCY = 100  # Batch size (reduce if low GPU memory)
NUM_WORKERS = 64       # Image preprocessing workers
SKIP_REPEAT = True     # Skip pages without EOS token

# Model
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'  # Path to model

# Prompt
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
```

**Preset Modes:**
- **Tiny**: `base_size=512, image_size=512, crop_mode=False`
- **Small**: `base_size=640, image_size=640, crop_mode=False`
- **Base**: `base_size=1024, image_size=1024, crop_mode=False`
- **Large**: `base_size=1280, image_size=1280, crop_mode=False`
- **Gundam** (recommended): `base_size=1024, image_size=640, crop_mode=True`

---

## PDF Processing Scripts

### 1. `pdf_processor.py`

Core PDF processing class with DeepSeek-OCR integration.

**Features:**
- PDF to image conversion
- OCR inference with layout detection
- Bounding box visualization
- Image extraction from documents

**Usage (as module):**
```python
from pdf_processor import PDFProcessor

processor = PDFProcessor()

result = processor.process_pdf(
    input_path="document.pdf",
    output_dir="output/",
    prompt="<image>\n<|grounding|>Convert the document to markdown.",
    skip_repeat=True,
    crop_mode=True
)

processor.cleanup()
```

**Output:**
```
output/
├── document.mmd                # Clean markdown
├── document_det.mmd            # Markdown with detection tags
├── document_layouts.pdf        # Annotated PDF with bounding boxes
└── images/                     # Extracted images
    ├── 0_0.jpg
    ├── 0_1.jpg
    └── ...
```

---

### 2. `task_parallel.py`

**Multi-GPU batch processing** - Distributes PDFs across multiple GPUs for parallel processing.

**When to use:** Processing multiple PDFs with multiple GPUs available.

**Usage:**
```bash
# Process directory with 2 GPUs
python task_parallel.py -p pdfs/ -o output --num-gpus 2

# Use specific GPUs
python task_parallel.py -p pdfs/ -o output --gpu-ids 0,3

# Process single PDF
python task_parallel.py -p document.pdf -o output
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `-p, --path` | Input PDF file or directory | Required |
| `-o, --output` | Output directory | Required |
| `--num-gpus` | Number of GPUs to use | 2 |
| `--gpu-ids` | Specific GPU IDs (e.g., '0,1,3') | None |
| `--prompt` | Custom OCR prompt | From config |
| `--dpi` | Image conversion DPI | 144 |
| `--no-skip-repeat` | Process all pages | False |
| `--no-crop-mode` | Disable crop mode | False |

**How it works:**
- Splits PDF files across multiple GPUs
- Each GPU runs a separate process (`pdf_worker.py`)
- Processes PDFs in parallel for faster throughput

---

### 3. `tensor_parallel.py`

**Tensor parallelism** - Splits the model across multiple GPUs to reduce memory usage per GPU.

**When to use:** 
- Large PDFs that require high max_model_len
- Limited GPU memory per device
- Prefer throughput over latency

**Usage:**
```bash
# Use 2 GPUs with tensor parallelism
python tensor_parallel.py -p document.pdf -o output --tensor-parallel-size 2

# Use specific GPUs
python tensor_parallel.py -p document.pdf -o output --tensor-parallel-size 2 --gpu-ids 0,1

# Process directory (sequential)
python tensor_parallel.py -p pdfs/ -o output --tensor-parallel-size 4
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `-p, --path` | Input PDF file or directory | Required |
| `-o, --output` | Output directory | Required |
| `--tensor-parallel-size` | Number of GPUs to split model across | 2 |
| `--gpu-ids` | Specific GPU IDs | None |
| `--model-path` | Path to model | From config |
| `--prompt` | Custom OCR prompt | From config |
| `--max-model-len` | Maximum sequence length | 8192 |
| `--gpu-memory-utilization` | GPU memory ratio | 0.9 |

**Memory Comparison:**

| Mode | Task Parallel | Tensor Parallel |
|------|--------------|-----------------|
| GPU 0 | 40GB (full model) | 20GB (half model) |
| GPU 1 | 40GB (full model) | 20GB (half model) |
| Total | 80GB, 2 PDFs parallel | 40GB, 1 PDF sequential |

---

### 4. `pdf_worker.py`

Worker script for `task_parallel.py` - **Do not run directly.**

This script is automatically launched by `task_parallel.py` for each GPU worker.

---

## Result Normalization

### `mmdNormalizer.py`

Converts markdown output to structured JSON format with proper section hierarchy.

**Features:**
- Removes page split markers
- Reconnects split paragraphs across pages
- Extracts hierarchical section structure
- Combines nested headings (e.g., "Chapter 1: Introduction: Background")

**Usage:**
```bash
# Single file
python mmdNormalizer.py -p output/document.mmd

# Directory
python mmdNormalizer.py -p output/ -o normalized/
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `-p, --path` | Input markdown file or directory |
| `-o, --output` | Output directory (optional, defaults to input location) |

---

## Workflow Examples

### Example 1: Process Single PDF
```bash
# 1. Process PDF
python -c "
from pdf_processor import PDFProcessor
processor = PDFProcessor()
processor.process_pdf('document.pdf', 'output/')
processor.cleanup()
"

# 2. Normalize result
python mmdNormalizer.py -p output/document.mmd
```

### Example 2: Batch Process with Multiple GPUs
```bash
# Process 100 PDFs using 4 GPUs
python task_parallel.py -p pdfs/ -o output --num-gpus 4

# Normalize all results
python mmdNormalizer.py -p output/ -o normalized/
```

### Example 3: Process Large PDF with Limited GPU Memory
```bash
# Use tensor parallelism to split model across 2 GPUs
python tensor_parallel.py \
  -p large_document.pdf \
  -o output \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85

# Normalize
python mmdNormalizer.py -p output/large_document.mmd
```

---

## Choosing the Right Script

| Scenario | Recommended Script | Reason |
|----------|-------------------|---------|
| Single PDF, 1 GPU | `pdf_processor.py` | Simple and direct |
| Multiple PDFs, multiple GPUs | `task_parallel.py` | Parallel processing for speed |
| Large PDF, limited memory | `tensor_parallel.py` | Distribute model across GPUs |
| Very large PDF (>100 pages) | `tensor_parallel.py` with high `max_model_len` | More tokens for long documents |

---

## Common Prompts

Configure in `config.py` or pass via `--prompt`:
```python
# Document OCR (default)
"<image>\n<|grounding|>Convert the document to markdown."

# Free OCR (no layout detection)
"<image>\nFree OCR."

# Figure parsing
"<image>\nParse the figure."

# General description
"<image>\nDescribe this image in detail."

# Object localization
"<image>\nLocate <|ref|>text here<|/ref|> in the image."
```

---

## Troubleshooting

### Out of Memory (OOM)

1. Reduce `MAX_CROPS` in `config.py` (try 4 or 2)
2. Reduce `MAX_CONCURRENCY` (try 50 or 25)
3. Use `tensor_parallel.py` to split model
4. Lower `--gpu-memory-utilization` (try 0.8)

### Pages Being Skipped

Set `SKIP_REPEAT = False` in `config.py` or use `--no-skip-repeat`

### Poor OCR Quality

1. Increase `IMAGE_SIZE` and `BASE_SIZE`
2. Enable `CROP_MODE = True`
3. Try different prompts
4. Increase DPI: `--dpi 200`