# DeepSeek-OCR-Instant-PDF
Quick python demo for processing pdf with DeepSeek-OCR and result normalization.

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

