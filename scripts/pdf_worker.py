"""
PDF Worker Script for Multi-GPU Processing

Worker process that handles PDF-to-Markdown conversion on a specific GPU.
This script is launched by main.py and should not be run directly.

USAGE:
    python pdf_worker.py <gpu_id> <pdf_files_json> <args_dict_json>

ARGUMENTS:
    gpu_id           GPU device ID to use (0, 1, 2, etc.)
    pdf_files_json   JSON string of PDF file paths to process
    args_dict_json   JSON string of processing arguments

ENVIRONMENT:
    Sets CUDA_VISIBLE_DEVICES before importing to ensure GPU isolation.
    Each worker process only sees its assigned GPU.

OUTPUT:
    Results are saved to: /tmp/gpu_{gpu_id}_results.json
    Main process collects these results after worker completes.

NOTE:
    This script is automatically called by main.py for parallel processing.
    Do not run manually unless for debugging purposes.
"""
import os
import sys
import json
from pathlib import Path

def run_worker(gpu_id, pdf_files, args_dict):
    # Set GPU BEFORE any imports
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Now import processor
    from scripts.pdf_processor import PDFProcessor
    import torch
    
    print(f"[GPU {gpu_id}] Worker started, visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[GPU {gpu_id}] CUDA available: {torch.cuda.is_available()}")
    print(f"[GPU {gpu_id}] Number of GPUs visible: {torch.cuda.device_count()}")
    
    processor = PDFProcessor()
    
    results = []
    for pdf_file_str in pdf_files:
        pdf_file = Path(pdf_file_str)
        try:
            print(f"\n[GPU {gpu_id}] Processing: {pdf_file.name}")
            
            output_dir = Path(args_dict['output'])
            pdf_output_dir = output_dir / pdf_file.stem
            
            result = processor.process_pdf(
                input_path=str(pdf_file),
                output_dir=str(pdf_output_dir),
                prompt=args_dict.get('prompt'),
                skip_repeat=args_dict['skip_repeat'],
                crop_mode=args_dict['crop_mode']
            )
            
            results.append({
                "pdf_file": str(pdf_file),
                "status": "success",
                "gpu_id": gpu_id,
                **result
            })
            
            # Clear cache after each PDF
            torch.cuda.empty_cache()
            
        except Exception as e:
            import traceback
            print(f"[GPU {gpu_id}] Error: {e}")
            print(traceback.format_exc())
            results.append({
                "pdf_file": str(pdf_file),
                "status": "failed",
                "gpu_id": gpu_id,
                "error": str(e)
            })
    
    processor.cleanup()
    
    # Save results to temporary file
    result_file = f"/tmp/gpu_{gpu_id}_results.json"
    with open(result_file, 'w') as f:
        json.dump(results, f)
    
    print(f"[GPU {gpu_id}] Worker finished, processed {len(pdf_files)} files")

if __name__ == "__main__":
    gpu_id = int(sys.argv[1])
    pdf_files = json.loads(sys.argv[2])
    args_dict = json.loads(sys.argv[3])
    run_worker(gpu_id, pdf_files, args_dict)