"""
An OOTB multi-GPU supported demo for distributing multiple pdf to A100 SXM4 80G.

Process PDF files to Markdown using DeepSeek OCR across multiple GPUs.

USAGE:
    python task_parallel.py -p <input_path> -o <output_dir> [options]

EXAMPLES:
    # Process directory with 2 GPUs
    python task_parallel.py -p pdfs/ -o output --num-gpus 2
    
    # Use specific GPUs
    python task_parallel.py -p pdfs/ -o output --gpu-ids 0,3
    
    # Process single PDF
    python task_parallel.py -p document.pdf -o output

REQUIRED ARGUMENTS:
    -p, --path          Input PDF file or directory
    -o, --output        Output directory

OPTIONAL ARGUMENTS:
    --num-gpus          Number of GPUs to use (default: 2)
    --gpu-ids           Specific GPU IDs, e.g., '0,1,3'
    --prompt            Custom OCR prompt
    --dpi               Image conversion DPI (default: 144)
    --no-skip-repeat    Process all pages
    --no-crop-mode      Disable crop mode

OUTPUT:
    output_dir/
    ├── pdf_name/
    │   ├── pdf_name.mmd              # Markdown
    │   ├── pdf_name_layouts.pdf      # PDF with annotations
    │   └── images/                   # Extracted images
    └── processing_log.txt            # Processing summary

REQUIREMENTS:
    - pdf_processor.py (PDFProcessor class)
    - pdf_worker.py (worker script)
    - config.py (ModelConfig)
"""
import argparse
import subprocess
from pathlib import Path
import json

def main():
    parser = argparse.ArgumentParser(description="Extract pdf files to markdown files.")
    parser.add_argument("-p", "--path", type=str, required=True,
                        help="Input directory of pdf files.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output directory for saving markdown files.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt for OCR processing (optional).")
    parser.add_argument("--skip-repeat", action="store_true", default=True,
                        help="Skip pages without EOS token (default: True).")
    parser.add_argument("--no-skip-repeat", dest="skip_repeat", action="store_false",
                        help="Process all pages regardless of EOS token.")
    parser.add_argument("--crop-mode", action="store_true", default=True,
                        help="Enable crop mode (default: True).")
    parser.add_argument("--no-crop-mode", dest="crop_mode", action="store_false",
                        help="Disable crop mode.")
    parser.add_argument("--dpi", type=int, default=144,
                        help="DPI for PDF to image conversion (default: 144).")
    parser.add_argument("--num-gpus", type=int, default=2,
                        help="Number of GPUs to use (default: 2).")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="Comma-separated GPU IDs to use (e.g., '0,1').")
    
    args = parser.parse_args()
    
    # Validate input path
    input_path = Path(args.path)
    if not input_path.exists():
        print(f"Error: Input path '{args.path}' does not exist.")
        return
    
    # Collect PDF files
    if input_path.is_file():
        if input_path.suffix.lower() == '.pdf':
            pdf_files = [input_path]
        else:
            print(f"Error: '{args.path}' is not a PDF file.")
            return
    elif input_path.is_dir():
        pdf_files = list(input_path.glob("*.pdf")) + list(input_path.glob("*.PDF"))
        if not pdf_files:
            print(f"Error: No PDF files found in '{args.path}'.")
            return
    else:
        print(f"Error: '{args.path}' is neither a file nor a directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to process.")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    else:
        gpu_ids = list(range(args.num_gpus))
    
    print(f"Using {len(gpu_ids)} GPU(s): {gpu_ids}")
    
    # Split PDFs among GPUs
    pdf_files_str = [str(f) for f in pdf_files]
    pdf_chunks = [pdf_files_str[i::len(gpu_ids)] for i in range(len(gpu_ids))]
    
    for i, (gpu_id, chunk) in enumerate(zip(gpu_ids, pdf_chunks)):
        print(f"GPU {gpu_id} will process {len(chunk)} files")
    
    # Prepare arguments
    args_dict = {
        'output': args.output,
        'prompt': args.prompt,
        'skip_repeat': args.skip_repeat,
        'crop_mode': args.crop_mode,
        'dpi': args.dpi
    }
    
    # Launch workers as separate processes
    processes = []
    for gpu_id, pdf_chunk in zip(gpu_ids, pdf_chunks):
        if not pdf_chunk:  # Skip if no files for this GPU
            continue
        
        cmd = [
            'python',
            'pdf_worker.py',
            str(gpu_id),
            json.dumps(pdf_chunk),
            json.dumps(args_dict)
        ]
        
        print(f"Starting worker for GPU {gpu_id}...")
        proc = subprocess.Popen(cmd)
        processes.append((gpu_id, proc))
    
    # Wait for all processes to complete
    print("\nWaiting for all workers to complete...")
    for gpu_id, proc in processes:
        proc.wait()
        print(f"Worker for GPU {gpu_id} completed with exit code {proc.returncode}")
    
    # Collect results
    all_results = []
    for gpu_id in gpu_ids:
        result_file = f"/tmp/gpu_{gpu_id}_results.json"
        if Path(result_file).exists():
            with open(result_file, 'r') as f:
                results = json.load(f)
                all_results.extend(results)
            # Clean up temp file
            Path(result_file).unlink()
    
    # Print summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    successful = sum(1 for r in all_results if r["status"] == "success")
    failed = sum(1 for r in all_results if r["status"] == "failed")
    print(f"Total files: {len(all_results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    # GPU distribution
    for gpu_id in gpu_ids:
        gpu_count = sum(1 for r in all_results if r.get("gpu_id") == gpu_id)
        gpu_success = sum(1 for r in all_results if r.get("gpu_id") == gpu_id and r["status"] == "success")
        print(f"GPU {gpu_id}: {gpu_count} files ({gpu_success} successful)")
    
    if successful > 0:
        print(f"\nOutput directory: {output_dir}")
    
    # Save processing log
    log_file = output_dir / "processing_log.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("PDF Processing Log\n")
        f.write("="*60 + "\n\n")
        for result in sorted(all_results, key=lambda x: x['pdf_file']):
            f.write(f"File: {result['pdf_file']}\n")
            f.write(f"GPU: {result.get('gpu_id', 'N/A')}\n")
            f.write(f"Status: {result['status']}\n")
            if result['status'] == 'success':
                f.write(f"Markdown: {result.get('markdown_file', 'N/A')}\n")
            else:
                f.write(f"Error: {result.get('error', 'Unknown error')}\n")
            f.write("\n")
    
    print(f"\nProcessing log saved to: {log_file}")

if __name__ == "__main__":
    main()