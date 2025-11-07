"""
Tensor-Parallel PDF Processor for DeepSeek OCR

Distributes model layers across multiple GPUs to reduce per-GPU memory usage.
Useful for processing large PDFs or when GPU memory is limited.

USAGE:
    python tensor_parallel_processor.py -p <input_path> -o <output_dir> [options]

EXAMPLES:
    # Use 2 GPUs with tensor parallelism
    python tensor_parallel_processor.py -p document.pdf -o output --tensor-parallel-size 2
    
    # Use specific GPUs
    python tensor_parallel_processor.py -p document.pdf -o output --tensor-parallel-size 2 --gpu-ids 0,1
    
    # Process directory sequentially with tensor parallelism
    python tensor_parallel_processor.py -p pdfs/ -o output --tensor-parallel-size 4

KEY DIFFERENCES FROM BATCH PROCESSOR:
    - Model is split across GPUs (lower memory per GPU)
    - Processes PDFs sequentially (not in parallel)
    - Better for large PDFs or limited GPU memory
    - Higher GPU utilization per file

MEMORY COMPARISON:
    Task Parallel (batch_pdf_processor.py):
        - GPU 0: Full model (40GB)
        - GPU 1: Full model (40GB)
        - Total: 80GB, processes 2 PDFs in parallel
    
    Tensor Parallel (this script):
        - GPU 0: Half model (20GB)
        - GPU 1: Half model (20GB)
        - Total: 40GB, processes 1 PDF at a time
"""
import os
import sys
import argparse
import torch
from pathlib import Path
import io
import re
from tqdm import tqdm
from typing import List, Optional

# Import vLLM components
from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry

# Import DeepSeek OCR components
from deepseek_ocr import DeepseekOCRForCausalLM
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor

# Import configuration
try:
    from config import MODEL_PATH, IMAGE_SIZE, BASE_SIZE, CROP_MODE, PROMPT
    CONFIG_AVAILABLE = True
except ImportError:
    # Fallback defaults if config.py is not available
    MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'
    IMAGE_SIZE = 640
    BASE_SIZE = 1024
    CROP_MODE = True
    PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
    CONFIG_AVAILABLE = False
    print("Warning: config.py not found, using default values")

# Import PDF processing utilities
try:
    import fitz  # PyMuPDF
    import img2pdf
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required library. Please install: {e}")
    sys.exit(1)


class TensorParallelPDFProcessor:
    """
    PDF processor using tensor parallelism to distribute model across GPUs.
    """
    
    def __init__(
        self,
        model_path: str = None,
        tensor_parallel_size: int = 2,
        gpu_ids: Optional[List[int]] = None,
        image_size: int = None,
        base_size: int = None,
        crop_mode: bool = None,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.9
    ):
        """
        Initialize tensor-parallel PDF processor.
        
        Args:
            model_path: Path to DeepSeek OCR model (defaults to config.MODEL_PATH)
            tensor_parallel_size: Number of GPUs to split model across
            gpu_ids: Specific GPU IDs to use (e.g., [0, 1, 3])
            image_size: Image size for processing (defaults to config.IMAGE_SIZE)
            base_size: Base size for global view (defaults to config.BASE_SIZE)
            crop_mode: Enable cropping mode (defaults to config.CROP_MODE)
            max_model_len: Maximum sequence length
            gpu_memory_utilization: GPU memory utilization ratio
        """
        # Use config values as defaults if not provided
        self.model_path = model_path if model_path is not None else MODEL_PATH
        self.image_size = image_size if image_size is not None else IMAGE_SIZE
        self.base_size = base_size if base_size is not None else BASE_SIZE
        self.crop_mode = crop_mode if crop_mode is not None else CROP_MODE
        self.tensor_parallel_size = tensor_parallel_size
        
        # Print configuration source
        if CONFIG_AVAILABLE:
            print(f"✓ Using configuration from config.py")
        else:
            print(f"⚠ Using default configuration (config.py not found)")
        
        # Set visible GPUs
        if gpu_ids:
            if len(gpu_ids) != tensor_parallel_size:
                raise ValueError(
                    f"Number of gpu_ids ({len(gpu_ids)}) must match "
                    f"tensor_parallel_size ({tensor_parallel_size})"
                )
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            print(f"Using GPUs: {gpu_ids}")
        else:
            gpu_ids = list(range(tensor_parallel_size))
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            print(f"Using GPUs: {gpu_ids}")
        
        # CUDA configuration for better memory management
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        if torch.version.cuda == '11.8':
            os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
        
        os.environ['VLLM_USE_V1'] = '0'
        
        # Register model
        ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
        
        print(f"\nInitializing model with tensor parallelism...")
        print(f"  - Model path: {self.model_path}")
        print(f"  - Tensor parallel size: {tensor_parallel_size}")
        print(f"  - Image size: {self.image_size}")
        print(f"  - Base size: {self.base_size}")
        print(f"  - Crop mode: {self.crop_mode}")
        print(f"  - Max model length: {max_model_len}")
        print(f"  - GPU memory utilization: {gpu_memory_utilization}")
        
        # Initialize LLM with tensor parallelism
        self.llm = LLM(
            model=self.model_path,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=256,
            enforce_eager=True,
            trust_remote_code=True,
            max_model_len=max_model_len,
            swap_space=0,
            max_num_seqs=1,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            disable_mm_preprocessor_cache=True
        )
        
        # Initialize processor
        self.processor = DeepseekOCRProcessor()
        
        # Setup logits processors
        self.logits_processors = [
            NoRepeatNGramLogitsProcessor(
                ngram_size=20,
                window_size=50,
                whitelist_token_ids={128821, 128822}  # <td>, </td>
            )
        ]
        
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_model_len,
            logits_processors=self.logits_processors,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        )
        
        print("Model initialized successfully!\n")
    
    def pdf_to_images(self, pdf_path: str, dpi: int = 144) -> List[Image.Image]:
        """Convert PDF to high-quality images."""
        images = []
        pdf_document = fitz.open(pdf_path)
        
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        
        print(f"Converting PDF to images (DPI: {dpi})...")
        for page_num in tqdm(range(pdf_document.page_count), desc="PDF pages"):
            page = pdf_document[page_num]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            
            Image.MAX_IMAGE_PIXELS = None
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            
            images.append(img)
        
        pdf_document.close()
        return images
    
    def process_single_image(self, image: Image.Image, prompt: str):
        """Process a single image through the model."""
        cache_item = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": self.processor.tokenize_with_images(
                    images=[image],
                    bos=True,
                    eos=True,
                    cropping=self.crop_mode
                )
            },
        }
        return cache_item
    
    def extract_coordinates_and_label(self, ref_text, image_width, image_height):
        """Extract bounding box coordinates from detection tags."""
        try:
            label_type = ref_text[1]
            cor_list = eval(ref_text[2])
        except Exception as e:
            print(f"Error extracting coordinates: {e}")
            return None
        return (label_type, cor_list)
    
    def draw_bounding_boxes(self, image: Image.Image, refs, output_dir: Path, page_idx: int):
        """Draw bounding boxes and extract image regions."""
        image_width, image_height = image.size
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        
        overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
        draw2 = ImageDraw.Draw(overlay)
        font = ImageFont.load_default()
        
        img_idx = 0
        
        for i, ref in enumerate(refs):
            try:
                result = self.extract_coordinates_and_label(ref, image_width, image_height)
                if result:
                    label_type, points_list = result
                    color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
                    color_a = color + (20,)
                    
                    for points in points_list:
                        x1, y1, x2, y2 = points
                        x1 = int(x1 / 999 * image_width)
                        y1 = int(y1 / 999 * image_height)
                        x2 = int(x2 / 999 * image_width)
                        y2 = int(y2 / 999 * image_height)
                        
                        # Extract image regions
                        if label_type == 'image':
                            try:
                                cropped = image.crop((x1, y1, x2, y2))
                                image_path = output_dir / f"{page_idx}_{img_idx}.jpg"
                                cropped.save(image_path)
                                print(f"  ✓ Saved image: {image_path}")
                            except Exception as e:
                                print(f"  ✗ Error saving cropped image: {e}")
                            img_idx += 1
                        
                        # Draw bounding boxes
                        try:
                            if label_type == 'title':
                                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                                draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                            else:
                                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                                draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                            
                            text_x = x1
                            text_y = max(0, y1 - 15)
                            text_bbox = draw.textbbox((0, 0), label_type, font=font)
                            text_width = text_bbox[2] - text_bbox[0]
                            text_height = text_bbox[3] - text_bbox[1]
                            draw.rectangle(
                                [text_x, text_y, text_x + text_width, text_y + text_height],
                                fill=(255, 255, 255, 30)
                            )
                            draw.text((text_x, text_y), label_type, font=font, fill=color)
                        except:
                            pass
            except:
                continue
        
        img_draw.paste(overlay, (0, 0), overlay)
        return img_draw
    
    def re_match(self, text):
        """Extract detection tags from output."""
        pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        matches_image = []
        matches_other = []
        for a_match in matches:
            if '<|ref|>image<|/ref|>' in a_match[0]:
                matches_image.append(a_match[0])
            else:
                matches_other.append(a_match[0])
        return matches, matches_image, matches_other
    
    def pil_to_pdf(self, pil_images: List[Image.Image], output_path: Path):
        """Convert PIL images to PDF."""
        if not pil_images:
            return
        
        image_bytes_list = []
        for img in pil_images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='JPEG', quality=95)
            img_bytes = img_buffer.getvalue()
            image_bytes_list.append(img_bytes)
        
        try:
            pdf_bytes = img2pdf.convert(image_bytes_list)
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)
        except Exception as e:
            print(f"Error creating PDF: {e}")
    
    def process_pdf(
        self,
        input_path: str,
        output_dir: str,
        prompt: str = '<image>\n<|grounding|>Convert the document to markdown.',
        skip_repeat: bool = True,
        dpi: int = 144
    ):
        """
        Process a PDF file to markdown with tensor parallelism.
        
        Args:
            input_path: Path to input PDF
            output_dir: Directory to save outputs
            prompt: OCR prompt
            skip_repeat: Skip pages without EOS token
            dpi: Image conversion DPI
        
        Returns:
            dict: Processing results
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify output directory is writable
        if not output_dir.exists():
            raise RuntimeError(f"Failed to create output directory: {output_dir}")
        if not os.access(output_dir, os.W_OK):
            raise RuntimeError(f"Output directory is not writable: {output_dir}")
        
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Images directory created: {images_dir}")
        
        print(f"\n{'='*60}")
        print(f"Processing: {input_path.name}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}\n")
        
        # Convert PDF to images
        images = self.pdf_to_images(str(input_path), dpi=dpi)
        print(f"Converted {len(images)} pages to images\n")
        
        # Preprocess images
        print("Preprocessing images...")
        batch_inputs = []
        for image in tqdm(images, desc="Preprocessing"):
            cache_item = self.process_single_image(image, prompt)
            batch_inputs.append(cache_item)
        
        # Generate outputs (tensor parallelism happens here)
        print("\nGenerating OCR outputs (using tensor parallelism)...")
        print(f"Model is distributed across {self.tensor_parallel_size} GPU(s)")
        outputs_list = self.llm.generate(batch_inputs, self.sampling_params)
        
        # Process results
        print("\nProcessing results...")
        contents_det = ''
        contents = ''
        draw_images = []
        page_separator = '\n<--- Page Split --->'
        
        valid_pages = 0
        skipped_pages = 0
        
        for page_idx, (output, img) in enumerate(tqdm(
            zip(outputs_list, images),
            total=len(images),
            desc="Processing pages"
        )):
            content = output.outputs[0].text
            
            # Check for EOS token
            if '<｜end▁of▁sentence｜>' in content:
                content = content.replace('<｜end▁of▁sentence｜>', '')
            else:
                if skip_repeat:
                    skipped_pages += 1
                    continue
            
            valid_pages += 1
            
            # Add to detection output
            contents_det += content + f'\n{page_separator}\n'
            
            # Extract bounding boxes
            matches_ref, matches_images, matches_other = self.re_match(content)
            
            # Draw visualizations
            image_draw = img.copy()
            result_image = self.draw_bounding_boxes(
                image_draw, matches_ref, images_dir, page_idx
            )
            draw_images.append(result_image)
            
            # Replace image tags with links
            for idx, a_match_image in enumerate(matches_images):
                content = content.replace(
                    a_match_image,
                    f'![](images/{page_idx}_{idx}.jpg)\n'
                )
            
            # Remove other detection tags
            for a_match_other in matches_other:
                content = content.replace(a_match_other, '').replace(
                    '\\coloneqq', ':='
                ).replace(
                    '\\eqqcolon', '=:'
                ).replace(
                    '\n\n\n\n', '\n\n'
                ).replace(
                    '\n\n\n', '\n\n'
                )
            
            contents += content + f'\n{page_separator}\n'
        
        # Save outputs
        print("\nSaving results...")
        
        # Warn if no valid content
        if not contents.strip() or valid_pages == 0:
            print(f"⚠ Warning: No valid pages processed (valid_pages={valid_pages})")
            print(f"⚠ This usually means all pages were skipped or had errors")
        
        # Save markdown files
        mmd_det_path = output_dir / f'{input_path.stem}_det.mmd'
        mmd_path = output_dir / f'{input_path.stem}.mmd'
        pdf_out_path = output_dir / f'{input_path.stem}_layouts.pdf'
        
        with open(mmd_det_path, 'w', encoding='utf-8') as f:
            f.write(contents_det)
        print(f"✓ Saved detection file: {mmd_det_path}")
        print(f"  File size: {mmd_det_path.stat().st_size} bytes")
        
        with open(mmd_path, 'w', encoding='utf-8') as f:
            f.write(contents)
        print(f"✓ Saved markdown file: {mmd_path}")
        print(f"  File size: {mmd_path.stat().st_size} bytes")
        
        # Save visualization PDF
        if draw_images:
            self.pil_to_pdf(draw_images, pdf_out_path)
            if pdf_out_path.exists():
                print(f"✓ Saved layout PDF: {pdf_out_path}")
                print(f"  File size: {pdf_out_path.stat().st_size} bytes")
            else:
                print(f"✗ Warning: PDF file not created: {pdf_out_path}")
        else:
            print(f"✗ Warning: No pages to save in PDF (all pages skipped)")
        
        result = {
            'total_pages': len(images),
            'valid_pages': valid_pages,
            'skipped_pages': skipped_pages,
            'markdown_file': str(mmd_path),
            'detection_file': str(mmd_det_path),
            'layout_pdf': str(pdf_out_path),
            'images_dir': str(images_dir)
        }
        
        print(f"\n{'='*60}")
        print("Processing Complete!")
        print(f"{'='*60}")
        print(f"Total pages: {result['total_pages']}")
        print(f"Valid pages: {result['valid_pages']}")
        print(f"Skipped pages: {result['skipped_pages']}")
        print(f"Markdown: {mmd_path}")
        print(f"Layout PDF: {pdf_out_path}")
        print(f"Images: {images_dir}")
        print(f"{'='*60}\n")
        
        return result
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'llm'):
            del self.llm
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Process PDFs using tensor parallelism across multiple GPUs.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "-p", "--path",
        type=str,
        required=True,
        help="Input PDF file or directory"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output directory"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=f"Path to DeepSeek OCR model (default: from config.py or '{MODEL_PATH}')"
    )
    
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Number of GPUs to split model across (default: 2)"
    )
    
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs (e.g., '0,1,3')"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=f"Custom OCR prompt (default: from config.py)"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=144,
        help="DPI for PDF to image conversion (default: 144)"
    )
    
    parser.add_argument(
        "--skip-repeat",
        action="store_true",
        default=True,
        help="Skip pages without EOS token (default: True)"
    )
    
    parser.add_argument(
        "--no-skip-repeat",
        dest="skip_repeat",
        action="store_false",
        help="Process all pages"
    )
    
    parser.add_argument(
        "--crop-mode",
        action="store_true",
        default=None,
        help=f"Enable crop mode (default: from config.py or {CROP_MODE})"
    )
    
    parser.add_argument(
        "--no-crop-mode",
        dest="crop_mode",
        action="store_false",
        help="Disable crop mode"
    )
    
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help=f"Image size for processing (default: from config.py or {IMAGE_SIZE})"
    )
    
    parser.add_argument(
        "--base-size",
        type=int,
        default=None,
        help=f"Base size for global view (default: from config.py or {BASE_SIZE})"
    )
    
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum sequence length (default: 8192)"
    )
    
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio (default: 0.9)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.path)
    if not input_path.exists():
        print(f"Error: Input path '{args.path}' does not exist.")
        return
    
    # Parse GPU IDs
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        if len(gpu_ids) != args.tensor_parallel_size:
            print(f"Error: Number of GPU IDs ({len(gpu_ids)}) must match tensor-parallel-size ({args.tensor_parallel_size})")
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
    
    print(f"\n{'='*60}")
    print("TENSOR PARALLEL PDF PROCESSOR")
    print(f"{'='*60}")
    print(f"Found {len(pdf_files)} PDF file(s) to process")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    if gpu_ids:
        print(f"Using GPUs: {gpu_ids}")
    print(f"{'='*60}\n")
    
    # Initialize processor
    processor = TensorParallelPDFProcessor(
        model_path=args.model_path if args.model_path else None,  # Will use config.MODEL_PATH
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_ids=gpu_ids,
        image_size=args.image_size,  # Will use config.IMAGE_SIZE if None
        base_size=args.base_size,    # Will use config.BASE_SIZE if None
        crop_mode=args.crop_mode,    # Will use config.CROP_MODE if None
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    # Use prompt from args or config
    prompt_to_use = args.prompt if args.prompt else PROMPT
    print(f"Using prompt: {prompt_to_use[:50]}..." if len(prompt_to_use) > 50 else f"Using prompt: {prompt_to_use}")
    print()
    
    # Process PDFs sequentially
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    for pdf_file in pdf_files:
        pdf_output_dir = output_dir / pdf_file.stem
        
        try:
            result = processor.process_pdf(
                input_path=str(pdf_file),
                output_dir=str(pdf_output_dir),
                prompt=prompt_to_use,  # Use the determined prompt
                skip_repeat=args.skip_repeat,
                dpi=args.dpi
            )
            
            result['pdf_file'] = str(pdf_file)
            result['status'] = 'success'
            all_results.append(result)
            
        except Exception as e:
            import traceback
            print(f"\nError processing {pdf_file.name}: {e}")
            print(traceback.format_exc())
            all_results.append({
                'pdf_file': str(pdf_file),
                'status': 'failed',
                'error': str(e)
            })
        
        # Clear cache between files
        torch.cuda.empty_cache()
    
    # Cleanup
    processor.cleanup()
    
    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    successful = sum(1 for r in all_results if r['status'] == 'success')
    failed = sum(1 for r in all_results if r['status'] == 'failed')
    print(f"Total files: {len(all_results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        total_pages = sum(r.get('total_pages', 0) for r in all_results if r['status'] == 'success')
        valid_pages = sum(r.get('valid_pages', 0) for r in all_results if r['status'] == 'success')
        print(f"Total pages processed: {total_pages}")
        print(f"Valid pages: {valid_pages}")
        print(f"\nOutput directory: {output_dir}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()