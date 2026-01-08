import os
import glob
import subprocess
import sys

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
output_dir = os.path.join(project_root, "functiongemma-tool-calling-sft")
llama_cpp_dir = os.path.join(project_root, "llama.cpp")

def install_llama_cpp():
    if not os.path.exists(llama_cpp_dir):
        print("Cloning llama.cpp...")
        subprocess.check_call(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", llama_cpp_dir])
        
    print("Installing llama.cpp requirements...")
    pip_install_cmd = [sys.executable, "-m", "pip", "install", "-r", os.path.join(llama_cpp_dir, "requirements.txt")]
    subprocess.check_call(pip_install_cmd)

def build_llama_cpp():
    print("Building llama.cpp...")
    # Run make in the llama.cpp directory
    # -j uses all available cores for faster compilation
    subprocess.check_call(["make", "-j"], cwd=llama_cpp_dir)

def get_latest_checkpoint(output_dir):
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda x: int(x.split("-")[-1]))

def quantize_model(input_gguf, method="Q4_K_M"):
    # Try different possible binary names for quantization
    quantize_bins = ["llama-quantize", "quantize"]
    quantize_bin_path = None
    
    for bin_name in quantize_bins:
        path = os.path.join(llama_cpp_dir, bin_name)
        if os.path.exists(path):
            quantize_bin_path = path
            break
            
    if not quantize_bin_path:
        print("Quantize binary not found. Attempting to build...")
        try:
            build_llama_cpp()
            # Retry finding binary
            for bin_name in quantize_bins:
                path = os.path.join(llama_cpp_dir, bin_name)
                if os.path.exists(path):
                    quantize_bin_path = path
                    break
        except Exception as e:
            print(f"Build failed: {e}")
            return

    if not quantize_bin_path:
        print("Error: Could not find 'llama-quantize' or 'quantize' binary.")
        return

    # Use a suffix for the quantized filename
    # e.g., model-f16.gguf -> model-Q4_K_M.gguf
    if "-f16" in input_gguf:
        output_gguf = input_gguf.replace("-f16", f"-{method}")
    else:
        output_gguf = input_gguf.replace(".gguf", f"-{method}.gguf")
    
    print(f"Quantizing {input_gguf} to {method}...")
    cmd = [quantize_bin_path, input_gguf, output_gguf, method]
    subprocess.check_call(cmd)
    print(f"Quantized model saved to: {output_gguf}")
    return output_gguf

def export_model(quantize_methods=None):
    if quantize_methods is None:
        quantize_methods = []

    checkpoint_path = get_latest_checkpoint(output_dir)
    if not checkpoint_path:
        print(f"No checkpoints found in {output_dir}")
        return

    print(f"Found checkpoint: {checkpoint_path}")
    
    # Define output path for GGUF (FP16 base)
    model_name = os.path.basename(checkpoint_path)
    # Adding -f16 suffix to distinguish from quantized versions
    output_gguf = os.path.join(project_root, f"functiongemma-270m-it-{model_name}-f16.gguf")
    
    # Conversion script path
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    
    if not os.path.exists(convert_script):
         # Fallback for older llama.cpp versions or moved scripts
         convert_script = os.path.join(llama_cpp_dir, "convert-hf-to-gguf.py")

    print(f"Converting to GGUF (FP16)...")
    cmd = [
        sys.executable, 
        convert_script, 
        checkpoint_path,
        "--outfile", output_gguf,
        "--outtype", "fp16"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    
    print(f"Successfully exported FP16 model to: {output_gguf}")
    
    # Build llama.cpp if we need to quantize
    if quantize_methods:
        # Check if binary already exists to avoid unnecessary rebuilds (though make handles that usually)
        if not os.path.exists(os.path.join(llama_cpp_dir, "llama-quantize")) and \
           not os.path.exists(os.path.join(llama_cpp_dir, "quantize")):
            build_llama_cpp()
        
    for method in quantize_methods:
        quantize_model(output_gguf, method)

if __name__ == "__main__":
    install_llama_cpp()

    if len(sys.argv) > 1 and sys.argv[1] == "--setup-only":
        build_llama_cpp()
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--all":
        # Example: export all common quantization types
        export_model(quantize_methods=["Q4_K_M", "Q8_0"])
        
    else:
        # Default: Export FP16 and a rational default quantization (Q4_K_M)
        export_model(quantize_methods=["Q4_K_M"])
