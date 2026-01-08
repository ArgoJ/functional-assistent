# Functional Assistent

```bash
sudo apt-get update && sudo apt-get install -y libasound2-dev portaudio19-dev python3-pyaudio build-essential cmake git
pip install pyaudio openwakeword numpy transformers llama-cpp-python
```


```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make -j4
bash ./models/download-ggml-model.sh base.en
```


```bash
python3 llama.cpp/convert_hf_to_gguf.py ./checkpoint-160/ --outfile functiongemma-pi.gguf
```

```bash
python src/raspi_assistant.py \
  --whisper_bin_path /home/pi/whisper.cpp/main \
  --whisper_model_path /home/pi/whisper.cpp/models/ggml-base.en.bin \
  --llm_model_path /pfad/zu/deinem/functiongemma-270m-it-custom.gguf
```