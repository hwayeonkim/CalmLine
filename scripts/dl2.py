from huggingface_hub import snapshot_download
for m in ["MoritzLaurer/deberta-v3-base-zeroshot-v2.0","cross-encoder/nli-deberta-v3-small","s-nlp/bart-base-detox"]:
    print("ok", snapshot_download(m, allow_patterns=["*.json","*.safetensors","*.txt","*.model","spm*","tokenizer*","vocab*","merges*"]), flush=True)
