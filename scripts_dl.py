from huggingface_hub import snapshot_download
for m in ["unitary/toxic-bert","unitary/unbiased-toxic-roberta","sentence-transformers/all-MiniLM-L6-v2","BAAI/bge-small-en-v1.5","unsloth/gemma-3-1b-it","Systran/faster-whisper-base.en","Systran/faster-whisper-small.en"]:
    p=snapshot_download(m, allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.bin","*.jinja","vocabulary*","*.py","tokenizer*"] if "whisper" not in m else None)
    print("ok",m,p,flush=True)
