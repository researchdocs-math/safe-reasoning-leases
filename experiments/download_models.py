"""Download the pinned quantized checkpoints used for GPU measurements."""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
from huggingface_hub import snapshot_download

def main():
 for m in json.loads((ROOT/'configs/model_manifest.json').read_text()):
  snapshot_download(m['repo'],revision=m['revision'],local_dir=ROOT/m['path'],max_workers=4)
  print(m['repo'],m['revision'])
if __name__=='__main__':main()
