"""
scripts/download_raddit_checkpoint.py

Helper script to download pretrained Rad-DiT model checkpoints from Hugging Face.
Usage:
    python scripts/download_raddit_checkpoint.py --model L16-mix15 --token <YOUR_HF_TOKEN>
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path
from tqdm import tqdm

CHECKPOINT_URLS = {
    "L16-mix15": (
        "https://huggingface.co/rad-dit/jit-radimagenet/resolve/main/L16-mix15/checkpoint-last.pth",
        "jit-l16-mix15.pth",
    ),
    "L16-mix25": (
        "https://huggingface.co/rad-dit/jit-radimagenet/resolve/main/L16-mix25/checkpoint-last.pth",
        "jit-l16-mix25.pth",
    ),
    "L16": (
        "https://huggingface.co/rad-dit/jit-radimagenet/resolve/main/L16/checkpoint-last.pth",
        "jit-l16.pth",
    ),
    "B16": (
        "https://huggingface.co/rad-dit/jit-radimagenet/resolve/main/B16/checkpoint-last.pth",
        "jit-b16.pth",
    ),
    "ldit-brats": (
        "https://huggingface.co/rad-dit/ldit-brats/resolve/main/ldit-xl1-vavae.pt",
        "ldit-xl1-brats.pt",
    ),
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_checkpoint(model_name: str, token: str = None, target_dir: str = None):
    if model_name not in CHECKPOINT_URLS:
        raise ValueError(
            f"Unknown model variant: {model_name}. Available: {list(CHECKPOINT_URLS.keys())}"
        )

    url, filename = CHECKPOINT_URLS[model_name]

    if target_dir is None:
        target_dir = Path(__file__).resolve().parents[1] / "baselines" / "rad-dit" / "checkpoints"
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    dest_path = target_dir / filename

    print(f"\nDownloading Rad-DiT model '{model_name}'...")
    print(f"URL: {url}")
    print(f"Destination: {dest_path}")

    # Use token from argument, env variable, or ~/.cache/huggingface/token
    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        token_file = Path.home() / ".cache" / "huggingface" / "token"
        if token_file.exists():
            hf_token = token_file.read_text().strip()

    req = urllib.request.Request(url)
    if hf_token:
        req.add_header("Authorization", f"Bearer {hf_token}")
        print("Using Hugging Face authorization token.")
    else:
        print("\n[!] Note: rad-dit repositories are gated on Hugging Face (CC-BY-NC-4.0).")
        print("    If you encounter HTTP 401 Unauthorized:")
        print("    1. Go to https://huggingface.co/rad-dit/jit-radimagenet and accept the terms.")
        print("    2. Re-run with --token <YOUR_HF_TOKEN> or export HF_TOKEN=<token>.\n")

    try:
        with urllib.request.urlopen(req) as response:
            total_size = int(response.info().get("Content-Length", 0))
            with open(dest_path, "wb") as f_out, tqdm(
                total=total_size, unit="B", unit_scale=True, desc=filename
            ) as pbar:
                while True:
                    chunk = response.read(1024 * 1024)  # 1 MB chunks
                    if not chunk:
                        break
                    f_out.write(chunk)
                    pbar.update(len(chunk))
        print(f"\n[✓] Successfully downloaded {filename} to {dest_path}")
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print(f"\n[X] HTTP Error 401: Unauthorized.")
            print("    Please accept the repository terms at:")
            print("    https://huggingface.co/rad-dit/jit-radimagenet")
            print("    and pass your user access token using --token <YOUR_TOKEN>.")
        else:
            print(f"\n[X] Download failed with HTTP Error {e.code}: {e.reason}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download Rad-DiT pretrained checkpoints")
    parser.add_argument(
        "--model",
        default="L16-mix15",
        choices=list(CHECKPOINT_URLS.keys()),
        help="Model variant to download (default: L16-mix15)",
    )
    parser.add_argument("--token", default=None, help="Hugging Face personal access token")
    parser.add_argument("--output-dir", default=None, help="Directory to save checkpoint")
    args = parser.parse_args()

    download_checkpoint(model_name=args.model, token=args.token, target_dir=args.output_dir)


if __name__ == "__main__":
    main()
