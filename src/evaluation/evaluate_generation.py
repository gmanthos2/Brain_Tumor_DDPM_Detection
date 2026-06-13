import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# We use torchmetrics for FID score computation
from torchmetrics.image.fid import FrechetInceptionDistance

from src.models.diffusion import GaussianDiffusion
from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = setup_logger("evaluate_generation")

class ImageFolderDataset(torch.utils.data.Dataset):
    """Simple dataset to load images from a directory."""
    def __init__(self, folder_path, transform=None):
        self.paths = list(Path(folder_path).glob("*.jpg"))
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

def main():
    parser = argparse.ArgumentParser(description="Evaluate Generation Quality using FID")
    parser.add_argument("--real-dir", type=str, default="dataset/no")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of synthetic images to generate for FID")
    parser.add_argument("--vae-config", type=str, default="configs/vae_config.yaml")
    parser.add_argument("--ddpm-config", type=str, default="configs/ddpm_config.yaml")
    parser.add_argument("--vae-checkpoint", type=str, default="checkpoints/vae/best.pt")
    parser.add_argument("--ddpm-checkpoint", type=str, default="checkpoints/ddpm/step_130000.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # FID expects uint8 RGB images in range [0, 255]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).byte())
    ])
    
    # 1. Load Real Images
    logger.info(f"Loading real images from {args.real_dir}...")
    real_dataset = ImageFolderDataset(project_root / args.real_dir, transform=transform)
    # Only use num_samples to match the generated count
    real_subset = torch.utils.data.Subset(real_dataset, range(min(args.num_samples, len(real_dataset))))
    real_loader = DataLoader(real_subset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize FID metric (feature=64 is faster, 2048 is standard paper quality)
    logger.info("Initializing FID metric (Inception V3)...")
    fid = FrechetInceptionDistance(feature=2048).to(device)
    
    # Process Real Images
    for batch in tqdm(real_loader, desc="Processing Real Images for FID"):
        fid.update(batch.to(device), real=True)
        
    # 2. Load Models to Generate Synthetic Images
    logger.info("Loading Generative Models...")
    from src.inference.generate import SyntheticGenerator
    
    generator = SyntheticGenerator(
        vae_config_path=str(project_root / args.vae_config),
        ddpm_config_path=str(project_root / args.ddpm_config),
        vae_checkpoint_path=str(project_root / args.vae_checkpoint),
        ddpm_checkpoint_path=str(project_root / args.ddpm_checkpoint)
    )
    
    # 3. Generate and Process Synthetic Images
    logger.info(f"Generating {args.num_samples} synthetic images...")
    
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    generated_count = 0
    
    for _ in tqdm(range(num_batches), desc="Generating Fake Images for FID"):
        current_batch_size = min(args.batch_size, args.num_samples - generated_count)
        
        # Generate images in range [-1, 1]
        with torch.no_grad():
            images_tensor = generator.generate(num_samples=current_batch_size)
            
        # Convert [-1, 1] -> [0, 1] -> [0, 255] uint8 -> RGB (3 channels)
        images_tensor = (images_tensor + 1) / 2.0
        images_tensor = (images_tensor * 255).clamp(0, 255).to(torch.uint8)
        
        if images_tensor.shape[1] == 1:
            images_tensor = images_tensor.repeat(1, 3, 1, 1) # Repeat grayscale to RGB
            
        fid.update(images_tensor.to(device), real=False)
        generated_count += current_batch_size
        
    # 4. Compute Final Score
    logger.info("Computing final FID score (this may take a moment)...")
    fid_score = fid.compute()
    
    logger.info(f"=========================================")
    logger.info(f"Final FID Score (Real vs Synthetic): {fid_score.item():.4f}")
    logger.info(f"=========================================")

if __name__ == "__main__":
    main()
