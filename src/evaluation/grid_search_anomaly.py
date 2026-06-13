import argparse
import sys
from pathlib import Path
import itertools
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.inference.reconstruct import AnomalyDetector
from src.evaluation.anomaly_metrics import evaluate_anomaly_detection
from src.utils.helpers import ensure_dir

def main():
    parser = argparse.ArgumentParser(description="Grid search for anomaly detection hyperparameters")
    parser.add_argument("--healthy-dir", type=str, default="data/processed/test/healthy")
    parser.add_argument("--anomalous-dir", type=str, default="data/processed/test/anomalous")
    parser.add_argument("--output", type=str, default="results/evaluation/grid_search")
    parser.add_argument("--vae-config", type=str, default="configs/vae_config.yaml")
    parser.add_argument("--ddpm-config", type=str, default="configs/ddpm_config.yaml")
    parser.add_argument("--vae-checkpoint", type=str, default="checkpoints/vae/best.pt")
    parser.add_argument("--ddpm-checkpoint", type=str, default="checkpoints/ddpm/step_130000.pt")
    parser.add_argument("--t-starts", type=int, nargs="+", default=[150, 250, 300, 400])
    parser.add_argument("--guidance-scales", type=float, nargs="+", default=[3.0, 5.0, 7.5])
    parser.add_argument("--ddim-steps", type=int, default=50)
    args = parser.parse_args()

    output_dir = ensure_dir(project_root / args.output)
    
    print("Loading models...")
    detector = AnomalyDetector(
        vae_config_path=str(project_root / args.vae_config),
        ddpm_config_path=str(project_root / args.ddpm_config),
        vae_checkpoint_path=str(project_root / args.vae_checkpoint),
        ddpm_checkpoint_path=str(project_root / args.ddpm_checkpoint),
    )

    results_list = []
    
    combinations = list(itertools.product(args.t_starts, args.guidance_scales))
    print(f"Starting grid search over {len(combinations)} combinations...")
    
    for t_start, guidance_scale in combinations:
        print(f"\n========================================")
        print(f"Testing t_start={t_start}, guidance_scale={guidance_scale}")
        print(f"========================================")
        
        try:
            results = evaluate_anomaly_detection(
                detector,
                healthy_dir=str(project_root / args.healthy_dir),
                anomalous_dir=str(project_root / args.anomalous_dir),
                t_start=t_start,
                guidance_scale=guidance_scale,
                ddim_steps=args.ddim_steps,
            )
            
            metrics = results["metrics"]
            
            # Record results
            row = {
                "t_start": t_start,
                "guidance_scale": guidance_scale,
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "f1_score": metrics["f1_score"],
                "accuracy": metrics["accuracy"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "precision": metrics["precision"],
                "optimal_threshold": metrics["optimal_threshold"]
            }
            results_list.append(row)
            
        except Exception as e:
            print(f"Failed configuration t_start={t_start}, guidance={guidance_scale}: {e}")
            continue

    if results_list:
        df = pd.DataFrame(results_list)
        # Sort by AUROC descending
        df = df.sort_values(by="auroc", ascending=False)
        
        csv_path = output_dir / "grid_search_results.csv"
        df.to_csv(csv_path, index=False)
        
        print("\n" + "=" * 50)
        print("GRID SEARCH COMPLETED")
        print("=" * 50)
        print(f"Top 3 Configurations (by AUROC):")
        print(df[["t_start", "guidance_scale", "auroc", "f1_score", "sensitivity"]].head(3).to_string(index=False))
        print(f"\nFull results saved to: {csv_path}")
    else:
        print("No configurations succeeded.")

if __name__ == "__main__":
    main()
