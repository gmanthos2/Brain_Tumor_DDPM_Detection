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
    
    # Accept multiple checkpoints
    parser.add_argument("--ddpm-checkpoints", type=str, nargs="+", default=[
        "checkpoints/ddpm/step_50000.pt",
        "checkpoints/ddpm/step_60000.pt",
        "checkpoints/ddpm/step_70000.pt",
        "checkpoints/ddpm/step_80000.pt",
        "checkpoints/ddpm/step_90000.pt",
    ])
    
    parser.add_argument("--t-starts", type=int, nargs="+", default=[50, 100, 150, 200])
    parser.add_argument("--guidance-scales", type=float, nargs="+", default=[3.0, 5.0, 7.5])
    parser.add_argument("--ddim-steps", type=int, default=50)
    args = parser.parse_args()

    output_dir = ensure_dir(project_root / args.output)
    csv_path = output_dir / "grid_search_results.csv"
    
    results_list = []
    evaluated_combinations = set()

    if csv_path.exists():
        print(f"Loading existing results from {csv_path}")
        df_existing = pd.read_csv(csv_path)
        results_list = df_existing.to_dict('records')
        for row in results_list:
            evaluated_combinations.add((row['checkpoint'], row['t_start'], row['guidance_scale']))
    
    combinations = list(itertools.product(args.ddpm_checkpoints, args.t_starts, args.guidance_scales))
    
    # Filter combinations that haven't been evaluated
    combinations_to_run = [
        c for c in combinations 
        if (Path(c[0]).name, c[1], c[2]) not in evaluated_combinations
    ]
    
    print(f"Total combinations requested: {len(combinations)}")
    print(f"Combinations already evaluated: {len(combinations) - len(combinations_to_run)}")
    print(f"Starting grid search over {len(combinations_to_run)} NEW combinations...")
    
    # Keep track of the currently loaded checkpoint to avoid reloading if it hasn't changed
    current_ckpt = None
    detector = None

    for ckpt_path, t_start, guidance_scale in combinations_to_run:
        print(f"\n========================================")
        print(f"Testing Checkpoint: {Path(ckpt_path).name}")
        print(f"Testing t_start: {t_start}, guidance_scale: {guidance_scale}")
        print(f"========================================")
        
        # Reload model only if checkpoint changes
        if current_ckpt != ckpt_path:
            print(f"Loading DDPM checkpoint: {ckpt_path} ...")
            try:
                detector = AnomalyDetector(
                    vae_config_path=str(project_root / args.vae_config),
                    ddpm_config_path=str(project_root / args.ddpm_config),
                    vae_checkpoint_path=str(project_root / args.vae_checkpoint),
                    ddpm_checkpoint_path=str(project_root / ckpt_path),
                )
                current_ckpt = ckpt_path
            except Exception as e:
                print(f"Failed to load checkpoint {ckpt_path}: {e}")
                continue
        
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
                "checkpoint": Path(ckpt_path).name,
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
            
            # Save incrementally so we don't lose progress if interrupted
            df = pd.DataFrame(results_list)
            df = df.sort_values(by="auroc", ascending=False)
            csv_path = output_dir / "grid_search_results_incremental.csv"
            df.to_csv(csv_path, index=False)
            
        except Exception as e:
            print(f"Failed configuration {Path(ckpt_path).name}, t_start={t_start}, guidance={guidance_scale}: {e}")
            continue

    if results_list:
        df = pd.DataFrame(results_list)
        # Sort by AUROC descending
        df = df.sort_values(by="auroc", ascending=False)
        
        csv_path = output_dir / "grid_search_results.csv"
        df.to_csv(csv_path, index=False)
        
        # Clean up incremental file if finished completely
        if (output_dir / "grid_search_results_incremental.csv").exists():
            (output_dir / "grid_search_results_incremental.csv").unlink()
        
        print("\n" + "=" * 50)
        print("GRID SEARCH COMPLETED")
        print("=" * 50)
        print(f"Top 5 Configurations (by AUROC):")
        print(df[["checkpoint", "t_start", "guidance_scale", "auroc", "f1_score", "sensitivity"]].head(5).to_string(index=False))
        print(f"\nFull results saved to: {csv_path}")
    else:
        print("No configurations succeeded.")

if __name__ == "__main__":
    main()
