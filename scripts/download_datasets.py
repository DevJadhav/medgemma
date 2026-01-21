#!/usr/bin/env python3
"""Download medical datasets for MedAI Compass.

CLI tool to download datasets from Section 12.3 of the implementation plan.

Usage:
    # List available datasets
    python scripts/download_datasets.py --list
    
    # Download specific dataset
    python scripts/download_datasets.py --dataset medquad --output ./data
    
    # Download recommended starter datasets
    python scripts/download_datasets.py --recommended --output ./data
    
    # Download with PhysioNet credentials (for MIMIC)
    python scripts/download_datasets.py --dataset mimic_iv --output ./data \\
        --physionet-user YOUR_USER --physionet-pass YOUR_PASS
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from medai_compass.utils.dataset_downloader import (
    DatasetDownloader,
    DATASETS,
    DatasetAccess,
)


def list_datasets():
    """List all available datasets."""
    print("\n" + "=" * 70)
    print("  Available Medical Datasets (from Implementation Plan Section 12.3)")
    print("=" * 70 + "\n")
    
    # Group by access type
    open_datasets = []
    credentialed_datasets = []
    
    for key, info in DATASETS.items():
        entry = {
            "id": key,
            "name": info.name,
            "size": info.size_gb,
            "url": info.url,
            "description": info.description,
        }
        if info.access == DatasetAccess.OPEN:
            open_datasets.append(entry)
        else:
            credentialed_datasets.append(entry)
    
    print("📂 OPEN ACCESS (no credentials required):")
    print("-" * 70)
    for ds in open_datasets:
        size_str = f"{ds['size']:.1f}GB" if ds["size"] >= 1 else f"{ds['size']*1024:.0f}MB"
        print(f"  {ds['id']:15} | {ds['name']:15} | {size_str:>8}")
        print(f"                  | {ds['description'][:50]}...")
    
    print("\n🔐 CREDENTIALED ACCESS (requires approval):")
    print("-" * 70)
    for ds in credentialed_datasets:
        size_str = f"{ds['size']:.1f}GB" if ds["size"] >= 1 else f"{ds['size']*1024:.0f}MB"
        print(f"  {ds['id']:15} | {ds['name']:15} | {size_str:>8}")
        print(f"                  | {ds['description'][:50]}...")
    
    print("\n💡 RECOMMENDED START:")
    print("-" * 70)
    print("  1. medquad     - Small QA dataset, instant download")
    print("  2. meddialog   - Medical conversations, instant download")
    print("  3. chestxray14 - Chest X-rays, manual download (~42GB)")
    
    print("\n📋 USAGE:")
    print("-" * 70)
    print("  # Download a specific dataset:")
    print("  python scripts/download_datasets.py --dataset medquad --output ./data")
    print()
    print("  # Download all recommended datasets:")
    print("  python scripts/download_datasets.py --recommended --output ./data")
    print()


def download_dataset(dataset_id: str, output_dir: Path, force: bool = False,
                     physionet_user: str = None, physionet_pass: str = None):
    """Download a specific dataset."""
    if dataset_id not in DATASETS:
        print(f"❌ Unknown dataset: {dataset_id}")
        print(f"   Available: {', '.join(DATASETS.keys())}")
        return False
    
    info = DATASETS[dataset_id]
    
    print(f"\n📥 Downloading: {info.name}")
    print(f"   Source: {info.url}")
    print(f"   Size: {info.size_gb:.1f}GB")
    print(f"   Output: {output_dir / dataset_id}")
    print()
    
    if info.requires_credentials and not (physionet_user and physionet_pass):
        print("⚠️  This dataset requires PhysioNet credentials.")
        print("   Set --physionet-user and --physionet-pass, or set environment variables:")
        print("   PHYSIONET_USERNAME and PHYSIONET_PASSWORD")
        print()
        print("   To get credentials:")
        print("   1. Register at https://physionet.org/")
        print("   2. Complete CITI training")
        print("   3. Apply for credentialed access")
        print()
    
    downloader = DatasetDownloader(
        output_dir=output_dir,
        physionet_username=physionet_user or os.environ.get("PHYSIONET_USERNAME"),
        physionet_password=physionet_pass or os.environ.get("PHYSIONET_PASSWORD"),
    )
    
    def progress(msg, pct):
        print(f"   {pct:5.1f}% | {msg}")
    
    downloader.set_progress_callback(progress)
    
    try:
        path = downloader.download(dataset_id, force=force)
        print(f"\n✅ Downloaded to: {path}")
        return True
    except NotImplementedError as e:
        print(f"\n⚠️  {e}")
        print(f"   See: {output_dir / dataset_id / 'README.md'} for instructions")
        return True  # Not an error, just needs manual steps
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False


def download_recommended(output_dir: Path):
    """Download recommended starter datasets."""
    recommended = ["medquad", "meddialog"]
    
    print("\n📥 Downloading recommended starter datasets...")
    print("   Datasets: " + ", ".join(recommended))
    print()
    
    success = True
    for dataset_id in recommended:
        if not download_dataset(dataset_id, output_dir):
            success = False
    
    if success:
        print("\n✅ All recommended datasets downloaded!")
        print("\nNext steps:")
        print("  1. Review downloaded data in ./data/")
        print("  2. For ChestX-ray14, follow manual download instructions")
        print("  3. Run ingestion: docker-compose up -d celery-worker")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Download medical datasets for MedAI Compass",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                           # Show available datasets
  %(prog)s --dataset medquad -o ./data      # Download specific dataset
  %(prog)s --recommended -o ./data          # Download starter datasets
  %(prog)s --dataset mimic_iv -o ./data \\
      --physionet-user USER --physionet-pass PASS
  %(prog)s --dataset mimic_iv -o ./data --max-size 1.0 \\
      --physionet-user USER --physionet-pass PASS  # Download up to 1GB
        """
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available datasets"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="Dataset ID to download"
    )
    parser.add_argument(
        "--recommended", "-r",
        action="store_true",
        help="Download recommended starter datasets"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data",
        help="Output directory (default: ./data)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing downloads"
    )
    parser.add_argument(
        "--physionet-user",
        type=str,
        help="PhysioNet username (for MIMIC datasets)"
    )
    parser.add_argument(
        "--physionet-pass",
        type=str,
        help="PhysioNet password"
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=None,
        help="Maximum size to download in GB (for PhysioNet datasets)"
    )
    
    args = parser.parse_args()
    
    # Default to list if no action specified
    if not args.dataset and not args.recommended and not args.list:
        args.list = True
    
    if args.list:
        list_datasets()
        return 0
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.recommended:
        success = download_recommended(output_dir)
    elif args.dataset:
        # Check if this is a PhysioNet dataset with size limit
        if args.max_size and args.dataset in ["mimic_iv", "mimic_cxr"]:
            success = download_physionet_limited(
                args.dataset,
                output_dir,
                max_size_gb=args.max_size,
                physionet_user=args.physionet_user,
                physionet_pass=args.physionet_pass,
            )
        else:
            success = download_dataset(
                args.dataset,
                output_dir,
                force=args.force,
                physionet_user=args.physionet_user,
                physionet_pass=args.physionet_pass,
            )
    else:
        parser.print_help()
        return 1
    
    return 0 if success else 1


def download_physionet_limited(dataset_id: str, output_dir: Path, max_size_gb: float,
                               physionet_user: str = None, physionet_pass: str = None) -> bool:
    """Download PhysioNet dataset with size limit."""
    from medai_compass.utils.dataset_downloader import DatasetDownloader
    
    username = physionet_user or os.environ.get("PHYSIONET_USERNAME")
    password = physionet_pass or os.environ.get("PHYSIONET_PASSWORD")
    
    if not username or not password:
        print("❌ PhysioNet credentials required.")
        print("   Set --physionet-user and --physionet-pass, or environment variables:")
        print("   PHYSIONET_USERNAME and PHYSIONET_PASSWORD")
        return False
    
    print(f"\n📥 Downloading: {dataset_id} (max {max_size_gb}GB)")
    print(f"   Output: {output_dir / dataset_id}")
    print()
    
    downloader = DatasetDownloader(
        output_dir=output_dir,
        physionet_username=username,
        physionet_password=password,
    )
    
    def progress(msg, pct):
        print(f"   {pct:5.1f}% | {msg}")
    
    downloader.set_progress_callback(progress)
    
    try:
        path = downloader.download_physionet_limited(dataset_id, max_size_gb=max_size_gb)
        print(f"\n✅ Downloaded to: {path}")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False


if __name__ == "__main__":
    sys.exit(main())
