#!/usr/bin/env python3
"""
Dataset Balance and Quality Analysis Tool

Analyzes the collected landmark dataset and provides recommendations.
"""

import sys
import pathlib
import csv
from collections import defaultdict
from typing import Dict, List, Tuple

# Resolve project root
_SCRIPT_DIR = pathlib.Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from models.sign_detector import SignType


def load_dataset(csv_path: pathlib.Path) -> Tuple[Dict[str, int], int]:
    """Load dataset and count samples per gesture."""
    gesture_counts = defaultdict(int)
    total_samples = 0
    
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            
            for row in reader:
                if row:
                    label = row[0]
                    gesture_counts[label] += 1
                    total_samples += 1
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {csv_path}")
        return {}, 0
    
    return dict(gesture_counts), total_samples


def analyze_dataset(csv_path: pathlib.Path):
    """Analyze and report on dataset quality."""
    
    print("\n" + "="*70)
    print("DATASET ANALYSIS REPORT")
    print("="*70)
    
    gesture_counts, total_samples = load_dataset(csv_path)
    
    if not gesture_counts:
        return
    
    print(f"\nDataset file: {csv_path}")
    print(f"Total samples: {total_samples}")
    print(f"Unique gestures: {len(gesture_counts)}")
    
    # Expected gestures
    expected_gestures = {
        "Closed Fist",
        "Open Palm",
        "Thumbs Up",
        "Thumbs Down",
        "Index Finger",
        "Peace Sign",
        "OK Sign",
        "I Love You",
        "Shaka",
        "Vulcan Salute",
    }
    
    missing_gestures = expected_gestures - set(gesture_counts.keys())
    extra_gestures = set(gesture_counts.keys()) - expected_gestures
    
    # Print gesture distribution
    print("\n" + "-"*70)
    print("GESTURE DISTRIBUTION:")
    print("-"*70)
    
    # Sort by count (descending)
    sorted_gestures = sorted(gesture_counts.items(), key=lambda x: x[1], reverse=True)
    
    max_label_width = max(len(g) for g in gesture_counts.keys())
    
    for gesture, count in sorted_gestures:
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        
        # Visual bar
        bar_length = int(percentage / 2)  # Scale to reasonable size
        bar = "█" * bar_length
        
        # Status indicator
        if gesture in expected_gestures:
            status = "✓"
        else:
            status = "⚠"
        
        print(f"{status} {gesture:<{max_label_width}}: {count:4} ({percentage:5.1f}%) {bar}")
    
    # Check for issues
    print("\n" + "-"*70)
    print("QUALITY CHECKS:")
    print("-"*70)
    
    all_issues = []
    
    # Check 1: Total samples
    if total_samples < 50:
        all_issues.append((0, "⚠️ CRITICAL: Very small dataset (<50 samples)", 
                          f"Collected only {total_samples} samples. Recommend collecting at least 200."))
    elif total_samples < 200:
        all_issues.append((1, "⚠️ WARNING: Small dataset (<200 samples)",
                          f"Collected {total_samples} samples. Recommend 200+ for good results."))
    else:
        print(f"✓ Dataset size: {total_samples} samples (GOOD)")
    
    # Check 2: Missing gestures
    if missing_gestures:
        all_issues.append((1, f"❌ Missing gestures: {len(missing_gestures)}",
                          f"Missing: {', '.join(sorted(missing_gestures))}"))
    else:
        print(f"✓ All expected gestures present (10/10)")
    
    # Check 3: Balance
    if gesture_counts:
        avg_count = total_samples / len(gesture_counts)
        max_count = max(gesture_counts.values())
        min_count = min(gesture_counts.values())
        
        # Threshold: 20% deviation from average
        threshold = avg_count * 0.2
        imbalance = max_count - min_count
        
        if imbalance > threshold * 2:
            pct_diff = (imbalance / avg_count) * 100
            all_issues.append((2, f"⚠️ Dataset imbalance: {pct_diff:.1f}% difference",
                              f"Range: {min_count}-{max_count} samples"))
        else:
            print(f"✓ Dataset balance: GOOD (range: {min_count}-{max_count})")
    
    # Check 4: Per-gesture minimums
    min_samples_per_gesture = 20
    under_minimum = [(g, c) for g, c in gesture_counts.items() if c < min_samples_per_gesture]
    
    if under_minimum:
        all_issues.append((2, f"⚠️ Gestures with < {min_samples_per_gesture} samples: {len(under_minimum)}",
                          f"Gestures: {', '.join([g for g, _ in under_minimum])}"))
    else:
        print(f"✓ All gestures have >= {min_samples_per_gesture} samples")
    
    # Print issues if any
    if all_issues:
        print()
        all_issues.sort(key=lambda x: x[0])
        for severity, message, detail in all_issues:
            print(f"\n{message}")
            print(f"  → {detail}")
    
    # Recommendations
    print("\n" + "-"*70)
    print("RECOMMENDATIONS:")
    print("-"*70)
    
    recommendations = []
    
    if total_samples < 200:
        recommendations.append("Collect more samples to reach 200+ total")
    
    if missing_gestures:
        recommendations.append(f"Add missing gestures: {', '.join(sorted(missing_gestures))}")
    
    if under_minimum:
        gestures_to_boost = [g for g, _ in under_minimum]
        recommendations.append(f"Collect more samples for: {', '.join(gestures_to_boost)}")
    
    if imbalance > threshold * 2 if gesture_counts else False:
        recommendations.append("Balance dataset by collecting more samples for underrepresented gestures")
    
    if not recommendations:
        recommendations.append("Dataset looks good! Ready to train model.")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    # Training readiness
    print("\n" + "-"*70)
    print("TRAINING READINESS:")
    print("-"*70)
    
    readiness_score = 0
    max_score = 100
    
    # Size (30 points)
    if total_samples >= 500:
        readiness_score += 30
    elif total_samples >= 200:
        readiness_score += 20
    elif total_samples >= 50:
        readiness_score += 10
    
    # Completeness (30 points)
    if len(gesture_counts) == len(expected_gestures):
        readiness_score += 30
    elif len(gesture_counts) >= 8:
        readiness_score += 20
    elif len(gesture_counts) >= 5:
        readiness_score += 10
    
    # Balance (40 points)
    if gesture_counts and total_samples > 0:
        avg = total_samples / len(gesture_counts)
        deviation = sum(abs(c - avg) for c in gesture_counts.values()) / len(gesture_counts)
        balance_pct = 100 - (deviation / avg * 100)
        
        if balance_pct >= 95:
            readiness_score += 40
        elif balance_pct >= 90:
            readiness_score += 30
        elif balance_pct >= 80:
            readiness_score += 20
        elif balance_pct >= 70:
            readiness_score += 10
    
    # Determine readiness level
    if readiness_score >= 90:
        level = "EXCELLENT - Ready for training!"
        emoji = "🟢"
    elif readiness_score >= 75:
        level = "GOOD - Proceed with caution"
        emoji = "🟡"
    elif readiness_score >= 50:
        level = "MARGINAL - Consider collecting more"
        emoji = "🟠"
    else:
        level = "POOR - Collect more data first"
        emoji = "🔴"
    
    print(f"{emoji} Readiness Score: {readiness_score}/{max_score}")
    print(f"   Status: {level}")
    
    # Next steps
    print("\n" + "-"*70)
    print("NEXT STEPS:")
    print("-"*70)
    
    if readiness_score >= 75:
        print("\n✓ Ready to train model!")
        print("  Command: python scripts/train_xgboost_model.py")
    else:
        print("\n⚠️ Collect more data before training")
        print("  Command: python scripts/collect_training_data.py")
    
    print("\n" + "="*70 + "\n")


def main():
    """Main entry point."""
    dataset_path = _PROJECT_ROOT / "data" / "training_data" / "landmark_data.csv"
    
    print(f"\nAnalyzing dataset: {dataset_path}")
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        print("\nTo create a dataset, run:")
        print("  python scripts/collect_training_data.py")
        return
    
    analyze_dataset(dataset_path)


if __name__ == "__main__":
    main()
