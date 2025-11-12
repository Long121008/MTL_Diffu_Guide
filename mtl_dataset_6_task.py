"""
Multi-Task Learning Dataset Builder (FIXED VERSION)

Creates unified ground truth dataset for 6 VRP tasks with:
- Proper task labeling
- Standardized problem format
- Tour validation
- Statistics & visualization
"""

import pickle
import os
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any

# =========================================================================
# CONFIGURATION
# =========================================================================

# Problem files (raw data from Step 1)
PROBLEM_FILES = {
    'CVRP': './data_train3phase/CVRP/cvrp50_uniform.pkl',
    'OVRP': './data_train3phase/OVRP/ovrp50_uniform.pkl',
    'VRPB': './data_train3phase/VRPB/vrpb50_uniform.pkl',
    'VRPL': './data_train3phase/VRPL/vrpl50_uniform.pkl',
    'VRPTW': './data_train3phase/VRPTW/vrptw50_uniform.pkl',
    'OVRPTW': './data_train3phase/OVRPTW/ovrptw50_uniform.pkl',
}

# Solution files (from LKH3/OR-Tools in Step 2)
RESULT_FILES = {
    'CVRP': './data_train3phase/CVRP/or_tools_200s_cvrp50_uniform.pkl',
    'OVRP': './data_train3phase/OVRP/or_tools_200s_ovrp50_uniform.pkl',
    'VRPB': './data_train3phase/VRPB/or_tools_200s_vrpb50_uniform.pkl',
    'VRPL': './data_train3phase/VRPL/or_tools_200s_vrpl50_uniform.pkl',
    'VRPTW': './data_train3phase/VRPTW/or_tools_200s_vrptw50_uniform.pkl',
    'OVRPTW': './data_train3phase/OVRPTW/or_tools_200s_ovrptw50_uniform.pkl',
}

# Output file
FINAL_OUTPUT_FILE = './data/train/mtl_6tasks_20k_groundtruth.pkl'

TASKS = ['CVRP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW', 'OVRPTW']

# Validation thresholds
MAX_COST_RATIO = 3.0  # Reject if cost > 3x optimal
MIN_TOUR_LENGTH = 2   # Must visit at least 1 customer


# =========================================================================
# UTILITY FUNCTIONS
# =========================================================================

def load_pickle(path: str) -> Any:
    """Load pickle file with error handling"""
    print(f"  → Loading: {path}")
    
    if not os.path.exists(path):
        print(f"  ✗ ERROR: File not found!")
        return None
    
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"  ✓ Loaded {len(data)} samples")
        return data
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return None


def standardize_problem_format(task: str, problem_tuple: Tuple) -> Dict:
    """
    Convert problem tuple to standardized dict format
    
    Args:
        task: Task name ('CVRP', 'VRPTW', etc.)
        problem_tuple: Raw problem data tuple
    
    Returns:
        problem_dict: Standardized format with all necessary fields
    """
    problem_dict = {
        'task': task,
        'problem_size': None,  # Will be filled
    }
    
    # === COMMON FIELDS (All tasks) ===
    if task in ['CVRP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW', 'OVRPTW']:
        problem_dict['depot_xy'] = problem_tuple[0]      # (2,) or list
        problem_dict['node_xy'] = problem_tuple[1]       # (n, 2)
        problem_dict['node_demand'] = problem_tuple[2]   # (n,)
        problem_dict['capacity'] = problem_tuple[3]      # scalar
        
        # Infer problem size
        if isinstance(problem_tuple[1], (list, tuple)):
            problem_dict['problem_size'] = len(problem_tuple[1])
        else:
            problem_dict['problem_size'] = problem_tuple[1].shape[0] if hasattr(problem_tuple[1], 'shape') else len(problem_tuple[1])
    
    # === LINEHAUL (VRPL variants) ===
    if task in ['VRPL', 'VRPBL', 'OVRPBL', 'OVRPL']:
        problem_dict['route_limit'] = problem_tuple[4]
    
    # === TIME WINDOWS (TW variants) ===
    if task in ['VRPTW', 'OVRPTW', 'VRPBTW', 'VRPLTW', 'OVRPBLTW', 'OVRPLTW']:
        # Adjust offset if linehaul present
        offset = 1 if 'L' in task else 0
        problem_dict['service_time'] = problem_tuple[4 + offset]
        problem_dict['tw_start'] = problem_tuple[5 + offset]
        problem_dict['tw_end'] = problem_tuple[6 + offset]
    
    # === BACKHAUL (B variants) ===
    if task in ['VRPB', 'OVRPB', 'VRPBL', 'VRPBTW', 'OVRPBL', 'OVRPBTW', 'VRPBLTW', 'OVRPBLTW']:
        # Backhaul data might be in node_demand or separate field
        # Adjust based on your actual data format
        pass  # Add if needed
    
    return problem_dict


def standardize_tour_format(task: str, tour_data: Any) -> Dict:
    """
    Standardize tour format
    
    Args:
        task: Task name
        tour_data: (tour, cost) tuple or other format
    
    Returns:
        tour_dict: {
            'tour': List of node indices or list of routes,
            'cost': float,
            'num_routes': int,
            'tour_type': 'single' or 'multiple'
        }
    """
    if isinstance(tour_data, tuple) and len(tour_data) == 2:
        cost, tour = tour_data
    else:
        tour = tour_data
        cost = None
    
    tour_dict = {}
    
    # Determine tour type
    if isinstance(tour, list):
        if len(tour) > 0 and isinstance(tour[0], list):
            # Multiple routes: [[0,1,2,0], [0,3,4,0]]
            tour_dict['tour_type'] = 'multiple'
            tour_dict['num_routes'] = len(tour)
            tour_dict['tour'] = tour
        else:
            # Single route: [0, 1, 2, 3, 0]
            tour_dict['tour_type'] = 'single'
            tour_dict['num_routes'] = 1
            tour_dict['tour'] = tour
    else:
        # Assume single tour
        tour_dict['tour_type'] = 'single'
        tour_dict['num_routes'] = 1
        tour_dict['tour'] = tour
    
    tour_dict['cost'] = float(cost) if cost is not None else None
    
    return tour_dict


def validate_sample(problem_dict: Dict, tour_dict: Dict, task: str) -> Tuple[bool, str]:
    """
    Validate a single problem-tour pair
    
    Returns:
        (is_valid, error_message)
    """
    # Check 1: Problem has required fields
    required_fields = ['depot_xy', 'node_xy', 'node_demand', 'capacity']
    for field in required_fields:
        if field not in problem_dict:
            return False, f"Missing field: {field}"
    
    # Check 2: Tour exists
    if 'tour' not in tour_dict or tour_dict['tour'] is None:
        return False, "Tour is None"
    
    # Check 3: Tour visits nodes
    tour = tour_dict['tour']
    if tour_dict['tour_type'] == 'single':
        if len(tour) < MIN_TOUR_LENGTH:
            return False, f"Tour too short: {len(tour)}"
    else:
        total_nodes = sum(len(route) for route in tour)
        if total_nodes < MIN_TOUR_LENGTH:
            return False, f"Total tour length too short: {total_nodes}"
    
    # Check 4: Cost is reasonable
    if tour_dict['cost'] is not None:
        problem_size = problem_dict['problem_size']
        # Heuristic: cost should be < 3x problem size (for normalized coords)
        if tour_dict['cost'] > problem_size * MAX_COST_RATIO:
            return False, f"Cost too high: {tour_dict['cost']:.2f}"
    
    # Check 5: Node indices in valid range
    problem_size = problem_dict['problem_size']
    if tour_dict['tour_type'] == 'single':
        max_idx = max(tour) if isinstance(tour, list) else tour.max()
        if max_idx >= problem_size + 1:  # +1 for depot
            return False, f"Invalid node index: {max_idx}"
    
    return True, "OK"


def compute_statistics(dataset: List[Dict]) -> Dict:
    """Compute dataset statistics"""
    stats = {
        'total_samples': len(dataset),
        'by_task': defaultdict(int),
        'cost_stats': defaultdict(list),
        'problem_size_stats': defaultdict(list),
    }
    
    for sample in dataset:
        task = sample['task']
        stats['by_task'][task] += 1
        
        if 'cost' in sample['tour_info'] and sample['tour_info']['cost'] is not None:
            stats['cost_stats'][task].append(sample['tour_info']['cost'])
        
        if 'problem_size' in sample['problem_dict']:
            stats['problem_size_stats'][task].append(sample['problem_dict']['problem_size'])
    
    # Compute mean/std
    for task in stats['by_task'].keys():
        if stats['cost_stats'][task]:
            costs = stats['cost_stats'][task]
            stats['cost_stats'][task] = {
                'mean': np.mean(costs),
                'std': np.std(costs),
                'min': np.min(costs),
                'max': np.max(costs),
            }
    
    return stats


def print_statistics(stats: Dict):
    """Pretty print statistics"""
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    print(f"\nTotal samples: {stats['total_samples']}")
    
    print("\nSamples per task:")
    for task, count in sorted(stats['by_task'].items()):
        percentage = count / stats['total_samples'] * 100
        print(f"  {task:10s}: {count:6d} ({percentage:5.2f}%)")
    
    print("\nCost statistics per task:")
    for task, cost_stats in sorted(stats['cost_stats'].items()):
        if isinstance(cost_stats, dict):
            print(f"  {task:10s}: mean={cost_stats['mean']:7.2f}, "
                  f"std={cost_stats['std']:6.2f}, "
                  f"min={cost_stats['min']:6.2f}, "
                  f"max={cost_stats['max']:7.2f}")
    
    print("="*80)


# =========================================================================
# MAIN PROCESSING
# =========================================================================

def main():
    """Main processing pipeline"""
    
    print("="*80)
    print("MULTI-TASK DATASET BUILDER (Phase 2 Ground Truth)")
    print("="*80)
    
    final_dataset = []
    validation_errors = defaultdict(int)
    
    # Process each task
    for task in TASKS:
        print(f"\n{'─'*80}")
        print(f"Processing task: {task}")
        print(f"{'─'*80}")
        
        # Load data
        problem_data_list = load_pickle(PROBLEM_FILES[task])
        results_list = load_pickle(RESULT_FILES[task])
        
        if problem_data_list is None or results_list is None:
            print(f"  ✗ Skipping {task} due to missing files")
            continue
        
        # Check length match
        if len(problem_data_list) != len(results_list):
            print(f"  ✗ ERROR: Length mismatch!")
            print(f"    Problems: {len(problem_data_list)}, Results: {len(results_list)}")
            continue
        
        # Process each sample
        valid_count = 0
        for i in range(len(problem_data_list)):
            # Standardize formats
            problem_dict = standardize_problem_format(task, problem_data_list[i])
            tour_dict = standardize_tour_format(task, results_list[i])
            
            # Validate
            is_valid, error_msg = validate_sample(problem_dict, tour_dict, task)
            
            if not is_valid:
                validation_errors[f"{task}_{error_msg}"] += 1
                continue
            
            # Create unified sample
            sample = {
                'task': task,
                'problem_dict': problem_dict,
                'tour_info': tour_dict,
                'sample_id': f"{task}_{i:06d}",
            }
            
            final_dataset.append(sample)
            valid_count += 1
        
        print(f"  ✓ Processed {valid_count}/{len(problem_data_list)} valid samples")
    
    # Print validation errors if any
    if validation_errors:
        print(f"\n{'─'*80}")
        print("VALIDATION ERRORS:")
        for error, count in sorted(validation_errors.items(), key=lambda x: -x[1]):
            print(f"  {error}: {count}")
    
    # Compute and print statistics
    stats = compute_statistics(final_dataset)
    print_statistics(stats)
    
    # Save dataset
    print(f"\n{'─'*80}")
    print("SAVING DATASET")
    print(f"{'─'*80}")
    
    output_dir = os.path.dirname(FINAL_OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"  → Created directory: {output_dir}")
    
    with open(FINAL_OUTPUT_FILE, 'wb') as f:
        pickle.dump(final_dataset, f, pickle.HIGHEST_PROTOCOL)
    
    print(f"  ✓ Saved {len(final_dataset)} samples to:")
    print(f"    {FINAL_OUTPUT_FILE}")
    
    # Save statistics separately
    stats_file = FINAL_OUTPUT_FILE.replace('.pkl', '_stats.pkl')
    with open(stats_file, 'wb') as f:
        pickle.dump(stats, f)
    print(f"  ✓ Saved statistics to:")
    print(f"    {stats_file}")
    
    print("\n" + "="*80)
    print("✅ DATASET BUILD COMPLETE!")
    print("="*80)
    
    # Print usage instructions
    print("\nUSAGE:")
    print("  In your Trainer.py:")
    print(f"    dataset = pickle.load(open('{FINAL_OUTPUT_FILE}', 'rb'))")
    print("    for sample in dataset:")
    print("        task = sample['task']")
    print("        problem = sample['problem_dict']")
    print("        optimal_tour = sample['tour_info']['tour']")
    print("        cost = sample['tour_info']['cost']")


# =========================================================================
# EXAMPLE USAGE
# =========================================================================

def example_load_and_use():
    """Example: How to load and use the dataset in training"""
    
    # Load dataset
    with open(FINAL_OUTPUT_FILE, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Loaded {len(dataset)} samples")
    
    # Example: Filter by task
    cvrp_samples = [s for s in dataset if s['task'] == 'CVRP']
    print(f"CVRP samples: {len(cvrp_samples)}")
    
    # Example: Access data
    sample = dataset[0]
    print(f"\nExample sample:")
    print(f"  Task: {sample['task']}")
    print(f"  Problem size: {sample['problem_dict']['problem_size']}")
    print(f"  Optimal cost: {sample['tour_info']['cost']}")
    print(f"  Tour type: {sample['tour_info']['tour_type']}")
    print(f"  Tour: {sample['tour_info']['tour'][:10]}...")  # First 10 nodes


if __name__ == "__main__":
    main()
    
    # Uncomment to test loading:
    # example_load_and_use()