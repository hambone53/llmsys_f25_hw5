import json
import statistics
import glob
import os

def analyze_training_results(workdir_path):
    """
    Analyze training statistics from all _results_ JSON files in the workdir.
    Separates analysis by rank (rank0 vs rank1).
    
    Args:
        workdir_path (str): Path to the workdir containing the results files
    """
    # Find all _results_ JSON files
    results_pattern = os.path.join(workdir_path, "*_results_*.json")
    results_files = glob.glob(results_pattern)
    
    if not results_files:
        print("No _results_ JSON files found in the workdir.")
        return
    
    # Separate files by rank
    rank0_files = [f for f in results_files if "rank0_results_" in os.path.basename(f)]
    rank1_files = [f for f in results_files if "rank1_results_" in os.path.basename(f)]
    
    print(f"Found {len(results_files)} results files total:")
    print(f"  - Rank0 files: {len(rank0_files)}")
    print(f"  - Rank1 files: {len(rank1_files)}")
    print()
    
    # Analyze each rank separately
    for rank_name, rank_files in [("Rank 0", rank0_files), ("Rank 1", rank1_files)]:
        if not rank_files:
            print(f"No {rank_name} files found.\n")
            continue
            
        print(f"=== {rank_name} Results ===")
        print(f"Files analyzed:")
        for file in sorted(rank_files):
            print(f"  - {os.path.basename(file)}")
        print()
        
        # Collect training times and tokens per second for this rank
        training_times = []
        tokens_per_sec = []
        
        for file_path in rank_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # Extract metrics
                    training_time = data.get('training_time')
                    tokens_per_sec_val = data.get('tokens_per_sec')
                    
                    if training_time is not None:
                        training_times.append(training_time)
                    if tokens_per_sec_val is not None:
                        tokens_per_sec.append(tokens_per_sec_val)
                        
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        # Calculate and display statistics for this rank
        if training_times:
            avg_training_time = statistics.mean(training_times)
            std_training_time = statistics.stdev(training_times) if len(training_times) > 1 else 0
            
            print("Training Time Statistics:")
            print(f"  Number of samples: {len(training_times)}")
            print(f"  Average: {avg_training_time:.6f} seconds")
            print(f"  Standard Deviation: {std_training_time:.6f} seconds")
            print(f"  Min: {min(training_times):.6f} seconds")
            print(f"  Max: {max(training_times):.6f} seconds")
            print()
        
        if tokens_per_sec:
            avg_tokens_per_sec = statistics.mean(tokens_per_sec)
            std_tokens_per_sec = statistics.stdev(tokens_per_sec) if len(tokens_per_sec) > 1 else 0
            
            print("Tokens Per Second Statistics:")
            print(f"  Number of samples: {len(tokens_per_sec)}")
            print(f"  Average: {avg_tokens_per_sec:.2f} tokens/sec")
            print(f"  Standard Deviation: {std_tokens_per_sec:.2f} tokens/sec")
            print(f"  Min: {min(tokens_per_sec):.2f} tokens/sec")
            print(f"  Max: {max(tokens_per_sec):.2f} tokens/sec")
        
        print("-" * 50)
        print()

if __name__ == "__main__":
    workdir = "/jet/home/rhamor/projects/llmsys_f25_hw5/workdir"
    analyze_training_results(workdir)