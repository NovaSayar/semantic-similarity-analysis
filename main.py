# Main execution script

from pipeline import SemanticSimilarityPipeline
from analysis import run_full_analysis, run_pilot_study_tests

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = SemanticSimilarityPipeline()
    
    # Run full analysis with CSV prompts
    csv_file_path = "prompt_matrix.csv"  # Update path as needed
    
    try:
        results_df = run_full_analysis(pipeline, csv_file_path)
        
        if results_df is not None and len(results_df) > 0:
            # Run pilot study statistical analysis (non-parametric tests)
            run_pilot_study_tests(results_df)
            
            print(f"\nPilot study analysis completed successfully!")
            print(f"Processed {len(results_df)} prompts")
            print(f"Results saved with timestamp")
            print(f"Bootstrap confidence intervals included for robust statistical analysis")
            print(f"Non-parametric tests used for robust small-sample analysis")
        else:
            print("No results to analyze")
            
    except FileNotFoundError:
        print(f"CSV file '{csv_file_path}' not found. Please check the file path.")
    except Exception as e:
        print(f"Error during analysis: {e}")
