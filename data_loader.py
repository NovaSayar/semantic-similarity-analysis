# Data loader for CSV prompts

import pandas as pd

def load_prompts_from_csv(csv_file_path):
    """Load and organize prompts from CSV file"""
    try:
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} prompts from {csv_file_path}")
        
        # Display CSV structure
        print(f"CSV columns: {list(df.columns)}")
        print(f"Categories: {df['Category'].unique()}")
        print(f"Sample data:")
        print(df.head())
        
        # Organize prompts by category and type
        prompts_dict = {}
        
        for category in df['Category'].unique():
            category_data = df[df['Category'] == category]
            
            category_lower = category.lower()

            if category_lower.startswith('vague_place') or category_lower.startswith('structured_place'):
                main_category = 'place'
            else:
                main_category = 'non_place'

            
            # Determine if it's vague or structured
            if 'Vague' in category:
                prompt_type = 'vague'
                prompt_column = 'Vague_Prompt'
            else:
                prompt_type = 'structured'
                prompt_column = 'Structured_Prompt'
            
            # Initialize nested dict structure
            if main_category not in prompts_dict:
                prompts_dict[main_category] = {}
            if prompt_type not in prompts_dict[main_category]:
                prompts_dict[main_category][prompt_type] = []
            
            # Add prompts using the correct column
            for _, row in category_data.iterrows():
                prompts_dict[main_category][prompt_type].append(row[prompt_column])
        
        # Print summary
        for main_cat, types in prompts_dict.items():
            for ptype, prompts in types.items():
                print(f"{main_cat.upper()} - {ptype.upper()}: {len(prompts)} prompts")
        
        return prompts_dict
        
    except Exception as e:
        print(f"Error loading prompts from CSV: {e}")
        raise
