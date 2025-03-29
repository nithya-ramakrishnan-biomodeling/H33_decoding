import os
import pandas as pd

def process_csv(input_dir: str, output_dir: str, window_size: int = 1000):
    """
    Processes CSV files in the input directory, extracts sequences based on H3.1 and H3.3 values,
    and saves the filtered sequences to the output directory.

    Parameters:
    input_dir (str): Path to the directory containing input CSV files.
    output_dir (str): Path to the directory where processed CSV files will be saved.
    window_size (int, optional): Size of the sliding window for processing. Default is 1000.
    """
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through all CSV files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_dir, file_name)
            df = pd.read_csv(file_path, sep=',')
            results = []
            
            # Process each sliding window in the dataframe
            for start_idx in range(len(df) - window_size + 1):
                window = df.iloc[start_idx:start_idx + window_size]

                # Generate 'source_sequence' based on H3.1 and H3.3, skipping invalid rows
                source_sequence = [
                    'true' if h3_1 == 1 and h3_3 == 0 else 'false' if h3_3 == 1 and h3_1 == 0 else None
                    for h3_1, h3_3 in zip(window['H3.1'], window['H3.3'])
                    if not (h3_1 == 0 and h3_3 == 0)
                ]
                
                if source_sequence:  # Add only if valid sequences exist
                    results.append({'source': source_sequence})
            
            # Save results if there is valid data
            if results:
                result_df = pd.DataFrame(results)
                output_file = os.path.join(output_dir, f"filtered_{file_name}")
                result_df.to_csv(output_file, index=False)
                print(f"Processed: {file_name} -> {output_file}")

if __name__ == "__main__":
    # Get user input for directories
    input_directory = input("Enter the input directory path: ").strip()
    output_directory = input("Enter the output directory path: ").strip()
    
    process_csv(input_directory, output_directory)
