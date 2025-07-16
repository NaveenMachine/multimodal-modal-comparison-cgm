import pandas as pd
import os
import numpy as np

def process_ecg_data(input_filepath, output_filepath):
    """
    Cleans, resamples, and aggregates ECG data from a CSV file.

    - Reads the data.
    - Converts the 'Time' column to datetime objects.
    - Sets the 'Time' column as the index.
    - Resamples the data to 1-minute intervals, calculating the mean for each interval.
    - Saves the processed data to a new CSV file.

    Args:
        input_filepath (str): The path to the input ECG data file.
        output_filepath (str): The path to save the processed ECG data file.
    """
    try:
        print(f"Processing ECG file: {input_filepath}...")
        ecg_df = pd.read_csv(input_filepath)
        if ecg_df.empty:
            print(f"Warning: Input ECG file is empty: {input_filepath}")
            return
        ecg_df.columns = ['Time', 'EcgWaveform']
        ecg_df['Time'] = pd.to_datetime(ecg_df['Time'], format='%d/%m/%Y %H:%M:%S.%f', errors='coerce')
        ecg_df.dropna(subset=['Time'], inplace=True)
        
        ecg_df.set_index('Time', inplace=True)
        ecg_resampled = ecg_df['EcgWaveform'].resample('1T').mean()
        
        ecg_resampled_df = ecg_resampled.reset_index()
        ecg_resampled_df.rename(columns={'Time': 'Timestamp'}, inplace=True)

        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        ecg_resampled_df.to_csv(output_filepath, index=False)
        print(f"Successfully processed ECG data and saved to {output_filepath}")
        
    except Exception as e:
        print(f"An error occurred while processing {input_filepath}: {e}")

def process_cgm_data(input_filepath, output_filepath):
    """
    Cleans, resamples, and interpolates CGM data from a CSV file.

    - Reads the data and creates a 'Timestamp' column.
    - Filters for 'cgm' type data.
    - Sets 'Timestamp' as the index.
    - Resamples to 1-minute intervals and uses linear interpolation to fill missing values.
    - Saves the processed data to a new CSV file.

    Args:
        input_filepath (str): The path to the input CGM data file.
        output_filepath (str): The path to save the processed CGM data file.
    """
    try:
        print(f"Processing CGM file: {input_filepath}...")
        cgm_df = pd.read_csv(input_filepath)
        if cgm_df.empty:
            print(f"Warning: Input CGM file is empty: {input_filepath}")
            return
        cgm_df['Timestamp'] = pd.to_datetime(cgm_df['date'] + ' ' + cgm_df['time'], errors='coerce')
        cgm_df.dropna(subset=['Timestamp'], inplace=True)
        cgm_df = cgm_df[cgm_df['type'] == 'cgm'].copy()
        
        cgm_df.set_index('Timestamp', inplace=True)
        cgm_resampled = cgm_df['glucose'].resample('1T').mean().interpolate(method='linear')
        
        cgm_resampled_df = cgm_resampled.reset_index()
        cgm_resampled_df.rename(columns={'glucose': 'Glucose'}, inplace=True)

        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        cgm_resampled_df.to_csv(output_filepath, index=False)
        print(f"Successfully processed CGM data and saved to {output_filepath}")

    except Exception as e:
        print(f"An error occurred while processing {input_filepath}: {e}")

def merge_resampled_data(cgm_file, ecg_files, output_filepath):
    """
    Merges one resampled CGM file with multiple resampled ECG files using an INNER join.

    - Reads and concatenates all ECG files.
    - Reads the CGM file.
    - Performs an INNER merge on the Timestamp to combine data only where timestamps overlap.
    - Rounds the numerical columns to two decimal places.
    - Saves the final merged dataset.

    Args:
        cgm_file (str): Path to the processed CGM file.
        ecg_files (list): A list of paths to the processed ECG files.
        output_filepath (str): The path to save the final merged data file.
    """
    try:
        print("\nMerging all resampled data...")
        
        # Load the processed CGM data
        if not os.path.exists(cgm_file):
            print(f"Error: Processed CGM file not found: {cgm_file}. Aborting merge.")
            return
        cgm_df = pd.read_csv(cgm_file, parse_dates=['Timestamp'])
        if cgm_df.empty:
            print("Warning: CGM data is empty after processing. Merge will likely be empty.")
            return

        # Load and concatenate all processed ECG files
        ecg_df_list = []
        for f in ecg_files:
            if os.path.exists(f):
                df = pd.read_csv(f, parse_dates=['Timestamp'])
                if not df.empty:
                    ecg_df_list.append(df)
            else:
                print(f"Warning: Processed ECG file not found: {f}. Skipping.")
        
        if not ecg_df_list:
            print("Error: No valid processed ECG data found. Aborting merge.")
            return
            
        all_ecg_df = pd.concat(ecg_df_list).sort_values(by='Timestamp').reset_index(drop=True)
        all_ecg_df.dropna(inplace=True) # Drop rows with NaN from resampling before merge

        print(f"Diagnostic: CGM data has {len(cgm_df)} rows. Time range: {cgm_df['Timestamp'].min()} to {cgm_df['Timestamp'].max()}")
        print(f"Diagnostic: Combined ECG data has {len(all_ecg_df)} rows. Time range: {all_ecg_df['Timestamp'].min()} to {all_ecg_df['Timestamp'].max()}")

        # Perform an INNER merge to keep only overlapping timestamps
        merged_df = pd.merge(cgm_df, all_ecg_df, on='Timestamp', how='inner')
        
        print(f"Diagnostic: Found {len(merged_df)} overlapping rows after merge.")

        if merged_df.empty:
            print("Warning: The merged file is empty. This means there were no overlapping timestamps between the CGM and ECG data.")
            # Still save the empty file with headers so the process completes
            pd.DataFrame(columns=['Timestamp', 'Glucose', 'EcgWaveform']).to_csv(output_filepath, index=False)
            return

        # Round the final data to two decimal places
        merged_df[['Glucose', 'EcgWaveform']] = merged_df[['Glucose', 'EcgWaveform']].round(2)
        
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        merged_df.to_csv(output_filepath, index=False)
        print(f"Successfully merged all data and saved to {output_filepath}")

    except Exception as e:
        print(f"An error occurred during the merge process: {e}")


if __name__ == '__main__':
    # Define file paths for a single patient
    patient_id = 'Patient_9'
    base_path = f'Patient_Data/{patient_id}'
    
    # Input files
    raw_ecg_files = [
        f'Patient_Data/Patient_9/ecg_data/2014_10_01-05_59_30/2014_10_01-05_59_30_ECG.csv',
        f'Patient_Data/Patient_9/ecg_data/2014_10_02-06_14_52/2014_10_02-06_14_52_ECG.csv',
        f'Patient_Data/Patient_9/ecg_data/2014_10_03-08_21_59/2014_10_03-08_21_59_ECG.csv',
        f'Patient_Data/Patient_9/ecg_data/2014_10_04-09_09_29/2014_10_04-09_09_29_ECG.csv',
        f'Patient_Data/Patient_9/ecg_data/2014_10_04-15_03_37/2014_10_04-15_03_37_ECG.csv',
    ]
    raw_cgm_file = f'Patient_Data/Patient_9/glucose.csv'

    # Define paths for processed (cleaned and resampled) files
    processed_ecg_files = [f.replace('.csv', '_processed.csv') for f in raw_ecg_files]
    processed_cgm_file = raw_cgm_file.replace('.csv', '_processed.csv')

    # --- Step 1: Process each individual file ---
    for raw_file, processed_file in zip(raw_ecg_files, processed_ecg_files):
        if os.path.exists(raw_file):
            process_ecg_data(raw_file, processed_file)
        else:
            print(f"File not found: {raw_file}. Skipping.")
    
    if os.path.exists(raw_cgm_file):
        process_cgm_data(raw_cgm_file, processed_cgm_file)
    else:
        print(f"File not found: {raw_cgm_file}. Skipping.")

    # --- Step 2: Merge all processed files for the patient ---
    final_output_file = f'{base_path}/{patient_id}_merged_data.csv'
    merge_resampled_data(processed_cgm_file, processed_ecg_files, final_output_file)

    print("\n--- Data processing pipeline complete! ---")
