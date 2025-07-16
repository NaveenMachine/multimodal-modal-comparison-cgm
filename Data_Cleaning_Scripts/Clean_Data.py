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
        ecg_df.columns = ['Time', 'EcgWaveform']
        ecg_df['Time'] = pd.to_datetime(ecg_df['Time'], format='%d/%m/%Y %H:%M:%S.%f', errors='coerce')
        ecg_df.dropna(subset=['Time'], inplace=True)
        
        # Set 'Time' as the index for resampling
        ecg_df.set_index('Time', inplace=True)
        
        # Resample to 1-minute intervals and calculate the mean
        ecg_resampled = ecg_df['EcgWaveform'].resample('1T').mean()
        
        # Convert back to a DataFrame
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
        cgm_df['Timestamp'] = pd.to_datetime(cgm_df['date'] + ' ' + cgm_df['time'], errors='coerce')
        cgm_df.dropna(subset=['Timestamp'], inplace=True)
        cgm_df = cgm_df[cgm_df['type'] == 'cgm'].copy()
        
        # Set 'Timestamp' as the index for resampling
        cgm_df.set_index('Timestamp', inplace=True)
        
        # Resample to 1-minute intervals and interpolate
        cgm_resampled = cgm_df['glucose'].resample('1T').interpolate(method='linear')
        
        # Convert back to a DataFrame
        cgm_resampled_df = cgm_resampled.reset_index()
        cgm_resampled_df.rename(columns={'glucose': 'Glucose'}, inplace=True)

        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        cgm_resampled_df.to_csv(output_filepath, index=False)
        print(f"Successfully processed CGM data and saved to {output_filepath}")

    except Exception as e:
        print(f"An error occurred while processing {input_filepath}: {e}")

def merge_resampled_data(cgm_file, ecg_files, output_filepath):
    """
    Merges one resampled CGM file with multiple resampled ECG files.

    - Reads and concatenates all ECG files.
    - Reads the CGM file.
    - Performs an outer merge on the Timestamp to combine all data.
    - Fills any remaining missing values after the merge.
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
        cgm_df = pd.read_csv(cgm_file, parse_dates=['Timestamp'])

        # Load and concatenate all processed ECG files
        ecg_df_list = [pd.read_csv(f, parse_dates=['Timestamp']) for f in ecg_files]
        all_ecg_df = pd.concat(ecg_df_list).sort_values(by='Timestamp').reset_index(drop=True)
        
        # Since ECG data is already averaged per minute per file, we can set index and merge
        cgm_df.set_index('Timestamp', inplace=True)
        all_ecg_df.set_index('Timestamp', inplace=True)

        # Merge CGM and ECG data using an outer join to keep all timestamps
        merged_df = pd.merge(cgm_df, all_ecg_df, on='Timestamp', how='outer')
        
        # After merging, it's a good practice to handle potential NaNs.
        # We can interpolate both signals again to fill gaps created by the outer merge.
        merged_df['Glucose'] = merged_df['Glucose'].interpolate(method='linear')
        merged_df['EcgWaveform'] = merged_df['EcgWaveform'].interpolate(method='linear')

        # Drop any rows that still have NaN values (usually at the very beginning or end)
        merged_df.dropna(inplace=True)
        
        # Round the final data to two decimal places
        merged_df[['Glucose', 'EcgWaveform']] = merged_df[['Glucose', 'EcgWaveform']].round(2)
        
        merged_df.reset_index(inplace=True)

        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        merged_df.to_csv(output_filepath, index=False)
        print(f"Successfully merged all data and saved to {output_filepath}")

    except Exception as e:
        print(f"An error occurred during the merge process: {e}")


if __name__ == '__main__':
    # Define file paths for a single patient
    patient_id = 'Patient_1'
    base_path = f'Patient_Data/{patient_id}'
    
    # Input files
    raw_ecg_files = [
        f'{base_path}/ecg_data/2014_10_01-10_09_39/2014_10_01-10_09_39_ECG.csv',
        f'{base_path}/ecg_data/2014_10_02-10_56_44/2014_10_02-10_56_44_ECG.csv',
        f'{base_path}/ecg_data/2014_10_03-06_36_24/2014_10_03-06_36_24_ECG.csv',
        f'{base_path}/ecg_data/2014_10_04-06_34_57/2014_10_04-06_34_57_ECG.csv',
    ]
    raw_cgm_file = f'{base_path}/glucose.csv'

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
