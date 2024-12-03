import os
import csv
import argparse
import scipy.io as sio

# Set up argument parser
parser = argparse.ArgumentParser(description="Generate .mat files from annotations CSV")
parser.add_argument("--csv_file", required=True, 
                    help="Path to the input CSV file containing annotations.")
parser.add_argument("--output_dir", required=True, 
                    help="Path to the directory where .mat files will be saved.")

args = parser.parse_args()

# Ensure the output directory exists
os.makedirs(args.output_dir, exist_ok=True)

print(f"Output directory verified: {args.output_dir}")

# Initialize a counter for the number of .mat files
mat_file_count = 0

# Process the CSV file
with open(args.csv_file, "r") as f:
    reader = csv.DictReader(f)
    print(f"Processing CSV file: {args.csv_file}")
    
    for i, row in enumerate(reader, start=1):
        video_name = row["video_name"]
        start_anno = int(row["start_annotation"])
        end_anno = int(row["end_annotation"])
        event_name = row["event_name"]

        # Extract second event information if available
        second_start_time = int(row["second_start_time"]) if row["second_start_time"].isdigit() else None
        second_end_time = int(row["seond_end_time"]) if row["seond_end_time"].isdigit() else None

        # Create the Anno list with primary and (if available) secondary event annotations
        anno_list = [[start_anno, end_anno]]
        if second_start_time is not None and second_end_time is not None:
            anno_list.append([second_start_time, second_end_time])

        # Create the dictionary to match the expected .mat file format
        data_dict = {
            "Annotation_file": {
                "Anno": anno_list,
                "EventName": event_name
            }
        }
        
        # Generate the output file name
        mat_file_name = f"{event_name}_{video_name}.mat"
        output_path = os.path.join(args.output_dir, mat_file_name)
        
        # Save the dictionary as a .mat file
        sio.savemat(output_path, data_dict)
        
        # Increment the .mat file counter
        mat_file_count += 1
        
        # Log progress
        print(f"[{i}] Saved .mat file: {output_path}")
        
# Print the total number of .mat files generated
print(f"Total .mat files generated: {mat_file_count}")
print("All files processed successfully!")
