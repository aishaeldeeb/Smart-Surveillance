import csv
import scipy.io as sio

# define the path to your CSV file
csv_file = "annotations.csv"

with open(csv_file, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        video_name = row["video_name"]
        start_anno = int(row["start_annotation"])
        end_anno = int(row["end_annotation"])
        event_name = row["event_name"]
        
        # create the dictionary to be saved as a .mat file
        data_dict = {
            "VideoName": video_name,
            "Anno": [start_anno, end_anno],
            "EventName": event_name
        }
        
        # save the dictionary as a .mat file
        mat_file_name = f"{event_name}_{video_name}.mat"

        
        # specify path of output file in mat_files directory
        output_path = f'mat_files/{video_name}.mat'
        
        # save dictionary as mat file
        sio.savemat(output_path, data_dict)
