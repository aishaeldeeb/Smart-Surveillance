import os

# The directory containing the files
directory = "/home/aishaeld/scratch/RTFM/test_data_i3d"

# The list to store the filenames
filenames = []

# Loop through all the files in the directory
for filename in os.listdir(directory):
    # Construct the full path to the file
    file_path = os.path.join(directory, filename)
    # Check if the file is a regular file
    if os.path.isfile(file_path):
        # Add the file to the list
        filenames.append(file_path)

# Write the list of filenames to the file
with open("test_i3d_list.list", "w") as f:
    for filename in filenames:
        f.write(filename + "\n")