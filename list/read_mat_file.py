from scipy.io import loadmat

# Load the .mat file
data = loadmat('/home/aishaeld/scratch/RTFM/list/Matlab_formate/Arrest001_x264.mat')

# Access the variable
annotation_file_content = data['Annotation_file']

# Print or explore the content
print(annotation_file_content)