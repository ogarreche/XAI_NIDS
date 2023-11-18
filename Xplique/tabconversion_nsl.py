import pandas as pd
import os
from IGTD_Functions import min_max_transform, table_to_image, select_features_by_variation



num_row = 4 #30    # Number of pixel rows in image representation
num_col = 4 # 30    # Number of pixel columns in image representation
num = num_row * num_col # Number of features to be included for analysis, which is also the total number of pixels in image representation
save_image_size = 3 # Size of pictures (in inches) saved during the execution of IGTD algorithm.
max_step = 10000    # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
val_step = 300  # The number of iterations for determining algorithm convergence. If the error reduction rate
                # is smaller than a pre-set threshold for val_step itertions, the algorithm converges.



# Using top features
req_cols =["flag",
           "src_bytes",
           "dst_bytes",
           "hot",
          "num_failed_logins",
          "num_compromised",
          "diff_srv_rate",

          "dst_host_same_srv_rate",
          "dst_host_srv_count",
          "dst_host_same_src_port_rate",
          "logged_in",
          "dst_host_serror_rate",
          "count",
          "srv_count",
          "dst_host_rerror_rate"
          ]
# 'KDDTrain+.txt'
# KDDTest+.txt
# Import the example data and linearly scale each feature so that its minimum and maximum values are 0 and 1, respectively.
# ('cicids_db/Wednesday-workingHours.pcap_ISCX.csv', usecols=req_cols)
# data = pd.read_csv ('KDDTrain+.txt', usecols=req_cols, low_memory=False, engine='c',
#                    na_values=['na', '-', ''], header=0, index_col=0)

data = pd.read_csv ('KDDTrain+.txt', names=req_cols)#, low_memory=False, engine='c',
                   #na_values=['na', '-', ''], header=0, index_col=0)
# Select features with large variations across samples


import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Assuming df is your DataFrame and req_cols are the required columns
# You need to replace 'your_dataset.csv' with the actual file path or dataset variable

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
df = data

# Subset DataFrame to only include the required columns
df_subset = df[req_cols]

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Identify and label encode categorical columns
for col in req_cols:
    if df_subset[col].dtype == 'object':  # Check if the column is categorical
        df_subset[col] = label_encoder.fit_transform(df_subset[col])

# Display the modified DataFrame
# print(df_subset)

data = df_subset

# print(data)
id = select_features_by_variation(data, variation_measure='var', num=num)
data = data.iloc[:, id]
# Perform min-max transformation so that the maximum and minimum values of every feature become 1 and 0, respectively.
norm_data = min_max_transform(data.values)
norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)

# Run the IGTD algorithm using (1) the Euclidean distance for calculating pairwise feature distances and pariwise pixel
# distances and (2) the absolute function for evaluating the difference between the feature distance ranking matrix and
# the pixel distance ranking matrix. Save the result in Test_1 folder.
fea_dist_method = 'Euclidean'
image_dist_method = 'Euclidean'
error = 'abs'
result_dir = '../Results/Table_To_Image_Conve_Nsl_dos/Test_1'
os.makedirs(name=result_dir, exist_ok=True)
table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
               max_step, val_step, result_dir, error)

# Run the IGTD algorithm using (1) the Pearson correlation coefficient for calculating pairwise feature distances,
# (2) the Manhattan distance for calculating pariwise pixel distances, and (3) the square function for evaluating
# the difference between the feature distance ranking matrix and the pixel distance ranking matrix.
# Save the result in Test_2 folder.
fea_dist_method = 'Pearson'
image_dist_method = 'Manhattan'
error = 'squared'
norm_data = norm_data.iloc[:, :800]
result_dir = '../Results/Table_To_Image_Conve_Nsl_dos/Test_2'
os.makedirs(name=result_dir, exist_ok=True)
table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
               max_step, val_step, result_dir, error)