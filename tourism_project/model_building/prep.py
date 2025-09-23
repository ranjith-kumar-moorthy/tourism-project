# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
from google.colab import userdata

# Define constants for the dataset and output paths

# Access the token from Colab secrets
HF_TOKEN = userdata.get('HF_TOKEN')
api = HfApi(token=HF_TOKEN)

# Access the token from Colab secrets and set it as an environment variable
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')

# print(f"After Token: {HF_TOKEN}")

# please create your dataset as you create your space
DATASET_PATH = "hf://datasets/ranjithkumarsundaramoorthy/tourism-project/tourism.csv"
# DATASET_PATH = "tourism_project/data/tourism.csv" # Read from local file
tourism_csv = pd.read_csv(DATASET_PATH)

df = tourism_csv
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop(columns=['CustomerID'], inplace=True)

# Encoding the categorical 'ProductPitched' column
label_encoder = LabelEncoder()
df['ProductPitched'] = label_encoder.fit_transform(df['ProductPitched'])
df['Passport'] = label_encoder.fit_transform(df['Passport'])
df['OwnCar'] = label_encoder.fit_transform(df['OwnCar'])
df['PitchSatisfactionScore'] = label_encoder.fit_transform(df['PitchSatisfactionScore'])
df['NumberOfChildrenVisiting'] = label_encoder.fit_transform(df['NumberOfChildrenVisiting'])
df['NumberOfPersonVisiting'] = label_encoder.fit_transform(df['NumberOfPersonVisiting'])
df['NumberOfFollowups'] = label_encoder.fit_transform(df['NumberOfFollowups'])
df['MaritalStatus'] = label_encoder.fit_transform(df['MaritalStatus'])
df['PreferredPropertyStar'] = label_encoder.fit_transform(df['PreferredPropertyStar'])
df['CityTier'] = label_encoder.fit_transform(df['CityTier'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])

target_col = 'ProdTaken' # Corrected target column name

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save the files to the model_building directory
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename

        repo_id="ranjithkumarsundaramoorthy/tourism-project",

        repo_type="dataset",
    )
