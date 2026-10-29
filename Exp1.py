import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("student_data_preprocessed.csv")
print(df.head())
df.info()


print("Missing values")
print(df.isnull().sum())

age_impute = SimpleImputer(strategy='mean')
df[['Age']] = age_impute.fit_transform(df[['Age']])


categorical_columns = ["Department", "City"]

mode_imputer = SimpleImputer(strategy="most_frequent")
df[categorical_columns] = mode_imputer.fit_transform(df[categorical_columns])


label_encoder = LabelEncoder()
df["Gender"] = label_encoder.fit_transform(df["Gender"])

df = pd.get_dummies(
    df,
    columns=["Department"],
    dtype=int
)

print("\nPreprocessed Dataset")
print(df)

