import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv(r'C:\Users\HARSHU\PycharmProjects\DSC_lab\PhiUSIIL_Phishing_URL_Dataset.csv')
print(data.head())

data.isna().sum()

print(data['CharContinuationRate'])