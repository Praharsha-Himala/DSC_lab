import pandas as pd


data = pd.read_csv(r"C:\Users\HARSHU\PycharmProjects\DSC_lab\PhiUSIIL_Phishing_URL_Dataset.csv")
print(data.head())
data1 = data[['URL', 'label']]
print(data1.head())

data_ = pd.read_csv(r"C:\Users\HARSHU\PycharmProjects\DSC_lab\dataset_phishing.csv")
print(data_.head())

label_mapping = {
    'legitimate': 1,
    'phishing': 0
}

# Replace values in 'label' column using replace()
data_['status'] = data_['status'].replace(label_mapping)
print(type(data_['url'].iloc[0]))

data2 = data_
print(data2.head())

data__ = pd.read_csv(r'C:\Users\HARSHU\PycharmProjects\DSC_lab\phishing_site_urls.csv')
data__['Label'] = data__['Label'].map({'good': 1, 'bad': 0})
print(data__.head())
print(type(data__['URL'].iloc[0]))
data3 = data__

combined_df = pd.concat([data1, data2, data3], ignore_index=True)
combined_df.to_csv(r'C:\Users\HARSHU\PycharmProjects\DSC_lab\combined_data.csv', index=False)