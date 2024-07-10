import pandas as pd
from urllib.parse import urlparse
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

class FeatureExtractor:
    def __init__(self, dataframe):
        self.df = dataframe

    def Length_feature_extractor(self, dataframe):
        self.df = dataframe.copy()
        self.df['url_length'] = self.df['URL'].apply(lambda i: len(str(i)))
        self.df['hostname_length'] = self.df['URL'].apply(lambda i: len(urlparse(i).netloc))
        self.df['path_length'] = self.df['URL'].apply(lambda i: len(urlparse(i).path))
        return self.df

    @staticmethod
    def fd_length(url):
        urlpath = urlparse(url).path
        try:
            return len(urlpath.split('/')[1])
        except:
            return 0

    @staticmethod
    def count_occurrences(url, char):
        return url.count(char)

    @staticmethod
    def digit_count(url):
        digits = 0
        for i in url:
            if i.isnumeric():
                digits = digits + 1
        return digits

    @staticmethod
    def letter_count(url):
        letters = 0
        for i in url:
            if i.isalpha():
                letters = letters + 1
        return letters

    @staticmethod
    def no_of_dir(url):
        urldir = urlparse(url).path
        return urldir.count('/')

    @staticmethod
    def having_ip_address(url):
        match = re.search(
            '(([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.'
            '([01]?\d\d?|2[0-4]\d|25[0-5])\/)|'  # IPv4
            '((0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\/)'
            '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)
        if match:
            return -1
        else:
            return 1

    @staticmethod
    def shortening_service(url):
        match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                          'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                          'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                          'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                          'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                          'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                          'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                          'tr\.im|link\.zip\.net',
                          url)
        if match:
            return -1
        else:
            return 1

    def Count_feature_extractor(self, dataframe):
        self.Length_feature_extractor(dataframe)
        self.df['fd_length'] = self.df['URL'].apply(lambda i: self.fd_length(i))

        chars_to_count = ['-', '@', '?', '%', '.', '=', 'http', 'https', 'www']
        for char in chars_to_count:
            column_name = f'count_{char}'
            self.df[column_name] = self.df['URL'].apply(lambda url: self.count_occurrences(url, char))

        self.df['count-digits'] = self.df['URL'].apply(lambda i: self.digit_count(i))
        self.df['count-letters'] = self.df['URL'].apply(lambda i: self.letter_count(i))
        self.df['count_dir'] = self.df['URL'].apply(lambda i: self.no_of_dir(i))
        self.df['use_of_ip'] = self.df['URL'].apply(lambda i: self.having_ip_address(i))
        self.df['short_url'] = self.df['URL'].apply(lambda i: self.shortening_service(i))
        return self.df

    @staticmethod
    def preview(df):
        return df.head()


data = pd.read_csv(r'C:\Users\HARSHU\PycharmProjects\DSC_lab\combined_data.csv')

print(data.info())
data = data[['URL', 'label']]
data['URL'] = data['URL'].apply(lambda x: str(x) if isinstance(x, float) else x)


nan_count = data['label'].isna().sum()
print(f"Number of rows with NaN values in 'label' column: {nan_count}")

data = data.dropna(subset=['label'])

dataframe = FeatureExtractor(data)
test = dataframe.Count_feature_extractor(data)
FeatureExtractor.preview(test)

X = test.drop(['URL', 'label'], axis=1)
y = test['label']
print(X.info())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(classification_report(y_pred, y_test))
y_pred_classes = [1 if prob > 0.5 else 0 for prob in y_pred]  # Assuming binary classification with threshold 0.5

print(confusion_matrix(y_test, y_pred_classes))

# Save the model and scaler
pickle.dump(model, open("RandomForest_model.pkl", 'wb'))
pickle.dump(scaler, open("scaler.pkl", 'wb'))