import streamlit as st
import pickle
from urllib.parse import urlparse
import re
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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
                digits += 1
        return digits

    @staticmethod
    def letter_count(url):
        letters = 0
        for i in url:
            if i.isalpha():
                letters += 1
        return letters

    @staticmethod
    def no_of_dir(url):
        urldir = urlparse(url).path
        return urldir.count('/')

    @staticmethod
    def having_ip_address(url):
        match = re.search(
            '(([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.'
            '([01]?\d\d?|2[0-4]\d|25[0-5])\/)|'
            '((0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\/)'
            '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)
        if match:
            return -1
        else:
            return 1

    @staticmethod
    def shortening_service(url):
        match = re.search(
            'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
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
def improve(url, label):
    # Load existing data and prepare for new data addition
    data = pd.read_csv(r'C:\Users\HARSHU\PycharmProjects\DSC_lab\combined_data.csv')
    data = data[['URL', 'label']]
    data['URL'] = data['URL'].apply(lambda x: str(x) if isinstance(x, float) else x)
    nan_count = data['label'].isna().sum()
    print('Nan values in combined dataset:', nan_count)
    data = data.dropna(subset=['label'])

    # Add new data
    new_data = pd.DataFrame({
        'URL': [url],
        'label': [label]
    })
    data = pd.concat([data, new_data], ignore_index=True)

    # Feature extraction and scaling
    dataframe = FeatureExtractor(data)
    test = dataframe.Count_feature_extractor(data)
    X = test.drop(['URL', 'label'], axis=1)
    y = test['label']

    # Load existing scaler and transform data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Load existing model
    model = RandomForestClassifier()
    model.fit(X_scaled, y)

    # Save updated model and scaler
    with open("C:/Users/HARSHU/PycharmProjects/DSC_lab/RandomForest_model.pkl", 'wb') as model_file:
        pickle.dump(model, model_file)

    with open("C:/Users/HARSHU/PycharmProjects/DSC_lab/scaler.pkl", 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
def app():
    st.title("Help us to improve the performance")
    st.markdown("""
    If you think the model has made mistakes in predicting your URL behavior, kindly share the URL and its label
    - Select the appropriate label (Legitimate or Unsafe) of URL based on your experience
    - Do NOT use this page if you are not sure about the label or the true nature of the URL
    """)
    user_input = st.text_input("Enter URL")
    if user_input:
        user_input = user_input.strip()
        if user_input:
            if st.button("Legitimate"):
                with st.spinner("Please Do NOT close this window"):
                    improve(user_input, 1.0)
                    st.write("Thanks for helping us to improve!")
                st.empty()
            if st.button("Unsafe"):
                with st.spinner("Please Do Not close this window"):
                    improve(user_input, 0.0)
                    st.write("Thanks for helping us to improve!")
                st.empty()
        else:
            st.warning("Please enter a valid URL.")