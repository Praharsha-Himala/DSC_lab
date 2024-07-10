import streamlit as st
import pickle
from urllib.parse import urlparse
import re
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

def predict(url):
    data = pd.DataFrame([url], columns=["URL"])
    dataframe = FeatureExtractor(data)
    test = dataframe.Count_feature_extractor(data)
    test = test.drop('URL', axis=1)
    scaler = pickle.load(open(r"C:\Users\HARSHU\PycharmProjects\DSC_lab\scaler.pkl", 'rb'))
    test_scaled = scaler.transform(test)
    model = pickle.load(open(r'C:\Users\HARSHU\PycharmProjects\DSC_lab\RandomForest_model.pkl', 'rb'))
    prediction = model.predict(test_scaled)
    return prediction

def app():
    st.title("Know your URLs")
    st.subheader("Scared of opening unknown URLs?")
    st.subheader(" Not anymore, use our phishing URL detection tool to know if the website is safe to proceed!\n", divider=True)

    user_input = st.text_input("Enter URL")

    if user_input:
        user_input = user_input.strip()
        if user_input:
            if st.button("Predict"):
                with st.spinner("Predicting..."):
                    prediction = predict(user_input)
                    if prediction[0] == 0:
                        st.error("🔴 Given URL might lead to a phishing website!")
                    else:
                        st.success("🟢 Given URL might be safe to use")
        else:
            st.warning("Please enter a valid URL.")

    st.markdown("**Disclaimer**: We try to provide an estimate of safety of URL's, using machine learning. While aiming for higher accuracy, no prediction can be completely accurate. We are working on improving for better results.")
    st.subheader("", divider=True)
    st.markdown("""
    :bulb: Tips
    - Copy the URL of the website you want to check 
    - Paste the valid URL into the text box above
    - Click enter and then predict button to analyze
    """)
    st.caption("Removing '/' at the end of the URL while pasting will help in better results!")