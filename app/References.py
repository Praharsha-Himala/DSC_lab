import streamlit as st

def app():
    st.header('References used to develop Phishing URL detection tool')
    st.markdown("""
    - Thanks to deepeshdm for developing useful features extraction techniques to determine phishing behavior
        - [Feature extraction code available here!](https://github.com/deepeshdm/Phishing-Attack-Domain-Detection/blob/main/Colab%20Notebooks/Data_Collection_and_Feature_Extraction_(Phishing_urls).ipynb)""")
    st.markdown("""
    - We have used following datasets to train our machine learning model
        - [PhiUSIIL dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
        - [Kaggle phishing sites url dataset](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls)""")
    st.markdown(""" 
    - Repository of code files and model are devleoped by Himala Praharsha Chittathuru
    
        - [Github repository for this tool](https://github.com/Praharsha-Himala/Phishing_URL_detection)
    """)