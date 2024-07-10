import streamlit as st


def app():
    st.header("Phishing Websites Detection: Leveraging Machine Learning for Enhanced Security", divider=True)
    st.markdown("Phishing is a fraudulent practice that retrieves valuable information to steal money from online users. As the world evolves into an expanding digital era, the threat of phishing looms further. Most phishing websites retrieve credit card information, social media account passwords and personal data. Though having considerable knowledge of proper online behaviour, it is hard to detect phishing websites as the links conceal phishing features. In recent years, Machine learning methods to detect phishing websites have emerged, which can learn the hidden features of phishing URLs. This paper explores machine learning models such as logistic regression, decision trees, random forests, and K-nearest neighbours to classify phishing websites from regular websites. Phishing website URLs have certain features which can help predict phishing behaviour; we have generated these features from URLs and used them with machine learning models for classification. A Kaggle dataset of phishing and regular website URLs is used for training and testing machine learning models.")
    st.subheader("", divider=True)
    st.caption("_You can find more about the dataset, machine learning models used and code in References page_")