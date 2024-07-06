# url: https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls

from urllib.parse import urlparse
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

    # Define a list of characters or substrings to count
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
            # print match.group()
            return -1
        else:
            # print 'No matching pattern found'
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


data = pd.read_csv(r'C:\Users\HARSHU\PycharmProjects\DSC_lab\phishing_site_urls.csv')

print(data.head())
data['Label'] = data['Label'].map({'good': 0, 'bad': 1})
dataframe = FeatureExtractor(data)
test = dataframe.Count_feature_extractor(data)
FeatureExtractor.preview(test)

X = test.drop(['URL','Label'], axis=1)
y = test['Label']
print(X.info())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)


from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Create a dictionary of models, yes, you can implement it this way
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Initialize KFold cross-validation with k=5
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring='accuracy')
    results[name] = scores.mean()


best_model = max(results, key=results.get)
best_score = results[best_model]

best_model_instance = models[best_model]
best_model_instance.fit(X_train_scaled, y_train)
test_accuracy = best_model_instance.score(X_test_scaled, y_test)

# Print the results
print(f'Cross-validation results:')
for name, score in results.items():
    print(f'{name}: {score:.4f}')

model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_pred, y_test))
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm,
                     index=['Normal', 'Phishing'],
                     columns=['Normal', 'Phishing'])

plt.figure(figsize=(14, 10))
sns.set(font_scale=2)  # Set the font scale
sns.heatmap(cm_df / np.sum(cm_df, axis=0), annot=True, fmt='.2%', cmap='Blues')
plt.ylabel('Actual values')
plt.xlabel('Predicted values')
plt.show()

models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'KNN']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
performance = {
    'Logistic Regression': [0.72, 0.90, 0.77, 0.63],
    'Decision Tree': [0.90, 0.90, 0.90, 0.90],
    'Random Forest': [0.90, 0.86, 0.88, 0.87],
    'KNN': [0.87, 0.87, 0.87, 0.87]
}

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Set width of bars
bar_width = 0.2

# Set position of bar on X axis
r = np.arange(len(models))

# Plot bars for each metric
for i, metric in enumerate(metrics):
    bars = [performance[model][i] for model in models]
    plt.bar(r + i * bar_width, bars, bar_width, label=metric)

# Add xticks
plt.xlabel('Models', fontweight='bold')
plt.ylabel('Percentage', fontweight='bold')
plt.xticks(r + bar_width * (len(metrics) - 1) / 2, models)

# Create legend & Show graphic
plt.legend()
plt.show()
