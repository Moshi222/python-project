# -------------------------------------------- 
# AI Based Sentiment Analysis (Basic Project) 
# -------------------------------------------- 
# Import Libraries 
import pandas as pd 
import numpy as np 
import re 
import nltk 
from nltk.corpus import stopwords 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import accuracy_score 
# Download stopwords 
nltk.download('stopwords') 
 
# Load stopwords 
stop_words = set(stopwords.words('english')) 
 
# -------------------------------------------- 
# Step 1: Create Sample Dataset 
# -------------------------------------------- 
 
data = { 
    'text': [ 
        'I love this movie', 
        'This product is amazing', 
        'The service was terrible', 
        'I hate this phone', 
        'The food was very good', 
        'The experience was bad', 
        'This is the best thing', 
        'Not a good product', 
        'Very happy with the service', 
        'Worst purchase ever' 
    ], 
     
    'sentiment': [ 
        'positive', 
        'positive', 
        'negative', 
        'negative', 
        'positive', 
        'negative', 
        'positive', 
        'negative', 
        'positive', 
        'negative' 
    ] 
} 
 
df = pd.DataFrame(data) 
print("Dataset:") 
print(df) 
# -------------------------------------------- 
# Step 2: Text Cleaning 
# -------------------------------------------- 
def clean_text(text): 
text = re.sub('[^a-zA-Z]', ' ', text) 
text = text.lower() 
text = text.split() 
text = [word for word in text if word not in stop_words] 
return " ".join(text) 
df['clean_text'] = df['text'].apply(clean_text) 
# -------------------------------------------- 
# Step 3: Convert Text to Numbers 
# -------------------------------------------- 
cv = CountVectorizer() 
X = cv.fit_transform(df['clean_text']).toarray() 
y = df['sentiment'] 
# -------------------------------------------- 
# Step 4: Train Test Split 
# -------------------------------------------- 
X_train, X_test, y_train, y_test = train_test_split( 
X, y, test_size=0.3, random_state=42 
# -------------------------------------------- 
# Step 5: Train Model 
# -------------------------------------------- 
model = MultinomialNB() 
model.fit(X_train, y_train) 
# -------------------------------------------- 
# Step 6: Prediction 
# -------------------------------------------- 
y_pred = model.predict(X_test) 
) 
print("\nModel Accuracy:", accuracy_score(y_test, y_pred)) 
# -------------------------------------------- 
# Step 7: Test with New Sentence 
# -------------------------------------------- 
def predict_sentiment(sentence): 
sentence = clean_text(sentence) 
vector = cv.transform([sentence]).toarray() 
prediction = model.predict(vector) 
return prediction[0] 
print("\nPrediction Examples") 
print("I really like this product →", predict_sentiment("I really like this product")) 
print("This phone is very bad →", predict_sentiment("This phone is very bad")) 
print("The movie was average →", predict_sentiment("The movie was average")) 