import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Data aur Vectorizer load karein
print("Loading data and vectorizer...")
df = pd.read_csv('data/processed_data.csv')
df.dropna(subset=['text'], inplace=True)

# Vectorizer load karein jo Day 4 mein banaya tha
tfidf = pickle.load(open('models/vectorizer.pkl', 'rb'))

# 2. Text ko Numbers mein badlein
X = tfidf.transform(df['text']) # Transform use karein (fit nahi)
Y = df['label']

# 3. Train-Test Split (The Exam Setup)
# 80% data se model seekhega (Train)
# 20% data se hum test karenge (Test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 4. Model Initialize aur Train karein
print("Training the Model (Brain)...")
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, Y_train)

# 5. Model ka Exam (Testing)
Y_pred = model.predict(X_test)
score = accuracy_score(Y_test, Y_pred)

print(f"✅ Day 5 Complete!")
print(f"🎯 Model Accuracy: {round(score*100, 2)}%")

# 6. Save the Model
pickle.dump(model, open('models/model.pkl', 'wb'))
print("Model saved in 'models/model.pkl'")

# 7. Optional: Confusion Matrix (Report ke liye best hai)
cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()