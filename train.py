# train.py
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

def transform_text(text):
    # Handle NaN or None values
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Load dataset
print("Loading dataset...")

# Try different ways to load the dataset
try:
    # Try loading with proper CSV format
    df = pd.read_csv('spam.csv')
    print(f"Dataset loaded successfully! Shape: {df.shape}")
except:
    try:
        # Try loading as tab-separated
        df = pd.read_csv('spam.csv', sep='\t')
        print(f"Dataset loaded with tab separator! Shape: {df.shape}")
    except:
        # Create dataset if it doesn't exist
        print("Creating dataset from scratch...")
        data = {
            'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam'],
            'message': [
                "Hey, how are you doing?",
                "Congratulations! You won a free prize! Click here to claim.",
                "Meeting at 3pm tomorrow",
                "URGENT: Your account has been compromised. Click to verify.",
                "Don't forget to pick up groceries",
                "FREE money! Limited time offer. Call now!",
                "See you at the party tonight",
                "You've been selected for a special offer. Click here.",
                "Thanks for your help yesterday",
                "WINNER! You have won $1000. Claim now!"
            ]
        }
        df = pd.DataFrame(data)
        df.to_csv('spam.csv', index=False)
        print("Dataset created and saved!")

print(f"Columns: {df.columns.tolist()}")
print(f"Sample data:\n{df.head()}")

# Clean the dataset
# Remove any rows with NaN values
df = df.dropna()

# Check if we have the right columns
if 'label' not in df.columns or 'message' not in df.columns:
    print("Error: Dataset must have 'label' and 'message' columns")
    print(f"Available columns: {df.columns.tolist()}")
    # Try to rename columns if needed
    if len(df.columns) == 2:
        df.columns = ['label', 'message']
        print("Renamed columns to 'label' and 'message'")

# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1, 'spam': 1, 'ham': 0})
df['label'] = df['label'].fillna(0)  # Replace any NaN with 0

# Verify the data
print(f"Class distribution:\n{df['label'].value_counts()}")

# Transform messages
print("Preprocessing messages...")
df['transformed'] = df['message'].apply(transform_text)

# Remove any empty transformed messages
df = df[df['transformed'] != ""]

# Prepare features and labels
X = df['transformed']
y = df['label']

print(f"Total samples: {len(X)}")
print(f"Spam samples: {sum(y)}")
print(f"Ham samples: {len(y) - sum(y)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit TfidfVectorizer
print("Vectorizing text...")
tfidf = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf.fit_transform(X_train)

# Train model
print("Training model...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Test accuracy
X_test_tfidf = tfidf.transform(X_test)
accuracy = model.score(X_test_tfidf, y_test)
print(f"Model accuracy: {accuracy:.4f}")

# Save the fitted objects
print("Saving model files...")
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
    
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model and vectorizer saved successfully!")
print(f"✅ Files created: vectorizer.pkl and model.pkl")