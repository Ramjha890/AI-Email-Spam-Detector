
import streamlit as st
import time
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

# Ensure required NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
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

# Load model and vectorizer
@st.cache_resource
def load_models():
    try:
        with open('vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return tfidf, model
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run train.py first.")
        st.info("Run: python train.py")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

# Load models
tfidf, model = load_models()

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Email/SMS Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Main content
st.title("📧 Email/SMS Spam Classifier")
st.markdown("Enter a message below to check if it's spam or not.")

input_sms = st.text_area("✏️ Enter your message here:", height=150, 
                         placeholder="Type or paste your email/SMS message...")

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_button = st.button("🔍 Predict", use_container_width=True)

if predict_button:
    if not input_sms or input_sms.strip() == "":
        st.warning("⚠️ Please enter a message to predict.")
    elif tfidf is None or model is None:
        st.error("❌ Model not loaded. Please run train.py first.")
    else:
        with st.spinner('🔄 Analyzing message...'):
            time.sleep(1)  # Simulate processing
            
            # Preprocess input text
            transformed_sms = transform_text(input_sms)
            
            try:
                # Vectorize input text
                vector_input = tfidf.transform([transformed_sms])
                # Predict using the model
                result = model.predict(vector_input)[0]
                
                # Get probability
                proba = model.predict_proba(vector_input)[0]
                spam_prob = proba[1] if len(proba) > 1 else 0.5
                
                # Display result with styling
                col_result, col_prob = st.columns([1, 1])
                
                if result == 1:
                    with col_result:
                        st.error("🚨 **SPAM**")
                        st.markdown("This message is classified as **SPAM**.")
                else:
                    with col_result:
                        st.success("✅ **NOT SPAM**")
                        st.markdown("This message is classified as **NOT SPAM**.")
                
                with col_prob:
                    st.info("📊 Confidence Scores")
                    st.write(f"Spam probability: {spam_prob:.2%}")
                    st.write(f"Ham probability: {1-spam_prob:.2%}")
                    
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
                st.info("Please make sure the model files are properly trained.")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About by R k Jha")
    st.write("This app uses Machine Learning to classify messages as Spam or Ham.")
    st.write("📊 **Features:**")
    st.write("- TF-IDF Vectorization")
    st.write("- Multinomial Naive Bayes")
    st.write("- Real-time prediction")
    st.write("- Confidence scores")
    
    st.divider()
    st.write("💡 **Tip:** Paste the complete message for better accuracy.")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
 st.markdown("""
            <div style="text-align:center; margin-top:20px;">
    <p>⚡ made by <b>R k Jha</b> ⚡</p>
<div style="
    display:flex;
    justify-content:center;
    align-items:center;
    gap:20px;
">
  
<a href="https://github.com/Ramjha890" target="_blank">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="40">
</a>

<a href="https://www.linkedin.com/in/r-k-jha-33313440b/" target="_blank">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="40">
</a>

<a href="https://tranquil-lolly-9dfe09.netlify.app/" target="_blank">
    <img src="https://img.icons8.com/ios-filled/50/globe.png" width="40">
</a>

</div>
</div>
""", unsafe_allow_html=True)

# Custom CSS for better mobile responsiveness
st.markdown("""
<style>
    .stTextArea textarea {
        min-height: 150px;
    }
    @media (max-width: 600px) {
        .stTextArea textarea {
            min-height: 100px !important;
        }
    }
    .main > div {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)
