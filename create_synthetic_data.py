# create_synthetic_data.py
import pandas as pd

# Create sample data
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
print("✅ Synthetic dataset created as spam.csv")
print("Now run: python train.py")