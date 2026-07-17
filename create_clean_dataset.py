# create_clean_dataset.py
import pandas as pd

# Create a clean dataset with proper spam/ham messages
data = {
    'label': [
        'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam',
        'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam'
    ],
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
        "WINNER! You have won $1000. Claim now!",
        "Can we reschedule our meeting?",
        "Dear Sir, You have won a lottery of 1 million dollars.",
        "I'll be there in 10 minutes",
        "Claim your free gift now! Limited time only.",
        "How was your weekend?",
        "Congratulations! You are our lucky winner. Click here.",
        "Please review the attached document",
        "URGENT: Your account will be suspended. Click to reactivate.",
        "Let's catch up over coffee",
        "FREE offer just for you! Don't miss out."
    ]
}

df = pd.DataFrame(data)
df.to_csv('spam.csv', index=False)
print("✅ Clean dataset created as spam.csv")
print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['label'].value_counts()}")