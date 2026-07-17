# download_data.py
import urllib.request
import pandas as pd

# Download the dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
urllib.request.urlretrieve(url, "spam.csv")

print("✅ Dataset downloaded successfully!")
print("Now run: python train.py")