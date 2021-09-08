import re
from bs4 import BeautifulSoup

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_special_characters(text)
    return text

def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9ğüşöçıİĞÜŞÖÇ\s]'
    text=re.sub(pattern,'',text)
    return text

def get_csv(filename):
    df = pd.read_csv(filename, names=["sentence", "target"])
    df = df.iloc[1:,:]

