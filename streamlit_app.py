import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

st.title('Automazione dei Link Interni')

api_key = st.text_input("Inserisci la tua OpenAI API Key", type="password")
if api_key:
    client = OpenAI(api_key=api_key)

language = st.selectbox('Seleziona la Lingua', ['Italiano', 'Inglese', 'Francese', 'Spagnolo', 'Tedesco'])

sitemap_url = st.text_input('Inserisci la URL della Sitemap')
page_url = st.text_input('Inserisci la URL della Pagina da Ottimizzare')
keywords = st.text_area('Inserisci le Keyword Target (opzionale)')

def fetch_sitemap(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'lxml')
    links = [loc.text for loc in soup.find_all('loc')]
    return links

def fetch_page_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text(), [a['href'] for a in soup.find_all('a', href=True)]

def extract_keywords_from_page(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Extract relevant keywords from the following text in {language}."},
            {"role": "user", "content": text}
        ]
    )
    keywords = response.choices[0].message.content.strip().split(',')
    return [kw.strip() for kw in keywords]

def cluster_pages(pages):
    if len(pages) < 5:
        return None, None  # Skip clustering if not enough samples
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(pages)
    model = KMeans(n_clusters=5, random_state=42)
    model.fit(X)
    return model, vectorizer

def find_link_opportunities(page_links, target_keywords, page_text):
    opportunities = []
    for link in page_links:
        for keyword in target_keywords:
            if keyword.lower() in link.lower() or keyword.lower() in page_text.lower():
                opportunities.append((link, keyword))
    return opportunities

if st.button('Esegui'):
    if api_key and sitemap_url and page_url:
        st.write("Sto elaborando la tua richiesta...")
        sitemap_links = fetch_sitemap(sitemap_url)
        page_text, page_links = fetch_page_content(page_url)
        target_keywords = keywords.split('\n') if keywords else extract_keywords_from_page(page_text)
        if len(sitemap_links) >= 5:
            clustered_model, vectorizer = cluster_pages([page_text])
        else:
            clustered_model, vectorizer = None, None
        link_opportunities = find_link_opportunities(page_links, target_keywords, page_text)
        st.write("Opportunità di link trovate:")
        for link, keyword in link_opportunities:
            st.write(f"Link: {link}, Keyword: {keyword}")
    else:
        st.error("Per favore, inserisci la tua API Key, la URL della Sitemap e della Pagina da Ottimizzare")
