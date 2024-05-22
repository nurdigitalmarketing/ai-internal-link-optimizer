# File: app.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import openai
import pandas as pd
from sklearn.cluster import KMeans
import json
import logging

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Funzione per scraping della sitemap con caching
@st.cache_data
def scrape_sitemap(sitemap_url, language):
    try:
        headers = {"Accept-Language": language}
        response = requests.get(sitemap_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'xml')
        urls = [loc.text for loc in soup.find_all('loc')]
        pages = []

        for url in urls:
            page_response = requests.get(url, headers=headers)
            page_response.raise_for_status()
            page_soup = BeautifulSoup(page_response.content, 'html.parser')
            title = page_soup.title.string if page_soup.title else 'No title'
            meta_desc = page_soup.find('meta', attrs={'name': 'description'})
            meta_desc = meta_desc['content'] if meta_desc else 'No description'
            pages.append({'url': url, 'title': title, 'description': meta_desc})
        
        return pages
    except requests.RequestException as e:
        logging.error(f"Errore durante lo scraping della sitemap: {e}")
        st.error("Errore durante lo scraping della sitemap. Controlla l'URL e riprova.")
        return []

# Funzione per scraping del contenuto della pagina con caching
@st.cache_data
def scrape_page_content(url, language):
    try:
        headers = {"Accept-Language": language}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        logging.error(f"Errore durante lo scraping del contenuto della pagina: {e}")
        st.error("Errore durante lo scraping del contenuto della pagina. Controlla l'URL e riprova.")
        return ""

# Funzione per clustering semantico con caching
@st.cache_data
def semantic_clustering(pages, target_post, model_name='sentence-transformers/all-MiniLM-L6-v2', n_clusters=5):
    try:
        model = SentenceTransformer(model_name)
        embeddings = [model.encode(page['title'] + ' ' + page['description'], convert_to_tensor=True) for page in pages]
        target_embedding = model.encode(target_post, convert_to_tensor=True)
        similarities = [util.pytorch_cos_sim(target_embedding, emb).item() for emb in embeddings]

        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        clusters = kmeans.fit_predict(embeddings)

        df = pd.DataFrame(pages)
        df['cluster'] = clusters
        target_cluster = kmeans.predict([target_embedding])[0]
        relevant_pages = df[df['cluster'] == target_cluster]

        return relevant_pages.to_dict('records')
    except Exception as e:
        logging.error(f"Errore durante il clustering semantico: {e}")
        st.error("Errore durante il clustering semantico.")
        return []

# Funzione per ottimizzazione dei link interni
def optimize_internal_links(relevant_pages, target_post, openai_api_key, model, temperature):
    try:
        openai.api_key = openai_api_key
        pages_text = "\n".join([f"{page['title']}\n{page['description']}\n{page['url']}" for page in relevant_pages])
        prompt = f"""
        List of Blog Posts:
        {pages_text}

        Target Blog Post:
        {target_post}

        You are given a list of blog posts that contains the url, title, and description for each post, you are also provided with the extracted contents of a 'target' blog post. Your task is to find internal linking opportunities in the target blog post using the list of blog posts. While reading the target blog post You must try to find natural ways to inject relevant internal links throughout the target blog post content.

        For example if the post is talking about topic X in one of the sections and you notice that there is a post in the provided list of blog posts that is relevant to it, you can add something like:
        <a href="https://example.com/blog/x">you can learn more about X here</a>
        try to maintain the same tone while making these minor changes.

        The links should be spaced out and naturally injected where it is best suited without feeling forced, here is a bad and good example of what I mean.

        Bad (too many links next to each other):

        I've written about cheese <a href="https://example.com/page1">so</a> <a href="https://example.com/page2">many</a> <a href="https://example.com/page3">times</a> <a href="https://example.com/page4">this</a> <a href="https://example.com/page5">year</a>.

        Better (links are spaced out with context):

        I've written about cheese so many times this year: who can forget the <a href="https://example.com/blue-cheese-vs-gorgonzola">controversy over blue cheese and gorgonzola</a>, the <a href="https://example.com/worlds-oldest-brie">world's oldest brie</a> piece that won the Cheesiest Research Medal, the epic retelling of <a href="https://example.com/the-lost-cheese">The Lost Cheese</a>, and my personal favorite, <a href="https://example.com/boy-and-his-cheese">A Boy and His Cheese: a story of two unlikely friends</a>.

        Of course this is just an example of how you would naturally inject natural link, the blog post has nothing to do with cheese.

        Don't force links where they are not relevant, if you can't find any relevant links to inject just respond with saying so. Otherwise your response should be the target post with the internal links injected. Please maintain the exact same body of text but you can change a wording a bit in the sections you want to fit an internal link for a more natural read.
        """
        
        system_message = "You are an SEO consultant that specializes in internal linking. Your task will be to naturally inject internal links to a given blog post."

        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=1500,
            temperature=temperature
        )

        optimized_post = response.choices[0].text.strip()
        return optimized_post
    except Exception as e:
        logging.error(f"Errore durante l'ottimizzazione dei link interni: {e}")
        st.error("Errore durante l'ottimizzazione dei link interni.")
        return ""

# Interfaccia Streamlit
def main():

    # Crea una riga con 3 colonne
col1, col2 = st.columns([1, 7])

# Colonna per l'immagine (a sinistra)
    with col1:
        # Assicurati di avere un'immagine nel percorso specificato o passa un URL diretto
        st.image("https://raw.githubusercontent.com/nurdigitalmarketing/previsione-del-traffico-futuro/9cdbf5d19d9132129474936c137bc8de1a67bd35/Nur-simbolo-1080x1080.png", width=80)

# Colonna per il titolo e il testo "by NUR® Digital Marketing" (al centro)
    with col2:
        st.title('Internal Linking Automation Tool')
        st.markdown('###### by [NUR® Digital Marketing](https://www.nur.it)')

    with st.expander("Istruzioni"):
        st.markdown("""
        1. **Seleziona la lingua**: Scegli la lingua desiderata per il scraping e l'analisi.
        2. **Inserisci la tua API Key di OpenAI**: Fornisci la tua chiave API in modo sicuro.
        3. **Seleziona il modello di OpenAI**: Scegli il modello GPT desiderato tra quelli disponibili.
        4. **Seleziona la temperatura**: Imposta la temperatura per il comportamento del modello.
        5. **Inserisci l'URL della Sitemap**: Fornisci l'URL della sitemap del sito web da analizzare.
        6. **Inserisci il prefisso dei Blog**: Fornisci il prefisso dei blog per filtrare gli URL (opzionale).
        7. **Inserisci l'URL del post target**: Fornisci l'URL del post target per l'ottimizzazione dei link interni.
        8. **Ottimizza Link Interni**: Clicca sul pulsante per avviare il processo di ottimizzazione.
        """)

    # Selezione della lingua
    language = st.selectbox("Seleziona la lingua", ["en", "it", "fr", "de", "es", "pt"])
    
    # Inserimento della chiave API di OpenAI
    openai_api_key = st.text_input("Inserisci la tua API Key di OpenAI", type="password")
    
    # Selezione del modello di OpenAI
    model = st.selectbox("Seleziona il modello di OpenAI", [
        "gpt-3.5-turbo-16k", 
        "gpt-3.5-turbo", 
        "gpt-3.5-turbo-1106", 
        "gpt-4-turbo", 
        "gpt-4"
    ])
    
    # Selezione della temperatura
    temperature = st.slider("Seleziona la temperatura", 0.0, 1.0, 0.2)
    
    # Input URL
    sitemap_url = st.text_input("Inserisci l'URL della Sitemap")
    blog_prefix = st.text_input("Inserisci il prefisso dei Blog")
    target_post_url = st.text_input("Inserisci l'URL del post target")
    
    if st.button("Ottimizza Link Interni"):
        with st.spinner("Scraping della Sitemap..."):
            pages = scrape_sitemap(sitemap_url, language)
        
        target_post = scrape_page_content(target_post_url, language)
        
        with st.spinner("Clustering Semantico..."):
            relevant_pages = semantic_clustering(pages, target_post)
        
        with st.spinner("Ottimizzazione dei Link Interni..."):
            optimized_post = optimize_internal_links(relevant_pages, target_post, openai_api_key, model, temperature)
        
        st.subheader("Post Ottimizzato")
        st.write(optimized_post)
        st.subheader("Post Originale")
        st.write(target_post)

if __name__ == "__main__":
    main()
