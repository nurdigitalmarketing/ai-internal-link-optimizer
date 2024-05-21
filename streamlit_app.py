import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from openai import OpenAI

# Funzione per estrarre gli URL dalla sitemap
def extract_sitemap_urls(sitemap_url):
    response = requests.get(sitemap_url)
    soup = BeautifulSoup(response.content, 'xml')
    urls = []
    for loc in soup.find_all('loc'):
        urls.append(loc.text)
    return urls

# Funzione per eseguire lo scraping del contenuto principale di una pagina web
def scrape_webpage(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = " ".join([para.get_text() for para in paragraphs])
    return text

# Funzione per eseguire il clustering K-Means sui testi
def kmeans_clustering(texts, n_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X)
    return model.labels_, model

# Funzione per riassumere i testi utilizzando GPT-3.5
def summarize_texts(texts, api_key):
    client = OpenAI(api_key=api_key)
    summarized_texts = []
    
    for text in texts:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an assistant that helps summarize text."},
                {"role": "user", "content": f"Summarize the following text: {text}"}
            ],
            model="gpt-3.5-turbo",
            max_tokens=300
        )
        summarized_texts.append(response['choices'][0]['message']['content'])
    
    return summarized_texts

# Funzione per generare i link interni usando GPT-4
def generate_internal_links(target_text, related_texts, api_key):
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    List of Blog Posts:
    {related_texts}

    Target Blog Post:
    {target_text}

    Your task is to find internal linking opportunities in the target blog post using the list of blog posts. Please provide the improved post with internal links.
    """

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an assistant that helps with SEO by finding internal linking opportunities in blog posts."},
            {"role": "user", "content": prompt}
        ],
        model="gpt-4"
    )
    
    return response['choices'][0]['message']['content']

# Streamlit app
st.title("Internal Linking Automation Tool")

sitemap_url = st.text_input("Enter the Sitemap URL")
blog_prefix = st.text_input("Enter the Blog Prefix (e.g., '/blog/')")
target_url = st.text_input("Enter the Target Blog Post URL")
openai_api_key = st.text_input("Enter OpenAI API Key", type="password")

if st.button("Run"):
    if sitemap_url and blog_prefix and target_url and openai_api_key:
        with st.spinner("Processing..."):
            # Step 1: Extract URLs from the sitemap
            urls = extract_sitemap_urls(sitemap_url)
            urls = [url for url in urls if blog_prefix in url]
            
            # Step 2: Scrape content for each URL
            posts = [scrape_webpage(url) for url in urls]
            
            # Step 3: Include the target post in the list if not present
            if target_url not in urls:
                urls.append(target_url)
                posts.append(scrape_webpage(target_url))
            
            # Step 4: Perform K-Means clustering
            labels, model = kmeans_clustering(posts)
            
            # Step 5: Filter for the cluster containing the target post
            target_index = urls.index(target_url)
            target_cluster = labels[target_index]
            related_posts = [posts[i] for i in range(len(posts)) if labels[i] == target_cluster]
            
            # Step 6: Summarize the related posts
            summarized_related_posts = summarize_texts(related_posts, openai_api_key)
            related_texts = "\n\n".join(summarized_related_posts)
            
            # Step 7: Use GPT-4 to generate internal links
            target_text = posts[target_index]
            improved_post = generate_internal_links(target_text, related_texts, openai_api_key)
            
            st.subheader("Original Post")
            st.write(target_text)
            
            st.subheader("Improved Post with Internal Links")
            st.write(improved_post)
    else:
        st.error("Please fill in all fields")
