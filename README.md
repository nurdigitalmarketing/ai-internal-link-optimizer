# Internal Linking Automation Tool

## Descrizione

Questo strumento avanzato di ottimizzazione dei link interni è stato sviluppato per aiutarti a migliorare la SEO del tuo sito web automatizzando il processo di creazione di link interni rilevanti tra i tuoi articoli di blog. Utilizzando tecnologie all'avanguardia come i modelli di linguaggio OpenAI, lo strumento estrae, analizza e inserisce link interni nei tuoi post in modo naturale e fluido.

## Funzionalità

1. **Scraping della Sitemap**: Recupera tutte le pagine della tua sitemap, inclusi titoli e descrizioni.
2. **Clustering Semantico**: Utilizza tecniche di clustering per trovare le pagine più rilevanti per il tuo post target.
3. **Ottimizzazione dei Link Interni**: Usa un modello di linguaggio per analizzare il post target e le pagine rilevanti, inserendo link interni dove appropriato.

## Requisiti

- Python 3.8+
- Le seguenti librerie Python:

```plaintext
streamlit==1.15.0
requests==2.28.1
beautifulsoup4==4.11.1
sentence-transformers==2.2.2
openai==0.27.0
pandas==1.3.5
scikit-learn==1.0.2
```

## Installazione

1. Clona il repository:

    ```sh
    git clone https://github.com/tuo-username/tuo-repo.git
    cd tuo-repo
    ```

2. Crea un ambiente virtuale e attivalo:

    ```sh
    python -m venv venv
    source venv/bin/activate  # Su Windows usa `venv\Scripts\activate`
    ```

3. Installa le dipendenze:

    ```sh
    pip install -r requirements.txt
    ```

## Utilizzo

1. Esegui il file `app.py` con Streamlit:

    ```sh
    streamlit run app.py
    ```

2. Segui le istruzioni nell'interfaccia per:
    - Selezionare la lingua
    - Inserire la tua API Key di OpenAI
    - Selezionare il modello di OpenAI
    - Impostare la temperatura
    - Inserire l'URL della Sitemap
    - Inserire il prefisso dei Blog (opzionale)
    - Inserire l'URL del post target

3. Clicca su "Ottimizza Link Interni" per avviare il processo.

## Contatti

Creato da **NUR© Digital Marketing**. Per ulteriori informazioni, visita il nostro [sito web](https://www.nur.it).

