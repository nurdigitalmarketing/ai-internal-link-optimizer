
# Internal Linking Automation Tool

Internal Linking Optimization Tool is a web application built with Streamlit that allows you to scrape a website's sitemap, perform semantic clustering on the pages, and optimize internal links within a target blog post. This tool is ideal for enhancing the SEO of your website by automating the process of creating relevant internal links.

## Features

- Scrape the website's sitemap and retrieve all pages, including their title and meta description
- Perform semantic clustering on the pages to find the most relevant ones for the target blog post
- Pass the shortlisted relevant pages to an LLM (Language Model) along with the target blog post to inject internal links where relevant

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/internal-linking-optimization-tool.git
   cd internal-linking-optimization-tool
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

2. Open your web browser and go to `http://localhost:8501`.

3. Follow the instructions on the web interface:
   - Select the language for the scraping and analysis.
   - Enter your OpenAI API key.
   - Select the desired OpenAI model.
   - Set the temperature for the model's behavior.
   - Enter the URL of the sitemap.
   - Enter the blog prefix (optional).
   - Enter the URL of the target blog post.
   - Click on "Optimize Internal Links" to start the process.

## Dependencies

- [Streamlit](https://streamlit.io)
- [Requests](https://pypi.org/project/requests/)
- [BeautifulSoup4](https://pypi.org/project/beautifulsoup4/)
- [Sentence Transformers](https://pypi.org/project/sentence-transformers/)
- [OpenAI](https://pypi.org/project/openai/)
- [Pandas](https://pypi.org/project/pandas/)
- [Scikit-learn](https://pypi.org/project/scikit-learn/)

## Contributing

1. Fork the repository.
2. Create a new branch: `git checkout -b my-feature-branch`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin my-feature-branch`
5. Open a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgments

Created by **NUR© Digital Marketing**. For more information, visit our [website](https://www.nur.it).
