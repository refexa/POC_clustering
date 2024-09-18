# Document Clustering and Topic Modeling with Word Clouds

This project provides an interactive web-based tool using Streamlit for clustering multiple PDF documents and visualizing the results through word clouds. The application allows users to upload PDF files, perform clustering and topic modeling on their content, and visualize the most relevant words from each cluster as word clouds.

## Features

- **PDF Upload**: Upload multiple PDF files for analysis.
- **Document Clustering**: Cluster documents based on the similarity of their content using KMeans clustering.
- **Topic Modeling**: Generate topics for each cluster using Latent Dirichlet Allocation (LDA) and display them in a readable format.
- **Word Clouds**: Visualize the most common words from each cluster as word clouds.
- **Interactive UI**: Customize the number of clusters using a slider and view the clusters and word clouds in a clean, user-friendly interface.

## Installation

**Clone the repository:**
`git clone https://github.com/Qammarbhat/POC_clustering.git`

**Navigate to the project directory:**
`cd document-clustering-wordcloud`

**Install the required dependencies:**
`pip install -r requirements.txt`

**Run the Streamlit app:**
`streamlit run main.py`

## Usage

- **Upload PDFs**: Upload one or more PDF files through the file uploader.
- **Set Number of Clusters**: Adjust the slider to select the number of clusters.
- **View Clusters**: The app will display the topics and word clouds for each cluster.
- **Interpret Results**: Use the topics and word clouds to understand the content distribution across the clusters.

## File Overview

### `main.py`

This file defines the Streamlit interface and handles user interactions, such as file uploading, clustering, and displaying results (topics and word clouds).

- **Key Functions**:
    - `create_wordcloud(tfidf_matrix, clusters, vectorizer, n_clusters)`: Generates word clouds for each cluster based on the TF-IDF matrix and displays them using matplotlib.
    - `st.file_uploader`: Allows users to upload multiple PDF files.
    - `st.slider`: Allows users to select the number of clusters.
    - `st.markdown`: Styles the application title using custom HTML and CSS.

### `utils.py`

Contains utility functions for processing PDF files, performing clustering, and extracting topics from clusters.

- **Key Functions**:
    - `read_pdf(file)`: Reads the content of a PDF file.
    - `split_document(doc_content)`: Splits the document into sentences/paragraphs for clustering.
    - `topic_modeling(tfidf_matrix, feature_names, n_topics)`: Applies LDA to extract topics from a cluster.
    - `classify_and_cluster(doc_content_list, n_clusters)`: Preprocesses document content, clusters it using KMeans, and applies topic modeling to the clusters.
    - `remove_duplicate_topics(topics, feature_names)`: Ensures that topics across clusters are unique by removing duplicates based on cosine similarity.

### `requirements.txt`

The required dependencies for running the project.

- `streamlit`: For building the web-based UI.
- `nltk`: For text processing and tokenization.
- `pandas`: For data manipulation.
- `scikit-learn`: For clustering (KMeans) and topic modeling (LDA).
- `PyPDF2`: For reading PDF files.
- `plotly`: For visualizing clusters (optional, but included for future enhancements).
- `wordcloud`: For generating word clouds for each cluster.
