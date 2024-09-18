import re
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation as LDA

from PyPDF2 import PdfReader
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt_tab')

# Helper function to read PDF file
def read_pdf(file):
    """
    Reads the text content from a PDF file.

    Args:
        file (str or file-like object): Path to the PDF file or file-like object.

    Returns:
        str: The extracted text content from the PDF file.
    """
    reader = PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to split document content into meaningful sections
def split_document(doc_content):
    """
    Splits the document content into paragraphs and sentences.

    Args:
        doc_content (str): The text content of the document.

    Returns:
        list: A list of sentences derived from the document content, with empty or short sentences removed.
    """
    # Split by multiple newlines or multiple spaces (for paragraphs)
    paragraphs = re.split(r'\n{2,}|\s{2,}', doc_content)

    # Tokenize each paragraph into sentences
    sentences = []
    for paragraph in paragraphs:
        sentences.extend(sent_tokenize(paragraph))

    # Filter out empty sentences or very short sentences
    sentences = [sentence for sentence in sentences if len(sentence.strip()) > 2]

    return sentences

def remove_duplicate_topics(topics, feature_names):
    """
    Removes duplicate topics based on cosine similarity between their word vectors.

    Args:
        topics (list of list of str): A list of topics where each topic is represented by a list of words.
        feature_names (list of str): The list of feature names corresponding to the words used in the topics.

    Returns:
        list of list of str: A list of unique topics, filtered based on cosine similarity.
    """
    unique_topics = []
    topic_vectors = []

    # Convert feature names to a list if it's not already
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    for topic in topics:
        # Create a vector representation of the topic
        topic_vector = [0] * len(feature_names)
        for word in topic:
            if word in feature_names:
                topic_vector[feature_names.index(word)] = 1
        topic_vectors.append(topic_vector)

    filtered_topics = []

    for i, topic_vector in enumerate(topic_vectors):
        is_unique = True
        for unique_vector in topic_vectors[:i]:
            if cosine_similarity([topic_vector], [unique_vector])[0][0] > 0.9:
                is_unique = False
                break
        if is_unique:
            filtered_topics.append(topics[i])

    return filtered_topics


def topic_modeling(tfidf_matrix, feature_names, n_topics=5):
    """
    Performs topic modeling using Latent Dirichlet Allocation (LDA).

    Args:
        tfidf_matrix (array-like): TF-IDF matrix of the document corpus.
        feature_names (list of str): The list of feature names (words) corresponding to the TF-IDF matrix.
        n_topics (int): The number of topics to generate.

    Returns:
        list of list of str: A list of topics, each represented by the top words in that topic.
    """
    lda = LDA(n_components=n_topics, random_state=0)
    lda.fit(tfidf_matrix)
    topics = []
    for idx, topic in enumerate(lda.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topics.append(topic_words)
    return topics

def classify_and_cluster(doc_content_list, n_clusters=5):
    """
    Classifies and clusters the document content into specified number of clusters.

    Args:
        doc_content_list (list of str): A list of document contents.
        n_clusters (int): The number of clusters to generate.

    Returns:
        tuple: A tuple containing:
            - tfidf_matrix (array-like): The TF-IDF matrix of the document content.
            - clusters (array): Cluster labels for each document.
            - topics (list of list of str): A list of topics for each cluster.
            - vectorizer (TfidfVectorizer): The TF-IDF vectorizer used for transforming the documents.
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
    tfidf_matrix = vectorizer.fit_transform(doc_content_list)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    clusters = kmeans.fit_predict(tfidf_matrix)

    # Get topics for each cluster using topic modeling
    topics = []
    feature_names = vectorizer.get_feature_names_out()  # Ensure this is a list of strings

    for cluster_num in range(n_clusters):
        cluster_indices = [i for i, label in enumerate(clusters) if label == cluster_num]

        # If a cluster has data points, generate topics for it
        if len(cluster_indices) > 0:
            cluster_tfidf = tfidf_matrix[cluster_indices]
            cluster_topics = topic_modeling(cluster_tfidf, feature_names)
            filtered_topics = remove_duplicate_topics(cluster_topics, feature_names)
            topics.append(filtered_topics)
        else:
            topics.append([])  # Empty list for clusters with no data points

    return tfidf_matrix, clusters, topics, vectorizer


