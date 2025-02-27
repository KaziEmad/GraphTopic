# GraphTopic

- [GraphTopic](#graphtopic)
  - [Introduction to GraphTopic](#introduction-to-graphtopic)
  - [Running GraphTopic in a Virtual Environment](#running-graphtopic-in-a-virtual-environment)
  - [Usage](#usage)
    - [Basic Example](#basic-example)
    - [Class `GraphTopic`](#class-graphtopic)
      - [`__init__(self, docs, TopM=500, TopN=10, keyphrase_extractor=getKeyPhrasesUsingKeyBERT, embedding_model=embeddingModel, pretty_print=False, prune_node_count_less_than=2, overlapping_threshold=0.5, max_topics=float("inf"), max_keyphrases_per_topic=100, keyphrases_selection_strategy=KeyphrasesSelectionStrategy.CDIST)`](#__init__self-docs-topm500-topn10-keyphrase_extractorgetkeyphrasesusingkeybert-embedding_modelembeddingmodel-pretty_printfalse-prune_node_count_less_than2-overlapping_threshold05-max_topicsfloatinf-max_keyphrases_per_topic100-keyphrases_selection_strategykeyphrasesselectionstrategycdist)
      - [`get_topics(self) -> List[List]`](#get_topicsself---listlist)
  - [Comparison of GraphTopic with Other Models](#comparison-of-graphtopic-with-other-models)
    - [**Datasets Used**](#datasets-used)
    - [**Topic Modeling Methods Evaluated**](#topic-modeling-methods-evaluated)
    - [**Evaluation Metrics**](#evaluation-metrics)
    - [**Processing Steps**](#processing-steps)
    - [**Results Interpretation**](#results-interpretation)
  - [**Contribution**](#contribution)
  - [**Conclusion**](#conclusion)


## Introduction to GraphTopic

GraphTopic is a topic modeling framework designed for extracting and merging topics from a list of documents using a graph-based approach. It leverages keyphrase extraction, sentence embeddings, and graph theory to identify coherent and meaningful topics within a document collection.

This method excels at discovering nuanced topic structures by:

1. **Extracting Keyphrases:**  Utilizes a keyphrase extraction technique (defaulting to KeyBERT) to identify important phrases within documents.
2. **Building Trivial Topics:** Initializes individual topics based on top keyphrases from a merged document representing the entire corpus.
3. **Merging Latent Topics:** Iteratively processes documents, merging related trivial topics based on shared keyphrases and semantic similarity. This forms a graph of interconnected topics.
4. **Final Topic Aggregation:** Merges semantically close latent topics using sentence embeddings to produce a concise and comprehensive set of final topics.

## Running GraphTopic in a Virtual Environment

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/dev-SR/GraphTopic.git
   ```

2. **Create a Virtual Environment**:
   It is recommended to use a virtual environment to manage dependencies.
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install Required Packages**:
   Install the necessary packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

5. **Download required `nltk` resources**
   Go to python shell and run:
   ```python
   venv\Scripts\activate
   python
   >>> import nltk
   >>> nltk.download("stopwords")
   >>> nltk.download("punkt")
   >>> nltk.download("punkt_tab")
   >>> nltk.download("averaged_perceptron_tagger")
   >>> nltk.download("averaged_perceptron_tagger_eng")
   ```

6. **Run a sample example**:
   To execute the tests and see the results, run:
   ```bash
   venv\Scripts\activate
   python runner.py
   ```

## Usage

### Basic Example

```python
from GraphTopic import GraphTopic

# List of documents to analyze
documents = [
    "Machine learning is a field of artificial intelligence that focuses on building models....",
    "Deep learning is a subset of machine learning that uses neural networks.....",
    "Graph neural networks are used for analyzing graph-structured data...."
]

# Initialize the GraphTopic model
graph_topic_model = GraphTopic(
    docs=documents,
    TopM=300,  # Number of keyphrases to extract from the merged document
    TopN=8,    # Number of keyphrases to extract from each document
    overlapping_threshold=0.6  # Threshold for merging topics
)

# Retrieve extracted topics
topics = graph_topic_model.get_topics()
print("Extracted Topics:", topics)
```


Here's a basic example demonstrating how to use the `GraphTopic` class:


### Class `GraphTopic`

#### `__init__(self, docs, TopM=500, TopN=10, keyphrase_extractor=getKeyPhrasesUsingKeyBERT, embedding_model=embeddingModel, pretty_print=False, prune_node_count_less_than=2, overlapping_threshold=0.5, max_topics=float("inf"), max_keyphrases_per_topic=100, keyphrases_selection_strategy=KeyphrasesSelectionStrategy.CDIST)`

**Initializes the GraphTopic model.**

**Parameters:**

* **`docs` (List[str])**:  A list of document texts to be analyzed.
* **`TopM` (int, optional)**:
    * **Default:** `500`
    * **Description:**  Determines the number of top keyphrases extracted from the merged document. These keyphrases are used to initialize trivial topics. A higher `TopM` value may lead to more granular and specific initial topics.
* **`TopN` (int, optional)**:
    * **Default:** `10`
    * **Description:**  Specifies the number of top keyphrases to extract from each individual document. These document keyphrases are used to merge and refine the trivial topics into latent topics. A lower `TopN` value may result in broader, more general latent topics.
* **`keyphrase_extractor` (Callable[[str, int], List[Tuple[str, float]]], optional)**:
    * **Default:** `getKeyPhrasesUsingKeyBERT`
    * **Description:** A function responsible for extracting keyphrases from a given text. It should accept the text and the desired number of keyphrases (`n`) as input and return a list of tuples, where each tuple contains a keyphrase (string) and its associated weight (float). You can provide a custom keyphrase extraction function if needed, conforming to this signature.
    * **Example Function Signature:**
        ```python
        def getKeyPhrases(text, n):
            # ... your keyphrase extraction logic ...
            return [("keyphrase1", 0.9), ("keyphrase2", 0.8), ...]
        ```
* **`embedding_model` (SentenceTransformer, optional)**:
    * **Default:** `embeddingModel` (from `GraphTopic.helpers.config`, typically a SentenceTransformer model)
    * **Description:**  A SentenceTransformer model used for embedding keyphrases when merging semantically similar topics. This model is crucial for determining topic similarity during the final topic merging phase.
* **`pretty_print` (bool, optional)**:
    * **Default:** `False`
    * **Description:**  If set to `True`, enables verbose output during the topic merging process, printing intermediate steps and similarity scores. Useful for debugging and understanding the merging process.
* **`prune_node_count_less_than` (int, optional)**:
    * **Default:** `2`
    * **Description:**  Specifies the minimum number of keyphrases (nodes in the topic graph) that a latent topic must have to be retained. Topics with fewer keyphrases than this threshold are pruned from the latent topic graph, helping to filter out less significant or overly specific topics.
* **`overlapping_threshold` (float, optional)**:
    * **Default:** `0.5`
    * **Description:**  This is the cosine similarity threshold used to merge semantically similar latent topics into final topics. A higher `overlapping_threshold` means topics need to be more similar to be merged. Increasing this threshold will generally result in fewer, more distinct final topics.
* **`max_topics` (float, optional)**:
    * **Default:** `float("inf")` (infinity - all topics are returned)
    * **Description:**  Limits the maximum number of final topics to be returned. If set to a finite number, the method will return only the top `max_topics` topics, ranked according to the chosen `keyphrases_selection_strategy`.
* **`max_keyphrases_per_topic` (int, optional)**:
    * **Default:** `100`
    * **Description:**  Specifies the maximum number of keyphrases to include in each final topic when retrieving the topics using `get_topics()`. This is useful for controlling the size and detail of the topic representations.
* **`keyphrases_selection_strategy` (KeyphrasesSelectionStrategy, optional)**:
    * **Default:** `KeyphrasesSelectionStrategy.CDIST`
    * **Description:**  Determines the strategy used to select the top keyphrases for each topic when `max_topics` is specified. Options include:
        * `KeyphrasesSelectionStrategy.IMP_NODE`: Selects keyphrases based on their importance in the topic graph (node outdegree centrality).
        * `KeyphrasesSelectionStrategy.CDIST`: Selects keyphrases that are most semantically central to the topic, based on cosine similarity to the topic centroid embedding.
        * `KeyphrasesSelectionStrategy.RNDOM`: Randomly selects keyphrases from the topic.

#### `get_topics(self) -> List[List]`

**Retrieves the final extracted topics.**

**Returns:**

* **`List[List]`**: A list of final topics. Each topic is represented as a list of keyphrases (strings). If `max_topics` is specified in the constructor, each topic will contain at most `max_keyphrases_per_topic` keyphrases, selected according to the `keyphrases_selection_strategy`.


## Comparison of GraphTopic with Other Models

The objective is to evaluate the performance of multiple topic extraction techniques and compare their effectiveness using various metrics.
The `test-runner.py` script provides a comprehensive comparison of different topic modeling techniques, including:

- **BERT Topic Modeling**: - Utilizes BERT embeddings to extract topics and evaluate their coherence, semantic similarity, and diversity.
- **LDA (Latent Dirichlet Allocation)**: A generative probabilistic model that identifies topics based on word distributions.
- **LSI (Latent Semantic Indexing)**: Uses singular value decomposition to reduce dimensionality and identify topics.
- **HDP (Hierarchical Dirichlet Process)**: A nonparametric Bayesian approach that allows for an unknown number of topics.
- **NMF (Non-negative Matrix Factorization)**: Factorizes the document-term matrix into non-negative factors to identify topics.


### **Datasets Used**
1. **20 Newsgroups(18840 documents)**
2. **Trump Dataset(56,570 documents)**
3. **BBC Dataset(11,125 documents)**

Each dataset consists of a large number of documents. To ensure robust evaluation, each dataset was divided into five equal chunks, and topic modeling techniques were applied separately to each chunk. The final reported results for each dataset are the average scores computed over these five chunks. The chunk sizes for each dataset are as follows: Trump Dataset - 11,314 documents per chunk, BBC Dataset - 2,225 documents per chunk, and 20 Newsgroups - 3,768 documents per chunk.

### **Topic Modeling Methods Evaluated**
- **Bertopic**
- **LDA (Latent Dirichlet Allocation)**
- **LSI (Latent Semantic Indexing)**
- **HDP (Hierarchical Dirichlet Process)**
- **NMF (Non-negative Matrix Factorization)**
- **GraphTopic**

### **Evaluation Metrics**
To compare the performance of the topic modeling techniques, the following evaluation metrics were used:
- **Coherence Scores:**
  - c_npmi
  - c_uci
  - c_v
  - u_mass
- **Semantic Coherence Metrics:**
  - Inter-topic distance average
  - Intra-topic similarity average
  - Overall coherence score of average of inter-topic distance and intra-topic similarity
- **Topic Diversity Score**

### **Processing Steps**
1. **Text Preprocessing**:
   - Extract noun phrases from documents.
   - Apply deep text cleaning.
2. **Corpus and Dictionary Creation**:
   - Convert text into a corpus using Gensim preprocessing.
3. **Topic Extraction**:
   - Apply each topic modeling method to extract topics.
4. **Score Calculation**:
   - Compute coherence, semantic coherence, and diversity scores for each method.
5. **Aggregation**:
   - Aggregate results over five chunks for each dataset to compute the final average scores.
6. **Result Storage**:
   - Save individual and average results to CSV files for further analysis.

### **Results Interpretation**
The final output includes:
- A dataframe of all test results.
- An aggregated dataframe with average results per method.
- CSV files for both individual and average results.

The average results help in assessing the overall performance of each method, providing insights into which techniques perform best across different datasets.



## **Contribution**

We welcome contributions to enhance this analysis. If you wish to contribute, please follow these steps:

1. **Fork the Repository**: Start by forking the project repository to your own GitHub account.
2. **Create a New Branch**: Use a descriptive name for your branch, such as `feature-improve-metrics` or `bugfix-data-processing`.
3. **Implement Changes**: Make the necessary modifications, whether it’s refining the evaluation metrics, improving the data preprocessing steps, or adding new topic modeling techniques.
4. **Test Your Changes**: Ensure that your modifications work correctly by running tests and verifying that results are consistent.
5. **Submit a Pull Request (PR)**:
   - Provide a clear and concise description of the changes made.
   - Include any relevant test results or explanations for new features.
   - Request a review from the maintainers.
6. **Collaborate on Review**: Address any feedback or requested changes from the project maintainers.

We appreciate all contributions and encourage discussions to improve the topic modeling evaluation framework further!



## **Conclusion**
This experimental setup ensures a fair and comprehensive evaluation of topic modeling methods. By analyzing the average results across multiple dataset chunks, we gain a better understanding of the strengths and weaknesses of each method in handling real-world text data.


---

This README provides a comprehensive overview of GraphTopics, guiding users through setup, usage, and performance evaluation against other topic modeling techniques. For further assistance, please refer to the documentation or contact the maintainers.