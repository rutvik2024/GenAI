# Advanced Document QA System

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Components](#components)
  - [Web Crawling](#web-crawling)
  - [Data Chunking and Vector Database Creation](#data-chunking-and-vector-database-creation)
  - [Retrieval and Re-ranking](#retrieval-and-re-ranking)
  - [Question Answering](#question-answering)
- [Configuration](#configuration)
- [Environment Variables](#environment-variables)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Task Description](#task-description)

## Project Overview

This project implements a sophisticated Question Answering (QA) system for technical documentation. It combines web crawling, advanced natural language processing techniques, and machine learning to provide accurate answers to user queries about complex technical topics.

## Features

- Web crawling of NVIDIA CUDA documentation up to 5 levels deep
- Intelligent data chunking based on semantic similarity and topic modeling
- Vector database storage using Pinecone with HNSW indexing
- Hybrid retrieval system combining BM25 and BERT/bi-encoder methods
- Query expansion for improved retrieval accuracy
- Re-ranking of retrieved results for enhanced relevance
- Integration with a Language Model for generating accurate answers


## Prerequisites

Before you begin, ensure you have the following:

1. Python 3.7 or higher installed
2. A Pinecone account (Sign up at [https://www.pinecone.io/](https://www.pinecone.io/))
3. A Hugging Face account (Sign up at [https://huggingface.co/](https://huggingface.co/))
4. Access to the LLaMA 3 model (Request access at [https://huggingface.co/meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B))

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/advanced-document-qa-system.git
   cd advanced-document-qa-system
   ```

2. Create a virtual environment:
   
   Using venv (Python 3.3+):
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```
   
   Or using conda:
   ```bash
   conda create -n doc-qa-env python=3.8
   conda activate doc-qa-env
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root directory with the following content:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   HuggingFace_API_KEY=your_huggingface_api_key
   ENVIRONMENT=your_pinecone_environment
   CLOUD=your_cloud_provider
   INDEX_NAME=your_index_name
   ```
   Replace the placeholders with your actual values:
   - Get your Pinecone API key from your Pinecone console
   - Get your Hugging Face API key from your Hugging Face account settings
   - `ENVIRONMENT`: Your Pinecone environment (e.g., "us-west1-gcp")
   - `CLOUD`: Your cloud provider for Pinecone (e.g., "gcp" or "aws")
   - `INDEX_NAME`: Choose any name for your Pinecone index (e.g., "document-qa-index")

## Usage

[... usage instructions remain unchanged ...]

## Components

[... components sections remain unchanged ...]

## Configuration

[... configuration section remains unchanged ...]

## Environment Variables

Create a `.env` file in the project root with the following variables:

- `PINECONE_API_KEY`: Your Pinecone API key
- `HuggingFace_API_KEY`: Your Hugging Face API key
- `ENVIRONMENT`: Pinecone environment
- `CLOUD`: Cloud provider for Pinecone
- `INDEX_NAME`: Name of your Pinecone index

## Dependencies

[... dependencies section remains unchanged ...]

## Contributing

[... contributing section remains unchanged ...]

## License

[... license section remains unchanged ...]

## Task Description

* Web Crawling:
   * Developed a web crawler that scrapes data from the provided website.
   * The web crawler retrieves data from sub-links present within the parent link.
   * Sub-link data retrieval is performed up to a depth of 5 levels.
   * The web crawler identifies and scrapes data from both parent links and their corresponding sub-links.
   * Content is fetched only from the `<div>` tag with "Class: document". Remaining content is discarded to avoid redundancy and navigation links.

* Data Chunking and Vector Database Creation:
   * Implemented advanced chunking techniques based on sentence/topic similarity.
   * Converted chunked data into embedding vectors.
   * Created a vector database using Pinecone and stored embedding vectors using the HNSW indexing method.
   * Stored relevant metadata (web-link, actual chunk) along with the chunks in the vector database.

* Retrieval and Re-ranking:
   * Implemented query expansion techniques to enhance the retrieval process.
   * Employed hybrid retrieval methods combining BM25 and BERT/bi-encoder based retrieval.
   * Implemented re-ranking of retrieved data based on relevance and similarity to the query.

* Question Answering:
   * Integrated the LLaMA 3 model for generating accurate answers based on the retrieved and re-ranked data.

This system provides a comprehensive solution for extracting, processing, and querying technical documentation, enabling efficient and accurate question answering capabilities.








## Features



## System Architecture

[Include a system architecture diagram here if available]

## Installation

```bash
git clone https://github.com/yourusername/nvidia-cuda-qa-system.git
cd nvidia-cuda-qa-system
pip install -r requirements.txt
```

## Usage

1. Set up your environment variables in a `.env` file (see [Environment Variables](#environment-variables) section).
2. Run the web crawling component (not included in the provided files).
3. Execute the data chunking and embedding creation:
   ```
   python 2_create_chunks.py
   python 3_create_embedding.py
   ```
4. Start the QA system:
   ```
   python 4_retrieval_rerank.py
   ```
5. Enter your questions when prompted.

## Components

### Web Crawling

- Scrapes data from https://docs.nvidia.com/cuda/
- Crawls sub-links up to a depth of 5 levels
- Focuses on content within `<div class="document">` tags

### Data Chunking and Vector Database Creation

Implemented in `2_create_chunks.py` and `3_create_embedding.py`:

- Uses SentenceTransformer model 'all-MiniLM-L6-v2' for semantic chunking
- Creates chunks with a maximum length of 128 tokens and a minimum of 2 chunks per document
- Removes stop words and punctuation from chunks
- Generates embeddings for each chunk
- Stores chunks and embeddings in a CSV file
- Creates a Pinecone index and uploads the vectors

### Retrieval and Re-ranking

Implemented in `4_retrieval_rerank.py`:

- Initializes Pinecone index for vector search
- Implements query expansion using SentenceTransformer
- Uses a hybrid search combining vector similarity and BM25
- Re-ranks results using a cross-encoder model (ms-marco-MiniLM-L-6-v2)

### Question Answering

Implemented in `4_retrieval_rerank.py`:

- Uses Meta-Llama-3-8B model for generating answers
- Processes retrieved and re-ranked data to produce accurate responses
- Provides top 5 most relevant document snippets along with the answer

## Configuration

The `config.py` file contains important configuration parameters:

- `json_file_path`: Path to the JSON file containing scraped data
- `csv_file_path`: Path to store the CSV file with chunks and embeddings
- `sentence_model_name`: Name of the SentenceTransformer model used
- `rerank_model_name`: Name of the re-ranking model used
- `llama_model_name`: Name of the language model used for question answering

## Environment Variables

Create a `.env` file in the project root with the following variables:

- `PINECONE_API_KEY`: Your Pinecone API key
- `HuggingFace_API_KEY`: Your Hugging Face API key
- `ENVIRONMENT`: Pinecone environment
- `CLOUD`: Cloud provider for Pinecone
- `INDEX_NAME`: Name of your Pinecone index

## Dependencies

Key dependencies include:

- scrapy
- beautifulsoup4
- pandas
- scikit-learn
- sentence_transformers
- pinecone-client
- transformers
- torch
- nltk

For a full list of dependencies, see the `requirements.txt` file.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.