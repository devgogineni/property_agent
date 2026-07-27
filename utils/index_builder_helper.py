from minsearch import Index, VectorSearch

from utils.embedder import Embedder


def build_text_index(documents, text_fields_list, keyword_fields_list):
    index = Index(
        text_fields=text_fields_list,
        keyword_fields=keyword_fields_list
    )
    index.fit(documents)
    return index


def text_search(index, query, num_results=5):
    return index.search(query, num_results=num_results)


def build_vector_index(documents, text_field, keyword_fields_list, embedder: Embedder):
    """Embed documents[text_field] with `embedder` and fit a VectorSearch index.

    VectorSearch.fit() takes pre-computed vectors, not an embedder, so the
    embedding step happens here rather than inside minsearch.
    """
    texts = [doc[text_field] for doc in documents]
    vectors = embedder.encode_batch(texts)

    vector_index = VectorSearch(keyword_fields=keyword_fields_list)
    vector_index.fit(vectors, documents)
    return vector_index


def vector_search(vector_index, query, embedder: Embedder, num_results=5, filter_dict=None):
    query_vector = embedder.encode(query)
    return vector_index.search(
        query_vector, num_results=num_results, filter_dict=filter_dict
    )
