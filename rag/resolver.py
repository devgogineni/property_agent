from minsearch import VectorSearch

from utils.embedder import Embedder


class LocationResolver:
    """Resolves fuzzy/misspelled user location text to canonical DB values.

    Wraps a VectorSearch index built (by ingestion/build_location_index.py)
    over the small set of distinct locality/town/district/county/region_name
    values in the data - not the raw 31M PPD rows.
    """

    def __init__(self, vector_index: VectorSearch, embedder: Embedder):
        self.vector_index = vector_index
        self.embedder = embedder

    @classmethod
    def load(cls, index_path: str, embedder: Embedder) -> "LocationResolver":
        vector_index = VectorSearch.load(index_path)
        return cls(vector_index, embedder)

    def resolve(self, text: str, field: str | None = None, num_results: int = 3) -> list[dict]:
        query_vector = self.embedder.encode(text)
        filter_dict = {"field": field} if field else None
        return self.vector_index.search(
            query_vector, filter_dict=filter_dict, num_results=num_results
        )

    def resolve_best(self, text: str, field: str | None = None) -> dict | None:
        results = self.resolve(text, field=field, num_results=1)
        return results[0] if results else None
