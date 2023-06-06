from langchain.embeddings import HuggingFaceEmbeddings

"""
this function is from the following stackoverflow post:
https://stackoverflow.com/questions/50008296/facebook-json-badly-encoded
It converts the latin-1 encoded strings to utf-8 encoded strings due to Facebook's bad encoding.
It iteratively does this for all strings in the json object.
"""
def unicode_converter(obj: dict):
    if isinstance(obj, str):
        return obj.encode('latin_1').decode('utf-8')

    if isinstance(obj, list):
        return [unicode_converter(o) for o in obj]

    if isinstance(obj, dict):
        return {key: unicode_converter(item) for key, item in obj.items()}

    return obj

def hf_embedding():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

