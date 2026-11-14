from rag_pipeline.ingestion.normalize import normalize_plaintext

messy = "Line one  \r\n\r\n\r\n\r\nLine two"
print(repr(normalize_plaintext(messy)))
# Expect: 'Line one\n\nLine two'