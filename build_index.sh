python -m pyserini.index.lucene \
  -collection JsonCollection \
  -generator DefaultLuceneDocumentGenerator \
  -input ./zhwiki/ \
  -index ./index \
  -language zh \
  -threads 64 \
  -storePositions -storeDocvectors -storeRaw