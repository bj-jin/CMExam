python -m pyserini.index.lucene \
  -collection JsonCollection \
  -generator DefaultLuceneDocumentGenerator \
  -input ./trainset/ref/ \
  -index ./index_ref \
  -language zh \
  -threads 20 \
  -storePositions -storeDocvectors -storeRaw