import chromadb

client = chromadb.Client()
collection = client.create_collection(name='test',
                                      metadata={"hnsw:space": "cosine"})

collection.add(documents=["User's name is Ubadah",
                          "The user is called Ubadah", "User likes volleyball"], ids=['1', '2', '3'])
result = collection.query(query_texts=["User's name is Ubadah"], n_results=3)
print(result
      )
