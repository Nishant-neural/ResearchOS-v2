docker run -d --name qdrant_server `
  -p 6333:6333 -p 6334:6334 `
  -v qdrant_data:/qdrant/storage `
  qdrant/qdrant:latest


  docker start qdrant_server