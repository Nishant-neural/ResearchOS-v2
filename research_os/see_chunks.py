from qdrant_client import QdrantClient
import os
import sys

client = QdrantClient(path="qdrant_data")  # your path

COLL_NAME = "research_chunks"

try:
	info = client.get_collection(COLL_NAME)
except ValueError:
	# Try to get collections via client API
	available = []
	try:
		resp = client.get_collections()
		# resp may have different shapes depending on qdrant-client version
		if hasattr(resp, "collections"):
			for c in resp.collections:
				name = getattr(c, "name", None)
				if name:
					available.append(name)
		elif hasattr(resp, "result"):
			for c in resp.result:
				name = getattr(c, "name", None)
				if name:
					available.append(name)
		elif isinstance(resp, list):
			for c in resp:
				available.append(getattr(c, "name", c))
	except Exception:
		pass

	# Fallback: inspect qdrant data folder for collections
	if not available:
		coll_dir = os.path.join("qdrant_data", "collection")
		if os.path.isdir(coll_dir):
			for entry in os.listdir(coll_dir):
				# skip files, take directories
				p = os.path.join(coll_dir, entry)
				if os.path.isdir(p):
					available.append(entry)

	if not available:
		print(f"Collection '{COLL_NAME}' not found and no collections detected.")
		print("Ensure your Qdrant local storage at 'qdrant_data' contains collections.")
		sys.exit(1)

	print(f"Collection '{COLL_NAME}' not found. Available collections:")
	for n in available:
		print(" -", n)

	# Choose the first available collection as a fallback
	chosen = available[1]
	print(f"Showing info for '{chosen}' instead.")
	info = client.get_collection(chosen)

print("Points:", info.points_count)