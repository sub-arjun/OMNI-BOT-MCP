import os
import logging
import asyncio
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)
from llama_index.vector_stores.docarray import DocArrayInMemoryVectorStore
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.openai import OpenAIEmbedding # Using OpenAI for embeddings as a common default

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the directory for storing documents to be indexed
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "indexed_documents")
if not os.path.exists(DOCUMENTS_DIR):
    os.makedirs(DOCUMENTS_DIR)
    logging.info(f"Created documents directory: {DOCUMENTS_DIR}")

class LlamaIndexRAGAgent:
    def __init__(self, anthropic_api_key: str, anthropic_model_name: str, openai_api_key: str | None = None):
        self.anthropic_api_key = anthropic_api_key
        self.anthropic_model_name = anthropic_model_name
        self.openai_api_key = openai_api_key

        if not self.anthropic_api_key:
            raise ValueError("Anthropic API key is required.")
        
        # It's good practice to also check for OpenAI API key if OpenAI embeddings are to be used
        # For now, we'll let it fail later if not set and openai_api_key is None.
        # A more robust solution would be to explicitly require it or allow configurable embedding models.

        try:
            Settings.llm = Anthropic(model=self.anthropic_model_name, api_key=self.anthropic_api_key)
            # If openai_api_key is provided, use it. Otherwise, OpenAIEmbedding will try to use OPENAI_API_KEY env var
            if self.openai_api_key:
                 Settings.embed_model = OpenAIEmbedding(api_key=self.openai_api_key)
            else:
                 Settings.embed_model = OpenAIEmbedding() # relies on OPENAI_API_KEY env var
            
            # For older LlamaIndex versions, you might pass llm and embed_model directly to ServiceContext or Settings.
            # The global `Settings` approach is common in newer versions.

        except Exception as e:
            logging.error(f"Failed to initialize LLM or embedding model: {e}")
            raise

        self.vector_store = DocArrayInMemoryVectorStore()
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Initialize an empty index first
        self.index = VectorStoreIndex.from_documents(
            [], 
            storage_context=self.storage_context,
        )
        
        # Load initial documents synchronously
        self._load_documents_from_dir(DOCUMENTS_DIR)

        # Create query engine after initial documents are loaded
        self.query_engine = self.index.as_query_engine(similarity_top_k=3)
        logging.info(f"LlamaIndexRAGAgent initialized. Query engine created. {len(self.index.docstore.docs)} documents loaded.")

    def _load_documents_from_dir(self, dir_path: str):
        """
        Loads all documents from the specified directory into the index.
        This is a synchronous method called during initialization.
        """
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            logging.warning(f"Initial documents directory not found or is not a directory: {dir_path}")
            return

        loaded_count = 0
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path):
                try:
                    logging.info(f"Loading initial document: {file_path}")
                    reader = SimpleDirectoryReader(input_files=[file_path])
                    documents = reader.load_data() # Synchronous call

                    if not documents:
                        logging.warning(f"No documents loaded from file: {file_path}")
                        continue

                    logging.info(f"Adding {len(documents)} document(s) from \'{filename}\' to the index during init.")
                    for doc in documents:
                        self.index.insert(doc) # Synchronous call
                    loaded_count += len(documents)
                except Exception as e:
                    logging.error(f"Failed to load/add initial file \'{filename}\' to index: {e}")
            else:
                logging.info(f"Skipping non-file item in documents directory: {filename}")
        
        if loaded_count > 0:
            logging.info(f"Successfully loaded {loaded_count} initial documents from {dir_path}.")
        else:
            logging.info(f"No initial documents found or loaded from {dir_path}.")


    async def _add_file_to_index_by_path(self, file_path: str, filename_for_logging: str):
        """
        Adds a single file (given its full path) to the in-memory vector index.
        Updates the query engine. This is an async helper.
        """
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            logging.warning(f"File not found or is not a file: {file_path}")
            return f"Error: File \'{filename_for_logging}\' not found at the specified path."

        try:
            logging.info(f"Loading documents from: {file_path}")
            reader = SimpleDirectoryReader(input_files=[file_path])
            documents = await asyncio.to_thread(reader.load_data)

            if not documents:
                logging.warning(f"No documents loaded from file: {file_path}")
                return f"Warning: No documents could be loaded from \'{filename_for_logging}\'."

            logging.info(f"Adding {len(documents)} document(s) from \'{filename_for_logging}\' to the index.")
            for doc in documents:
                await asyncio.to_thread(self.index.insert, doc)
            
            self.query_engine = await asyncio.to_thread(self.index.as_query_engine, similarity_top_k=3)
            logging.info(f"Successfully added \'{filename_for_logging}\' to index and updated query engine.")
            return f"Successfully added \'{filename_for_logging}\' to the knowledge base."
        except Exception as e:
            logging.error(f"Failed to add file \'{filename_for_logging}\' to index: {e}")
            return f"Error processing file \'{filename_for_logging}\': {e}"

    async def add_file_to_index(self, filename: str):
        """
        Adds a file to the in-memory vector index.
        The filename is relative to the \'indexed_documents\' directory.
        This method is intended for dynamic additions post-initialization.
        """
        file_path = os.path.join(DOCUMENTS_DIR, filename)
        return await self._add_file_to_index_by_path(file_path, filename)

    async def query(self, user_query: str):
        """
        Queries the LlamaIndex agent.
        """
        if not self.query_engine:
            logging.error("Query engine is not initialized.")
            return "Error: The agent's query engine is not ready."
        
        logging.info(f"Querying LlamaIndex agent with: '{user_query}'")
        try:
            # Run blocking query in a separate thread
            response = await asyncio.to_thread(self.query_engine.query, user_query)
            logging.info(f"LlamaIndex agent response: {response.response}")
            return response.response
        except Exception as e:
            logging.error(f"Error during LlamaIndex query: {e}")
            return f"Error querying the LlamaIndex agent: {e}"

async def test_agent():
    print("Testing LlamaIndexRAGAgent...")
    try:
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        llm_model = os.environ.get("LLM_MODEL", "claude-3-haiku-20240307")

        if not anthropic_key:
            print("ANTHROPIC_API_KEY environment variable not set. Skipping test.")
            return
        if not openai_key:
            print("OPENAI_API_KEY environment variable not set for embeddings. Skipping test.")
            return
        
        agent = LlamaIndexRAGAgent(
            anthropic_api_key=anthropic_key, 
            anthropic_model_name=llm_model,
            openai_api_key=openai_key
        )
        
        dummy_file_path = os.path.join(DOCUMENTS_DIR, "test_doc.txt")
        with open(dummy_file_path, "w") as f:
            f.write("This is a test document. LlamaIndex is a framework for building LLM applications.")
        
        print(await agent.add_file_to_index("test_doc.txt"))
        
        response = await agent.query("What is LlamaIndex?")
        print(f"Test Query Response: {response}")

        response_unknown = await agent.query("Tell me about quantum physics.")
        print(f"Test Query Response (Unknown): {response_unknown}")
        
        os.remove(dummy_file_path)

    except Exception as e:
        print(f"Error during LlamaIndexRAGAgent test: {e}")

if __name__ == '__main__':
    # Run the async test function
    asyncio.run(test_agent()) 