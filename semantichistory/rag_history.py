import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
import chromadb


class RAGHistory:
    #MARK: init
    def __init__(self):
        self.embed_model = 'nomic-embed-text'
        self.original_db_path = os.path.join("/home/neeraj/.config/google-chrome", "Profile 7", "History")
        self.copied_db_path = "/tmp/chrome_history_copy.db"
        self.llm="llama3.1:8b"

    #MARK: decode_transition
    def decode_transition(self,transition):
        """
        Decode Chrome transition types into human-readable descriptions.
        """
        core_type = transition & 0xFF  # Extract lower 8 bits
        qualifiers = {
            0x01000000: "Redirect (server-side)",
            0x02000000: "Redirect (user-initiated)",
            0x04000000: "Forward/Back navigation",
            0x08000000: "From address bar entry",
            0x10000000: "Home page load"
        }
        core_types = {
            0: "LINK (Clicked a hyperlink)",
            1: "TYPED (Manually typed URL)",
            2: "AUTO_BOOKMARK (Opened from a bookmark)",
            3: "AUTO_SUBFRAME (Auto-loaded frame, like an ad)",
            4: "MANUAL_SUBFRAME (Manually opened a subframe)",
            5: "GENERATED (Autocomplete suggestion)",
            6: "START_PAGE (Chrome’s homepage)",
            7: "FORM_SUBMIT (Submitted a form)",
            8: "RELOAD (Page refresh)",
            9: "KEYWORD (Search query typed in Omnibox)",
            10: "KEYWORD_GENERATED (Autocomplete search)"
        }
        meaning = core_types.get(core_type, "UNKNOWN")
        for mask, desc in qualifiers.items():
            if transition & mask:
                meaning += f", {desc}"
        return meaning

    #MARK: get_chrome_history
    def get_chrome_history(self):
        """
        Extract Chrome browsing history and save it to CSV files.
        """
        # Define original and copied database paths

        # Copy the database to avoid locking issues
        shutil.copy2(self.original_db_path, self.copied_db_path)

        # Connect to copied database
        conn = sqlite3.connect(self.copied_db_path)
        cursor = conn.cursor()

        # Query history data with additional information
        query = """
            SELECT urls.url, urls.title, urls.visit_count, urls.typed_count, 
                visits.visit_time, visits.transition 
            FROM urls 
            JOIN visits ON urls.id = visits.url 
            ORDER BY visits.visit_time DESC
        """

        cursor.execute(query)
        results = cursor.fetchall()

        # Convert Chrome timestamp to datetime
        chrome_epoch = datetime(1601, 1, 1)
        data = []
        for url, title, visit_count, typed_count, timestamp, transition in results:
            if timestamp:
                dt = chrome_epoch + timedelta(microseconds=timestamp)
                data.append({
                    "url": url,
                    "title": title,
                    "visit_count": visit_count,
                    "typed_count": typed_count,
                    "timestamp": dt.isoformat(),
                    "transition": transition,
                    "transition_meaning": self.decode_transition(transition)
                })

        # Fetch search queries
        search_query = """
            SELECT urls.url, keyword_search_terms.term 
            FROM keyword_search_terms 
            JOIN urls ON keyword_search_terms.url_id = urls.id;
        """
        cursor.execute(search_query)
        search_results = cursor.fetchall()

        search_data = [{"url": url, "search_term": term} for url, term in search_results]

        conn.close()

        # Create DataFrames
        history_df = pd.DataFrame(data)
        search_df = pd.DataFrame(search_data)

        # Save to CSV files for easy viewing
        history_df.to_csv("docs/chrome_browsing_history.csv", index=False)
        search_df.to_csv("docs/chrome_search_queries.csv", index=False)

        # Merge search queries into browsing history
        merged_df = history_df.merge(search_df, on="url", how="left")
        merged_df["search_term"] = merged_df["search_term"].fillna("")
        merged_df.to_csv("docs/chrome_merged_history.csv", index=False)

        # Convert DataFrame to text format
        merged_text = merged_df.apply(
            lambda row: ', '.join(f"{col}: {row[col]}" for col in merged_df.columns), axis=1
        )
        final_text = '\n'.join(merged_text)
        return final_text

    #MARK: generate_embeddings
    def generate_embeddings(self,text):
        """
        Generate embeddings for the given text using Ollama.
        """
        response = ollama.embeddings(model=self.embed_model, prompt=text)
        return response['embedding']
    
    #MARK: set_up_chromadb
    def set_up_chromadb(self):
        client = chromadb.PersistentClient(path="local_chroma_db")

        # Check if the collection already exists
        existing_collections = client.list_collections()
        
        if "history" in existing_collections:
            collection = client.get_collection(name="history")
        else:
            collection = client.create_collection(name="history")

        return collection

    #MARK: embed_history
    def embed_history(self):
        """
        Set up ChromaDB with Chrome browsing history data.
        """
        # Get Chrome browsing history as text
        history_text = self.get_chrome_history()
        entries = history_text.split('\n')

        # Create meaningful documents with metadata
        documents = []
        metadatas = []

        for entry in entries:
            # Parse entry back into key-value pairs
            metadata = {}
            fields = entry.split(', ')
            for field in fields:
                if ': ' in field:
                    key, value = field.split(': ', 1)
                    metadata[key] = value

            # Create formatted document text
            doc_text = f"""Title: {metadata.get('title', '')}
    URL: {metadata.get('url', '')}
    Visit Count: {metadata.get('visit_count', '')}
    Typed Count: {metadata.get('typed_count', '')}
    Timestamp: {metadata.get('timestamp', '')}
    Transition: {metadata.get('transition', '')}
    Transition_meaning: {metadata.get('transition_meaning', '')}
    Search Term: {metadata.get('search_term', '')}"""

            documents.append(doc_text)
            metadatas.append(metadata)

        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True
        )

        # Split documents
        all_splits = []
        all_metadatas = []

        for doc, meta in zip(documents, metadatas):
            splits = text_splitter.create_documents([doc], [meta])
            for split in splits:
                all_splits.append(split.page_content)
                # Add position information to metadata
                all_metadatas.append({
                    **meta,
                    "chunk_start": split.metadata.get('start_index', 0),
                    "chunk_end": split.metadata.get('start_index', 0) + len(split.page_content)
                })

        # Generate embeddings for all chunks
        embeddings = [self.generate_embeddings(text) for text in all_splits]

        # Create Chroma collection
        collection = self.set_up_chromadb()

        # Add to ChromaDB with proper chunking
        collection.add(
            embeddings=embeddings,
            documents=all_splits,
            metadatas=all_metadatas,
            ids=[str(i) for i in range(len(all_splits))]
        )

    def query_history_assistant(self, query: str):
        """
        Query ChromaDB and pass the retrieved documents to Ollama for a history assistant chatbot response.
        """
        # Retrieve the ChromaDB collection
        collection = self.set_up_chromadb()

        # Search for relevant browsing history entries (top 5 matches)
        results = collection.query(
            query_embeddings=[self.generate_embeddings(query)],  # Convert query to embedding
            n_results=5
        )

        # Extract retrieved documents and metadata
        # retrieved_docs = results.get("documents", [[]])[0]  # List of text chunks
        # retrieved_metadata = results.get("metadatas", [[]])[0]  # Corresponding metadata

        # if not retrieved_docs:
        #     return "No relevant browsing history found for your query."

        # # Format retrieved history entries
        # formatted_entries = "\n\n".join(
        #     [f"Title: {meta.get('title', 'Unknown')}\nURL: {meta.get('url', 'N/A')}\nSnippet: {doc[:200]}..."
        #     for doc, meta in zip(retrieved_docs, retrieved_metadata)]
        # )
        # Extract retrieved documents and metadata
        retrieved_docs = results.get("documents", [[]])[0]  # List of text chunks
        retrieved_metadata = results.get("metadatas", [[]])[0]  # Corresponding metadata
        retrieved_distances = results.get("distances", [[]])[0]  # Similarity scores

        if not retrieved_docs:
            return "No relevant browsing history found for your query."

        # Format retrieved history entries
        formatted_list = []
        for i in range(len(retrieved_docs)):
            meta = retrieved_metadata[i]
            formatted_list.append(f"""
            Document: Title: {meta.get('title', 'Unknown')}
            URL: {meta.get('url', 'N/A')}
            Visit Count: {meta.get('visit_count', 'N/A')}
            Typed Count: {meta.get('typed_count', 'N/A')}
            Timestamp: {meta.get('timestamp', 'N/A')}
            Transition: {meta.get('transition', 'N/A')}
            Transition Meaning: {meta.get('transition_meaning', 'N/A')}
            Search Term: {meta.get('search_term', 'N/A')}
            Metadata: {meta}
            Distance: {retrieved_distances[i]:.4f}""")

        formatted_entries = "\n".join(formatted_list)

        print(formatted_entries)
        print("\n\n")

        # Construct the best prompt for Ollama
        prompt = """
        You are a history assistant chatbot that helps users recall their past browsing history.

        The user has asked: "{query}"

        Answer the user question from the below given context:

        {formatted_entries}
        """

        # Call Ollama to generate a response
        response = ollama.chat(model=self.llm, messages=[{"role": "user", "content": prompt.format(query=query,formatted_entries=formatted_entries)}])

        return response['message']['content']

if __name__ == '__main__':
    rgh = RAGHistory()
    response = rgh.query_history_assistant("Give me the time I looked into henrythe9th git repo")
    print(response)