from typing import Optional, List
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .document_loader import MultiFileLoader

class DocumentSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
        text_splitter: Optional[RecursiveCharacterTextSplitter] = None,
        max_token_per_chunk: Optional[int]=2500
    ):
        """
        Initialize the DocumentSplitter with chunk size, overlap, and optional separators.
        """
        self.chunk_size =  chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_token_per_chunk = max_token_per_chunk
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]
        if text_splitter is None:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators
            )
        else:
            self.text_splitter = text_splitter
        
    def split(self, docs):
        chunks = []
        for doc in docs:
            chunks.extend(self.text_splitter.split_documents([doc]))
        return chunks
    
    def tokenize_splitter(self, input_text):
        self.tokenizer = AutoTokenizer.from_pretrained("model-artifacts/bge-tokenizer")
        tokens = self.tokenizer.encode(input_text, add_special_tokens=True)
        chunks = []
        for i in range(0, len(tokens), self.max_token_per_chunk - self.chunk_overlap):
            chunk = tokens[i:i + self.max_token_per_chunk]
            chunks.append(self.tokenizer.decode(chunk, skip_special_tokens=True))
        return chunks
    
    def tokenize_split(self, docs):
        # excel: merge multiple rows to one chunk
        chunks = []
        buffer = ""
        buffer_tokens = []

        for doc in docs:
            row_text = doc.page_content
            row_tokens = self.tokenizer.encode(row_text)

            if len(row_tokens) > self.max_token_per_chunk:
                tmp_chunk = self.tokenize_splitter(row_text)
                chunks.extend(tmp_chunk)
            else:
                if len(buffer_tokens) + len(row_tokens) <= self.max_token_per_chunk:
                    buffer += row_text + "\n"
                    buffer_tokens.extend(row_tokens)
                else:
                    if buffer:
                        chunks.append(buffer.strip)
                    buffer = row_text + "\n"
                    buffer_tokens = row_tokens
        if buffer:
            chunks.append(buffer.strip())
        
        return chunks


def main() -> None:
    """Example usage of document_loader with document_splitter."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python document_splitter.py <path> [--recurse] [--progress] [--no-docling]")
        print("  path: Path to file or directory to load and split")
        print("  --recurse: Recursively search subdirectories")
        print("  --progress: Show loading progress")
        print("  --no-docling: Disable docling preprocessing for PDF/Word files")
        sys.exit(1)
    
    docs_path = sys.argv[1]
    recurse = "--recurse" in sys.argv
    show_progress = "--progress" in sys.argv
    use_docling = "--no-docling" not in sys.argv
    
    try:
        # Step 1: Load documents using MultiFileLoader
        loader = MultiFileLoader(
            path=docs_path,
            recurse=recurse,
            show_progress=show_progress,
            use_docling=use_docling
        )
        docs = loader.load_all()
        print(f"Loaded {len(docs)} documents.")
        
        if not docs:
            print("No documents were loaded.")
            return
        
        # Step 2: Split documents into chunks using DocumentSplitter
        splitter = DocumentSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split(docs)
        print(f"Split into {len(chunks)} chunks.")
        
        # Show sample results
        if chunks:
            print(f"\nFirst chunk preview:")
            print(f"Content: {chunks[0].page_content[:200]}...")
            print(f"Metadata: {chunks[0].metadata}")
            
            # Show docling processing statistics
            docling_processed = sum(1 for chunk in chunks if chunk.metadata.get('processed_with') == 'docling')
            if docling_processed > 0:
                print(f"\nChunks from docling-processed documents: {docling_processed}/{len(chunks)}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
