import gradio as gr
from typing import Optional, List, Tuple
from dataclasses import dataclass
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer, AutoModelForCausalLM, Pipeline, BitsAndBytesConfig, pipeline
import datasets
from datasets import Dataset as HFDataset  # Added explicit import
import torch
from ragatouille import RAGPretrainedModel
from tqdm import tqdm

@dataclass
class RAGConfig:
    """Configuration class for RAG pipeline"""
    embedding_model_name: str = "thenlper/gte-small"
    chunk_size: int = 512
    chunk_overlap: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class RAGPipeline:
    """Main RAG pipeline implementation"""
    def __init__(self, config: RAGConfig):
        self.config = config
        self.markdown_separators = [
            "\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n",
            "\n___+\n", "\n\n", "\n", " ", ""
        ]
        self.knowledge_base = None
        self.vector_store = None
        self.llm = None
        self.reranker = None
        self.tokenizer = None
        
    def prepare_documents(self, dataset: HFDataset) -> List[LangchainDocument]:  # Updated type hint
        """Convert dataset to LangChain documents"""
        return [
            LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
            for doc in tqdm(dataset)
        ]

    # [Rest of the code remains the same...]
    
    def split_documents(self, documents: List[LangchainDocument]) -> List[LangchainDocument]:
        """Split documents into chunks"""
        tokenizer = AutoTokenizer.from_pretrained(self.config.embedding_model_name)
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
            separators=self.markdown_separators
        )
        
        docs_processed = []
        for doc in documents:
            docs_processed.extend(text_splitter.split_documents([doc]))
            
        # Remove duplicates
        unique_texts = {}
        docs_processed_unique = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)
                
        return docs_processed_unique
    
    def initialize_vector_store(self, documents: List[LangchainDocument]):
        """Initialize FAISS vector store with processed documents"""
        embedding_model = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model_name,
            model_kwargs={"device": self.config.device},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        self.vector_store = FAISS.from_documents(
            documents, embedding_model, distance_strategy=DistanceStrategy.COSINE
        )
    
    def initialize_llm(self, model_name: str):
        """Initialize the LLM for answering questions"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.llm = pipeline(
            model=model,
            tokenizer=self.tokenizer,
            task="text-generation",
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=500,
        )
        
        # Set up chat template
        prompt_template = [
            {
                "role": "system",
                "content": """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.""",
            },
            {
                "role": "user",
                "content": """Context:
{context}
---
Now here is the question you need to answer.

Question: {question}""",
            },
        ]
        
        self.tokenizer.chat_template = """
        {% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}
        {% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}
        {% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
        """
        
        self.prompt_template = self.tokenizer.apply_chat_template(
            prompt_template, tokenize=False, add_generation_prompt=True
        )
    
    def initialize_reranker(self):
        """Initialize the reranker model"""
        self.reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    
    def answer_question(
        self,
        question: str,
        num_retrieved_docs: int = 30,
        num_docs_final: int = 5
    ) -> Tuple[str, List[str]]:
        """Generate answer using RAG pipeline"""
        # Retrieve documents
        relevant_docs = self.vector_store.similarity_search(
            query=question, k=num_retrieved_docs
        )
        relevant_docs = [doc.page_content for doc in relevant_docs]
        
        # Rerank if reranker is available
        if self.reranker:
            relevant_docs = self.reranker.rerank(
                question, relevant_docs, k=num_docs_final
            )
            relevant_docs = [doc["content"] for doc in relevant_docs]
        
        relevant_docs = relevant_docs[:num_docs_final]
        
        # Build context and final prompt
        context = "\nExtracted documents:\n"
        context += "".join(
            [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)]
        )
        
        final_prompt = self.prompt_template.format(
            question=question, context=context
        )
        
        # Generate answer
        answer = self.llm(final_prompt)[0]["generated_text"]
        
        return answer, relevant_docs


# [Previous code for RAGConfig and RAGPipeline classes remains exactly the same]

def create_gradio_interface():
    """Create Gradio interface for the RAG pipeline"""
    
    def process_dataset(
        dataset_name: str,
        embedding_model: str,
        llm_model: str,
        use_reranker: bool,
        progress=gr.Progress(),
        state=None
    ):
        try:
            # Initialize RAG pipeline
            config = RAGConfig(embedding_model_name=embedding_model)
            pipeline = RAGPipeline(config)
            
            # Load and process dataset
            progress(0, desc="Loading dataset...")
            try:
                dataset = datasets.load_dataset(dataset_name, split="train")
            except Exception as e:
                return None, f"Error loading dataset: {str(e)}"
            
            progress(0.2, desc="Preparing documents...")
            documents = pipeline.prepare_documents(dataset)
            
            progress(0.4, desc="Splitting documents...")
            processed_docs = pipeline.split_documents(documents)
            
            progress(0.6, desc="Initializing vector store...")
            pipeline.initialize_vector_store(processed_docs)
            
            progress(0.8, desc="Initializing LLM...")
            try:
                pipeline.initialize_llm(llm_model)
            except Exception as e:
                return None, f"Error initializing LLM: {str(e)}"
            
            if use_reranker:
                progress(0.9, desc="Initializing reranker...")
                try:
                    pipeline.initialize_reranker()
                except Exception as e:
                    return None, f"Error initializing reranker: {str(e)}"
            
            progress(1.0, desc="Ready!")
            return pipeline, "RAG pipeline initialized successfully!"
            
        except Exception as e:
            return None, f"Error initializing pipeline: {str(e)}"
    
    def answer_query(query: str, state):
        if state is None:
            return "Please initialize the RAG pipeline first!", ""
        
        try:
            answer, sources = state.answer_question(query)
            sources_text = "\n\nSources:\n" + "\n\n".join(
                [f"Document {i}: {doc[:200]}..." for i, doc in enumerate(sources)]
            )
            return answer, sources_text
        except Exception as e:
            return f"Error generating answer: {str(e)}", ""
    
    # Create the interface
    with gr.Blocks() as demo:
        # Create state
        state = gr.State(None)
        
        gr.Markdown("# RAG Pipeline Interface")
        
        with gr.Tab("Setup"):
            dataset_input = gr.Textbox(
                label="Dataset Name (e.g., 'm-ric/huggingface_doc')",
                value="m-ric/huggingface_doc"
            )
            embedding_model_input = gr.Textbox(
                label="Embedding Model Name",
                value="thenlper/gte-small"
            )
            llm_model_input = gr.Textbox(
                label="LLM Model Name",
                value="meta-llama/Llama-2-7b-chat-hf"
            )
            use_reranker_input = gr.Checkbox(
                label="Use Reranker",
                value=True
            )
            setup_button = gr.Button("Initialize Pipeline")
            setup_output = gr.Textbox(label="Setup Status")
            
            setup_button.click(
                process_dataset,
                inputs=[
                    dataset_input,
                    embedding_model_input,
                    llm_model_input,
                    use_reranker_input,
                    state
                ],
                outputs=[state, setup_output]
            )
        
        with gr.Tab("Query"):
            query_input = gr.Textbox(label="Your Question")
            query_button = gr.Button("Get Answer")
            answer_output = gr.Textbox(label="Answer")
            sources_output = gr.Textbox(label="Source Documents")
            
            query_button.click(
                answer_query,
                inputs=[query_input, state],
                outputs=[answer_output, sources_output]
            )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()