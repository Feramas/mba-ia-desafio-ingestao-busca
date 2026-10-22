import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogle
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_documents")
SEARCH_K = int(os.getenv("SEARCH_K", 10))

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

_embeddings = None
_vector_store = None
_chain = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        if LLM_PROVIDER == "google":
            api_key = os.getenv("GOOGLE_API_KEY")
            model = os.getenv("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set")
            _embeddings = GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            _embeddings = OpenAIEmbeddings(model=model, api_key=api_key)
    return _embeddings


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        embeddings = get_embeddings()
        _vector_store = PGVector(
            collection_name=COLLECTION_NAME,
            connection_string=DATABASE_URL,
            embedding_function=embeddings
        )
    return _vector_store


def get_llm():
    if LLM_PROVIDER == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        model = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.0-flash")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        return ChatGoogle(model=model, google_api_key=api_key)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return ChatOpenAI(model=model, api_key=api_key, temperature=0)


def search_prompt(question=None):
    global _chain

    if _chain is None:
        llm = get_llm()
        prompt = PromptTemplate(
            input_variables=["contexto", "pergunta"],
            template=PROMPT_TEMPLATE
        )
        _chain = LLMChain(llm=llm, prompt=prompt)

    if question is None:
        return _chain

    vector_store = get_vector_store()
    docs = vector_store.similarity_search(question, k=SEARCH_K)

    if not docs:
        context = ""
    else:
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    response = _chain.run(contexto=context, pergunta=question)
    return response.strip()