"""
To make the RAG system usable in real applications, it needs to be exposed through an API. 
While the core RAG pipeline can retrieve documents and generate answers, an API is required 
to allow users or external services to interact with it in a structured way. FastAPI acts as 
a lightweight interface between the RAG system and its clients, enabling queries to be sent 
and responses to be returned via HTTP. Thanks to its asynchronous and high-performance design, 
FastAPI can handle multiple requests concurrently, which is essential for AI systems that 
perform retrieval and language generation. In this step, the RAG logic is integrated into an 
API layer, making the system scalable, accessible, and ready for real-world usage."""

from fastapi import FastAPI
from endpoints import router # this are the endpoints
app = FastAPI() # in this way we initialize the app that will handle incoming requests and route them to the appropriate endpoints
app.include_router(router) # we are adding all the endpoints to the FastAPI app