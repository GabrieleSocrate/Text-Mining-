from fastapi import APIRouter, HTTPException
from RAG import rag_answer
router = APIRouter() # here there are all the endpoints. It acts as a container where routes are registered and later attached to the main FastAPI app.

@router.get("/query/") 
# The decorator @router.get("/query/") defines an API endpoint and associates it with a specific URL path. 
# The string "/query/" represents the address that users must call to interact with the API. 
# When an HTTP GET request is sent to this path, FastAPI automatically executes the function defined below, 
# allowing clients to send a query and receive a response from the RAG system.
async def query_rag_system(query):
    """
    The endpoint function (query_rag_system) is asynchronous because FastAPI is built
    to efficiently handle multiple requests at the same time. When a request arrives, 
    the server may need to wait for slow operations, such as network calls or heavy 
    processing. By declaring the endpoint as async, FastAPI does not block the server 
    while waiting for a response. Instead, the server can continue accepting and processing 
    other incoming requests concurrently. This makes the API more responsive and allows it 
    to scale better when several users query the system at the same time."""
    try:
        response = await rag_answer(query)
        """
        The keyword await is what actually executes an asynchronous function and retrieves its result.
        When the endpoint calls await rag_answer(query), it means: “run this asynchronous operation 
        and wait for its final result, without blocking the server while waiting.”
        If await is not used, the asynchronous function is not executed to completion and no response is produced. 
        Therefore, await is required both to obtain the real output of the function and to allow the 
        server to handle other requests concurrently while waiting."""
        return {"query": query, "response": response}
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))