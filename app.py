import gradio as gr
import asyncio
from RAG_dataset import rag_answer

def ask(question):
    # rag_answer is defined as async, so it cannot be called directly from a normal function.
    # asyncio.run() acts as a bridge: it starts an event loop and runs the async function to completion,
    # returning the result as if it were a regular synchronous call.
    return asyncio.run(rag_answer(question))

gr.Interface(fn=ask, inputs="text", outputs="text").launch()