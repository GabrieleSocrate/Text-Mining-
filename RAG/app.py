import gradio as gr

from RAG import rag_answer

def ask(question):
    # rag_answer is defined as async, so it cannot be called directly from a normal function.
    # asyncio.run() runs the async function to completio returning the result.
    return rag_answer(question)

gr.Interface(fn=ask, inputs="text", outputs="text").launch()