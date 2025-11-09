import gradio as gr
from langchain_groq import ChatGroq
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

# --- Initialize LLM ---
groq_api_key = "your_groq_api_key"
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# --- Define State ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

# --- Build Graph ---
graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": llm.invoke(state["messages"])}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# --- Chat Function for Gradio ---
def chat_with_bot(user_input, history):
    """
    This function will handle conversation and maintain chat history.
    """
    state = {"messages": ("user", user_input)}
    output = None
    for event in graph.stream(state):
        for value in event.values():
            output = value["messages"].content  # get assistant message

    history.append((user_input, output))
    return history, history

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown("## 🤖 LangGraph + Groq Chatbot (llama-3.1-8b-instant)")
    chatbot_ui = gr.Chatbot(label="Chat with your AI Assistant")
    user_input = gr.Textbox(label="Type your message:", placeholder="Ask me anything...")
    clear = gr.ClearButton([chatbot_ui, user_input])

    # When user submits input
    user_input.submit(chat_with_bot, [user_input, chatbot_ui], [chatbot_ui, chatbot_ui])

# --- Launch App ---
demo.launch(share=True)
