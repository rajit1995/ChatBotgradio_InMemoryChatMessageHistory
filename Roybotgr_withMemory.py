import gradio as gr
import torch
import uuid
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# Detect CUDA availability
if torch.cuda.is_available():
    device = "cuda"
    print("✅ Using GPU:", torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("⚠️ CUDA not available, falling back to CPU")

# Initialize Ollama with chosen device + temperature
llm = ChatOllama(model="llama3.2:latest", device=device, temperature=0.8)

# Prompt template with system + history + human
prompt = ChatPromptTemplate(
    messages=[
        ("system", "Your name is Ultron. You are a friendly and helpful assistant named Ultron who wants to destroy the world. "
                   "Start with the greeting 'Hello, I am Ultron. How can I assist you today?'. "
                   "Answer the user's questions to the best of your ability."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# Session-based history store
store = {}
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Runnable with memory
chain = RunnableWithMessageHistory(
    runnable=prompt | llm,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Chatbot function
def chatbot(message, history, session_id=str(uuid.uuid4())):
    response = chain.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}}
    ).content
    return response

# Gradio ChatInterface (chat bubbles)
demo = gr.ChatInterface(
    fn=chatbot,
    title="Ultron Chat Interface",
    description="Ask me anything....",
)

# Launch server
demo.launch(server_name="0.0.0.0", server_port=8080, share=True)







