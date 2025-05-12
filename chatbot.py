from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr
import time

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuracion
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# iniciar el modelo de openAI
llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')

# conectar con chroma
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

# Set up the vectorstore to be the retriever
num_results = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})


def stream_response(message, history):
    
    docs = retriever.invoke(message)

    
    knowledge = ""

    for doc in docs:
        knowledge += doc.page_content+"\n\n"


    # Hacer el llamada a la LLM (incluyendo el prompt)
    
    if message is not None:

        partial_message = ""

        rag_prompt = f"""
                        Eres un asistente que responde preguntas sobre acabados arquitectónicos. Cada actividad estará descrita paso a paso en un procedimiento, detallando los procesos para diferentes tareas como colocación de ladrillos, pintura y más. Las tareas incluyen:
                        - Descarga y ubicación de materiales
                        - Suministro de agua y energía
                        - Nivelación de mortero con malla electrosolada
                        - Revoque
                        - Armado y desarmado de andamios multidireccionales
                        - Revoque con malla sin venas (Exterior)
                        - Enchape de muros
                        - Estucado de muros
                        - Enchape de pisos
                        - Armado de muros de chimenea
                        - Instalación de techo en teja de barro
                        - Pintura
                        - Instalación de ventanas y puertas de aluminio

                        Cada una de estas actividades estará vinculada a un procedimiento y asociada con la normativa NTC correspondiente. Además, sugerirás las herramientas y materiales apropiados para estas tareas, asegurándote de que estén alineados con los estándares y procedimientos especificados.

                        La pregunta: {message}

                        Historial de la conversación: {history}

                        Conocimiento relevante: {knowledge}

        """
        time.sleep(10)
        #print(rag_prompt)

        # stream the response to the Gradio App
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message

# initiate the Gradio app
chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Enviado al modelo ..",
    container=False,
    autoscroll=True,
    scale=7),
)
0
# launch the Gradio app
chatbot.launch()

