import os
import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
# from langchain.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory


# set google API key to environment
os.environ["GOOGLE_API_KEY"] = st.secrets["google-api-key"]

class YTask:
    def __init__(self, url):
        self.url = url
        self.embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = GoogleGenerativeAI(model="models/gemini-pro", temperature=0)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    def load_yt_transcript(self):
        loader = YoutubeLoader.from_youtube_url(youtube_url=self.url, add_video_info=True)
        data = loader.load()
        
        return data
    
    def split_and_embed(self):
        data = self.load_yt_transcript()
        # st.write(data)
        # splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        # docs = splitter.split_text(data)
        
        vector_db = Chroma.from_documents(data, embedding=self.embedding)
        return vector_db.as_retriever()
    
    def process_query(self, query):
        retriever = self.split_and_embed()
        chat_bot = ConversationalRetrievalChain.from_llm(self.llm, retriever, memory=self.memory, verbose=False)
        
        response = chat_bot.invoke({"question" : query})
        return response["answer"]
        

if __name__ == "__main__":
    st.title("Chat With Youtube Video")
    with st.sidebar:
        url = st.text_input("Provide youtube video link")
    if url:
        chat_instance = YTask(url)
              
        chat_instance = YTask(url)
        chat_instance.split_and_embed()
        st.success("Video data loaded, proceed to chats")
            
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display for all the messages
        for message, kind in st.session_state.messages:
                with st.chat_message(kind):
                    st.markdown(message)
                    
        prompt = st.chat_input("Ask your questions ...")
            
        if prompt:
            # Handling prompts and rendering to the chat interface
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append([prompt, "user"]) # updating the list of prompts 


            with st.spinner("Generating response"):
                answer = chat_instance.process_query(prompt)
                if answer:
                    st.chat_message("ai").markdown(answer)
                    st.session_state.messages.append([answer, "ai"])
    else:
        st.error("Youtube video link not added yet")   
     
    
            
        
        
    
