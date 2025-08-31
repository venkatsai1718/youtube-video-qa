import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

key = st.secrets["Y_key"]

# llm model
llm = ChatGoogleGenerativeAI(api_key=key, model='gemini-1.5-flash', temparature=0.2)
embeddings = GoogleGenerativeAIEmbeddings(google_api_key=key, model="models/embedding-001")

def gettranscript(link):
    # Getting Transcript
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(link)
        for entry in transcript_list[:5]:
            print(entry)
    except TranscriptsDisabled:
        print("Transcripts are disabled for this video.")
                
    transcript = " ".join(chunk.text for chunk in transcript_list)
    return transcript

def getcontext(text):
    # Text Splitting
    splitter = RecursiveCharacterTextSplitter(
                chunk_size=100,
                chunk_overlap=20
            )
    chunks = splitter.create_documents([text])
            
    # Embedding
    vector_store = FAISS.from_documents(chunks, embeddings)
            
    # Retriever
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})
    
    return retriever

def getprompt():
            
    prompt = PromptTemplate(
    template="""
        You are a helpful assitant, Answer the Question: \n {question} \n based on provided text: \n
        {context}

        """,
        input_variables=['context', 'question']
    )
    
    return prompt

video_id = None


# Streamlit UI 

st.title("Insert YouTube Video Link")

# Input for YouTube link
youtube_link = st.text_input("Enter YouTube Link:")
if youtube_link:
    if "youtube.com" in youtube_link:
        parsed_url = urlparse(youtube_link)
        query_params = parse_qs(parsed_url.query)
        video_id = query_params.get("v", [None])[0]
    elif "youtu.be" in youtube_link:
        parsed_url = urlparse(youtube_link)
        video_id = parsed_url.path.lstrip("/")
            
    st.subheader('Input Video')
    st.markdown(f"""
            <iframe width="640" height="360" 
            src="{youtube_link.replace("watch?v=", "embed/")}" 
            frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
            allowfullscreen></iframe>
            """,
            unsafe_allow_html=True,
        )
                
        
    user_text = st.text_area("Ask a Question:")

    # Button to send
    if st.button("Send"):
        if youtube_link and video_id:
            transcript = gettranscript(video_id)
            retriever = getcontext(transcript)
            prompt = getprompt()
                        
            format_docs = lambda x: "\n\n".join(doc.page_content for doc in x)

            parallel_chain = RunnableParallel({
                        'context': retriever | RunnableLambda(format_docs),
                        'question': RunnablePassthrough()
                        })
            parser = StrOutputParser()

            main_chain = parallel_chain | prompt | llm | parser

            result = main_chain.invoke(user_text)
            st.info("ℹ️ Answer is purely based on provided youtube video.")  

            st.write(result)
                
        else:

            st.error("Please check the link.")




