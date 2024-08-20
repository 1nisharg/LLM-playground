import streamlit as st
from langchain_groq import ChatGroq
import time

# Set up the page configuration
st.set_page_config(
    page_title="LLM Playground",
    page_icon="🤖",
    layout="wide",
)

# Title of the app
st.title("LLM Playground🏌️")

# Sidebar for model selection
with st.sidebar:
    st.title("Choose your LLM")
    api_key = st.text_input("Enter your API key", type="password", value="enter your GROQ API Key")
    selected_llm = st.selectbox(
        "Select LLM Model:",
        (
            "llama-3.1-8b-instant", 
            "llama3-8b-8192",
            "llama3-70b-8192", 
            "mixtral-8x7b-32768", 
            "gemma-7b-it", 
            "gemma2-9b-it"
        )
    )
st.write("")
    st.write("")
    st.write("")
    st.sidebar.markdown("How to get GROQ API Key?")    
    st.sidebar.markdown("""
- Login/Signup https://console.groq.com/keys 
- Go to API Keys section
- Name your api key and make it yours!
NOTE: You won't be able to see your API key again. Make sure you store it somewhere.  

""")
# Input section for user prompt
prompt = st.text_area("Enter your prompt here:", height=7)

# Function to get LLM response with timing and token usage
def get_llm_response(model_name, prompt_text, key):
    llm = ChatGroq(model=model_name, api_key=key)
    messages = [{"role": "user", "content": prompt_text}]
    
    start_time = time.time()
    response = llm.invoke(messages)
    end_time = time.time()
    
    inference_time = end_time - start_time
    
    # Extract token usage from the response
    tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else "N/A"
    
    return response.content, inference_time, tokens_used

# Response generation
if st.button("Generate Response"):
    if prompt and api_key:
        with st.spinner(f"Generating response using {selected_llm}..."):
            try:
                # Generate response
                response, inference_time, tokens_used = get_llm_response(selected_llm, prompt, api_key)

                # Display the response
                st.subheader(f"Response from {selected_llm}:")
                st.write(response)
                
                # Model statistics (real-time info)
                st.subheader("Model Statistics")
                st.write(f"Inference Time: {inference_time:.2f}s")
                st.write(f"Tokens Used: {tokens_used}")
                st.write(f"Model: {selected_llm}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error(f"Error type: {type(e)}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
    else:
        st.error("Please enter a prompt and API key.")

# Customization and layout enhancements
st.write("---")
st.subheader("Compare Responses")
cols = st.columns(2)

with cols[0]:
    model1 = st.selectbox("Select Model 1", [ "llama-3.1-8b-instant", "llama3-8b-8192", "llama3-70b-8192"])
    if st.button("Generate for Model 1"):
        response1, inference_time1, tokens_used1 = get_llm_response(model1, prompt, api_key)
        st.write(response1)
        st.write(f"Inference Time: {inference_time1:.2f}s")
        st.write(f"Tokens Used: {tokens_used1}")

with cols[1]:
    model2 = st.selectbox("Select Model 2", ["mixtral-8x7b-32768", "gemma-7b-it", "gemma2-9b-it"])
    if st.button("Generate for Model 2"):
        response2, inference_time2, tokens_used2 = get_llm_response(model2, prompt, api_key)
        st.write(response2)
        st.write(f"Inference Time: {inference_time2:.2f}s")
        st.write(f"Tokens Used: {tokens_used2}")

# Footer
st.write("---")
st.markdown("**LLM Playground** © 2024 OpenRAG")
