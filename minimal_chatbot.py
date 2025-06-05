#pip freeze > requirements.txt; gather all dependencies
import streamlit as st
import os
import google.generativeai as genai #set up api key
from google.ai import generativelanguage as gl
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import ResourceExhausted

st.title("Minimal Chat App with Google Gemini")
st.markdown("Enter your Google API Key in the sidebar to use the chatbot.")

#store API key securely across reruns
if "api_key_configured" not in st.session_state:
    st.session_state.api_key_configured = False
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "available_models" not in st.session_state:
    st.session_state.available_models = []

# Your Google Cloud API key or Service Account JSON path
google_api_key = st.sidebar.text_input(
    "Personal Google API Key", 
    type="password",
    value=st.session_state.api_key or os.getenv("GOOGLE_API_KEY", ""),
    help="You can get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).",
    key="api_key_text_input"
)

configure_button=st.button("Configure API Key") 

if google_api_key and configure_button:
    try:
        genai.configure(api_key=google_api_key)
        # Test the configuration by trying to list models
        list(genai.list_models()) # This will raise an exception if the key is bad
        st.session_state.api_key = google_api_key
        st.session_state.api_key_configured = True
        st.success("API Key configured successfully! You can now chat.")
        # Rerun to update UI and show chat interface
        st.rerun()
    except Exception as e:
        st.error(f"Error configuring API Key: {e}. Please check your key.")
        st.session_state.api_key_configured = False
elif configure_button and not google_api_key:
    st.warning("Please enter your API Key.")

# --- 2. Chat Interface (only visible if API key is configured) ---
if st.session_state.api_key_configured:
    st.markdown("---") # Separator

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Model selection (now part of the visible UI)
    model_name = st.selectbox(
        "Choose a Gemini Model:",
        options=[
            "gemini-pro",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemma-3-2b-it", 
        ],
        help="Select the generative AI model to use for the chat.",
        key="model_selector"
    )

    # Initialize the generative model
    @st.cache_resource
    def get_model(model_name_param): # Use a parameter to distinguish from session_state
        # Ensure genai is configured with the current API key in session state
        # This is crucial for cached resources
        genai.configure(api_key=st.session_state.api_key)
        return genai.GenerativeModel(model_name_param)

    model = get_model(model_name)

    # Accept user input
    if prompt := st.chat_input("Ask Gemini anything...", disabled=not st.session_state.api_key_configured):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = model.generate_content(prompt)
                    full_response = response.text
                except Exception as e:
                    full_response = f"An error occurred: {e}. Your API key might be invalid or you've hit a quota. Please re-enter your key or try a different one."
                    st.error(full_response)
                    # Optionally, you might want to remove the last user message from history
                    # if the generation failed completely to avoid confusion.
                    # st.session_state.messages.pop()

                st.markdown(full_response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.info("Please enter your API key and click 'Configure API Key' to enable the chatbot.")

