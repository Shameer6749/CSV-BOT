
import os
import numpy as np
import pandas as pd
import streamlit as st
import openai
from dotenv import load_dotenv
from streamlit_chat import message

# Load environment variables from .env file
load_dotenv()

# Hide traceback
st.set_option('client.showErrorDetails', False)

# Setting page title and header
st.set_page_config(page_title="CSV BOT", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>CSV BOT - Ask questions to your data</h1>", unsafe_allow_html=True)

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# Get the OpenAI API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key is None:
    st.warning("OpenAI API key is not found in the .env file. Please make sure it is set correctly.")
else:
    # Set the OpenAI API key as an environment variable
    os.environ["OPENAI_API_KEY"] = openai_api_key
    # Set the OpenAI API key directly
    openai.api_key = openai_api_key

    # Check if the API key is valid by making a simple API call
    try:
        models = openai.Model.list()
    except Exception as e:
        st.error("Error testing API key: {}".format(e))

# Allow user to upload CSV file
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # Read uploaded file as a Pandas DataFrame
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
    data_quality_check = st.checkbox('Request Data Quality Check')

    if data_quality_check:
        st.write("The following data quality analysis has been made")
        st.markdown("**1. The dataset column names have been checked for trailing spaces**")
        trailing_spaces = dataframe.columns[dataframe.columns.str.contains("\s+$", regex=True)]
        if trailing_spaces.empty:
            st.markdown('*Columns_ names_ are_ found_ ok*')
        else:
            st.markdown("*Columns with trailing spaces:* ")
            st.write(f"{', '.join(trailing_spaces)}")

        # Check data type of columns with name 'date'
        st.markdown("**2. The dataset's date columns have been checked for the correct data type**")
        date_cols = dataframe.select_dtypes(include="object").filter(regex="(?i)date").columns
        for col in date_cols:
            if pd.to_datetime(dataframe[col], errors="coerce").isna().sum() > 0:
                st.write(f"Column {col} should contain dates but has the wrong data type")
            else:
                st.write("Columns with date are of the correct data type")
        st.markdown("**:red[CSV BOT recommends fixing data quality issues prior to querying your data]**")

# Define function to generate response from user input
def generate_response(input_text):
    # Define the prompt for GPT-3.5
    prompt = f"You are given a user input: '{input_text}'. Please provide a one word response. Determine if the user inpit is asking for a "

    # Generate a response using GPT-3.5
    response = openai.Completion.create(
        engine="text-davinci-002",  # You can specify the engine you want to use
        prompt=prompt,
        max_tokens=50,  # You can adjust this based on the desired response length
        temperature=0.7,  # Adjust the temperature for response randomness
    )

    return response.choices[0].text

# container for chat history
response_container = st.container()

# container for text box
input_container = st.container()

with input_container:
    # Create a form for user input
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        # If the user submits input, generate a response and store input and response in session state variables
        try:
            # Generate the response using GPT-3.5
            query_response = generate_response(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(query_response)
        except Exception as e:
            st.error("An error occurred: {}".format(e))

if st.session_state['generated']:
    # Display chat history in a container
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))