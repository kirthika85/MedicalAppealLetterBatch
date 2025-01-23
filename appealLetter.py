import streamlit as st
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import pandas as pd
import re
from datetime import datetime
from io import BytesIO

# Function to extract text from uploaded PDFs
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to extract claims from a document based on regex pattern
def extract_claims(text, pattern):
    claims = re.findall(pattern, text, re.DOTALL)
    return claims

# Function to extract patient details from the medical records text
def extract_patient_info(medical_text):
    name = re.search(r"Patient Name:\s*(.*)", medical_text)
    address = re.search(r"Address:\s*(.*)", medical_text)
    return {
        "name": name.group(1) if name else "[Patient Name]",
        "address": address.group(1) if address else "[Patient Address]"
    }

# Initialize GPT-4 Chat Model with LangChain
def initialize_agent(api_key):
    try:
        llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4",
            openai_api_key=api_key
        )
        memory = ConversationBufferMemory()
        return ConversationChain(llm=llm, memory=memory)
    except Exception as e:
        st.error(f"Error initializing OpenAI agent: {e}")
        return None

# Function to check if the claim is within the appealable timeframe
def is_claim_within_timeframe(claim_date, denial_date, max_days=30):
    try:
        claim_date_obj = datetime.strptime(claim_date, "%Y-%m-%d")
        denial_date_obj = datetime.strptime(denial_date, "%Y-%m-%d")
        return (denial_date_obj - claim_date_obj).days <= max_days
    except ValueError:
        return False

# Streamlit setup
st.set_page_config(page_title="Medical Claim Appeal Generator", page_icon="🩺", layout="wide")
col1, col2 = st.columns([1, 6])
with col1:
    st.image("Mool.png", width=150)
with col2:
    st.markdown("<h1 style='margin-top: 10px;'>Medical Claim Appeal Generator</h1>", unsafe_allow_html=True)

st.write("Generate medical claim appeal letters for each claim in the provided documents.")
current_date = datetime.now().strftime("%A, %B %d, %Y")

# Sidebar for OpenAI API Key input
api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key",
    type="password",
    help="Your API key is required to use GPT-4 for generating appeal letters."
)

# Upload documents
st.header("Upload Documents")
eob_file = st.file_uploader("Upload the Explanation of Benefits (EOB)", type=["pdf"])
medical_file = st.file_uploader("Upload the Medical Records", type=["pdf"])
denial_file = st.file_uploader("Upload the Denial Letter", type=["pdf"])

if eob_file and medical_file and denial_file:
    # Extract text from uploaded files
    eob_text = extract_text_from_pdf(eob_file)
    medical_text = extract_text_from_pdf(medical_file)
    denial_text = extract_text_from_pdf(denial_file)

    # Extract claims using patterns
    claim_pattern = r"Claim Number:\s*(\d+).*?Service Description:\s*(.*?)\nAmount Billed:\s*\$([\d,.]+).*?Claim Date:\s*(\d{4}-\d{2}-\d{2})"
    denial_pattern = r"Claim Number:\s*(\d+).*?Reason for Denial:\s*(.*?)(?=\nClaim Number:|\n$)"
    
    eob_claims = extract_claims(eob_text, claim_pattern)
    denial_claims = extract_claims(denial_text, denial_pattern)

    st.write("EOB Claims Extracted:", eob_claims)
    st.write("Denial Claims Extracted:", denial_claims)

    # Match claims and process appeals
    results = []
    appeal_letters = {}
    processed_claims = set()

    for claim_number, service_desc, billed_amt, claim_date in eob_claims:
        if claim_number in processed_claims:
            continue

        denial_match = next((d for d in denial_claims if d[0] == claim_number), None)

        if denial_match:
            reason_for_denial = denial_match[1]
            if is_claim_within_timeframe(claim_date, current_date):
                # Appeal Prompt
                appeal_prompt = f"""
                Validate the claim based on these inputs:
                1. Explanation of Benefits (EOB): Service - {service_desc}, Amount Billed - {billed_amt}
                2. Medical Records: {medical_text}
                3. Denial Letter: {reason_for_denial}

                If valid, generate a professional appeal letter:
                - Use a polite and professional tone.
                - Clearly state the reason for the appeal.
                - Explain the medical necessity of the procedures.
                - Suggest why the denial reason should be reconsidered.

                Patient Name: {extract_patient_info(medical_text)['name']}
                Current Date: {current_date}
                """

                try:
                    agent = initialize_agent(api_key)
                    if agent:
                        appeal_letter = agent.run(appeal_prompt)
                        appeal_letters[claim_number] = appeal_letter
                        results.append({
                            "Claim Number": claim_number,
                            "Patient Name": extract_patient_info(medical_text)['name'],
                            "Appeal Letter Sent": "Yes",
                            "Reason": ""
                        })
                except Exception as e:
                    results.append({
                        "Claim Number": claim_number,
                        "Patient Name": extract_patient_info(medical_text)['name'],
                        "Appeal Letter Sent": "No",
                        "Reason": f"Error: {str(e)}"
                    })
            else:
                results.append({
                    "Claim Number": claim_number,
                    "Patient Name": extract_patient_info(medical_text)['name'],
                    "Appeal Letter Sent": "No",
                    "Reason": "Claim outside appealable timeframe"
                })
        else:
            results.append({
                "Claim Number": claim_number,
                "Patient Name": extract_patient_info(medical_text)['name'],
                "Appeal Letter Sent": "No",
                "Reason": "No Matching Denial Found"
            })
        processed_claims.add(claim_number)

    # Display Results as a Table
    st.subheader("Claim Processing Results")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # Download Results as CSV
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="claim_processing_results.csv",
        mime="text/csv"
    )

    # Download Appeal Letters
    for claim_number, appeal_letter in appeal_letters.items():
        appeal_file = BytesIO()
        appeal_file.write(appeal_letter.encode("utf-8"))
        appeal_file.seek(0)
        st.download_button(
            label=f"Download Appeal Letter for Claim {claim_number}",
            data=appeal_file,
            file_name=f"appeal_letter_claim_{claim_number}.txt",
            mime="text/plain"
        )
else:
    st.error("Please upload all required documents.")
