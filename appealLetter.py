import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import re
from datetime import datetime
from io import BytesIO
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    # Clean up text
    text = re.sub(r'\s+', ' ', text)  # Replace all whitespace (including newlines) with a single space
    text = re.sub(r'([A-Za-z])\s([A-Za-z])', r'\1\2', text)  # Join fragmented letters
    text = text.replace("A m o u n t", "Amount").replace("P a t i e n t", "Patient")  # Fix known patterns
    return text.strip()

# Function to extract claims using regex
def extract_claims(text, pattern):
    return re.findall(pattern, text, re.DOTALL)

# Extract patient details
def extract_patient_info(medical_text):
    name = re.search(r"Patient Name:\s*(.*)", medical_text)
    return name.group(1) if name else "Unknown Patient"

# Initialize AI agent
def initialize_agent(api_key):
    try:
        llm = ChatOpenAI(temperature=0.7, model="gpt-4", openai_api_key=api_key)
        memory = ConversationBufferMemory()
        return ConversationChain(llm=llm, memory=memory)
    except Exception as e:
        st.error(f"Error initializing AI agent: {e}")
        return None

# Function to check "Claim Submitted Late"
def is_claim_late(claim_date, denial_date, max_days=90):
    try:
        claim_date_obj = datetime.strptime(claim_date, "%Y-%m-%d")
        denial_date_obj = datetime.strptime(denial_date, "%Y-%m-%d")
        return (denial_date_obj - claim_date_obj).days > max_days
    except ValueError:
        return False

# Function to check "Service Not Covered"
def is_service_not_covered(service_desc, non_covered_services):
    return any(nc.lower() in service_desc.lower() for nc in non_covered_services)

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
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# File upload
st.header("Upload Documents")
eob_file = st.file_uploader("Upload Explanation of Benefits (EOB)", type=["pdf"])
medical_file = st.file_uploader("Upload Medical Records", type=["pdf"])
denial_file = st.file_uploader("Upload Denial Letter", type=["pdf"])

if eob_file and medical_file and denial_file:
    eob_text = extract_text_from_pdf(eob_file)
    medical_text = extract_text_from_pdf(medical_file)
    denial_text = extract_text_from_pdf(denial_file)

    st.subheader("File Previews")
    st.write("**EOB Preview:**")
    st.text(eob_text[:1000])  # Show first 1000 characters
    st.write("**Medical Records Preview:**")
    st.text(medical_text[:1000])
    st.write("**Denial Letter Preview:**")
    st.text(denial_text[:1000])

    # Extract claims
    claim_pattern = r"Claim Number:\s*(\d+).*?Service Description:\s*(.*?)\nAmount Billed:\s*\$([\d,.]+).*?Claim Date:\s*(\d{4}-\d{2}-\d{2})"
    denial_pattern = r"Claim Number:\s*(\d+).*?Reason for Denial:\s*(.*?)(?=\nClaim Number:|\n$)"

    eob_claims = extract_claims(eob_text, claim_pattern)
    denial_claims = extract_claims(denial_text, denial_pattern)

    # Define non-covered services
    non_covered_services = ["cosmetic surgery", "experimental treatment", "unnecessary procedure"]

    # Process claims
    results = []
    appeal_letters = {}

    for claim_number, service_desc, billed_amt, claim_date in eob_claims:
        denial_match = next((d for d in denial_claims if d[0] == claim_number), None)
        denial_reason = denial_match[1] if denial_match else "No Denial Reason Found"

        if is_claim_late(claim_date, datetime.now().strftime("%Y-%m-%d")):
            results.append({
                "Customer Name": extract_patient_info(medical_text),
                "Claim Number": claim_number,
                "Appeal Letter Sent": "No",
                "Reason": "Claim Submitted Late"
            })
        elif is_service_not_covered(service_desc, non_covered_services):
            results.append({
                "Customer Name": extract_patient_info(medical_text),
                "Claim Number": claim_number,
                "Appeal Letter Sent": "No",
                "Reason": "Service Not Covered"
            })
        else:
            appeal_prompt = f"""
            Generate a professional appeal letter based on these inputs:
            1. Explanation of Benefits (EOB):
            {eob_text}

            2. Medical Records:
            {medical_text}

            3. Denial Letter:
            {denial_text}

            The appeal letter should:
            - Use a polite and professional tone.
            - Clearly state the reason for the appeal.
            - Explain the medical necessity of the procedures.
            - Suggest why the denial reason should be reconsidered.

            Please use the following patient details at the beginning of the letter:
            Patient Name: {extract_patient_info(medical_text)}

            Start the letter with the patient's full name and address, followed by the current date ({current_date}).
            """
            

            agent = initialize_agent(api_key)
            if agent:
                try:
                    appeal_letter = agent.run(appeal_prompt)
                    appeal_letters[claim_number] = appeal_letter
                    results.append({
                        "Customer Name": extract_patient_info(medical_text),
                        "Claim Number": claim_number,
                        "Appeal Letter Sent": "Yes",
                        "Reason": "",
                    })
                except Exception as e:
                    results.append({
                        "Customer Name": extract_patient_info(medical_text),
                        "Claim Number": claim_number,
                        "Appeal Letter Sent": "No",
                        "Reason": f"Error: {str(e)}",
                    })

    # Results table
    results_df = pd.DataFrame(results)
    st.subheader("Claim Results")
    st.dataframe(results_df)

    # Download CSV
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="claim_results.csv",
        mime="text/csv",
    )

    # Download appeal letters
    for claim_number, appeal_letter in appeal_letters.items():
        appeal_file = BytesIO()
        appeal_file.write(appeal_letter.encode("utf-8"))
        appeal_file.seek(0)
        st.download_button(
            label=f"Download Appeal Letter for Claim {claim_number}",
            data=appeal_file,
            file_name=f"AppealLetter_{claim_number}.txt",
            mime="text/plain",
        )
else:
    st.error("Please upload all required files.")
