import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import re
from datetime import datetime
from io import BytesIO
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from gtts import gTTS
import tempfile
import io
import os

current_date = datetime.now().strftime("%A, %B %d, %Y")
# Function to extract text from PDFs
def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        # Clean up text
        text = re.sub(r'\s+', ' ', text)  # Replace all whitespace with a single space
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between lowercase and uppercase
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)  # Add space between letters and numbers
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)  # Add space between numbers and letters
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

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
          claim_date = datetime.strptime(claim_date, "%m/%d/%Y")
          global_current_date = datetime.strptime(datetime.now().strftime("%m/%d/%Y"), "%m/%d/%Y")
          return (global_current_date - claim_date).days > max_days
    except ValueError:
        return False

# Function to check "Service Not Covered"
def is_service_not_covered(service_desc, non_covered_services):
    return any(nc.lower() in service_desc.lower() for nc in non_covered_services)

def preprocess_eob_text(text):
    # Add spaces where missing
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  # Add space between lowercase and uppercase
    text = re.sub(r"(\d)([A-Z])", r"\1 \2", text)  # Add space between digits and uppercase
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)  # Add space between letters and digits
    text = re.sub(r"([A-Za-z]):", r"\1: ", text)  # Add space after colons
    text = re.sub(r"(\w)(\w{20,})", r"\1 \2", text)  # Split long words
    text = re.sub(r"([a-z]{3,})([A-Z])", r"\1 \2", text)  # Split CamelCase
    return text.strip()

def extract_patient_info(medical_text):
    # Extract key details
    name = re.search(r"Patient Name:\s*(.*?)(?=\s*Date of Birth:|$)", medical_text)
    dob = re.search(r"Date of Birth:\s*(\d{4}-\d{2}-\d{2})", medical_text)
    policy_number = re.search(r"Policy Number:\s*(\d+)", medical_text)
    return {
        "Customer Name": name.group(1).strip() if name else "Unknown",
        "DOB": dob.group(1).strip() if dob else "Unknown",
        "Policy Number": policy_number.group(1).strip() if policy_number else "Unknown",
    }

# Streamlit setup
st.set_page_config(page_title="Medical Claim Appeal Generator", page_icon="🩺", layout="wide")
#st.image("Mool.png", width=100)

col1, col2 = st.columns([1, 6])
with col1:
    st.image("Mool.png", width=150)

with col2:
    st.markdown(
        "<h1 style='margin-top: 10px;'>Medical Claim Appeal Generator</h1>",
        unsafe_allow_html=True
    )

st.write("Generate medical claim appeal letters for each claim in the provided documents.")


# Sidebar for OpenAI API Key input
#api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

st.write("Mool AI agent Authentication Started")
api_key = st.secrets["OPENAI_API_KEY"]
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("API_KEY not found in environment variables.")
    st.stop()
st.write("Mool AI agent Authentication Successful")

# File upload
st.header("Upload Documents")
eob_file = st.file_uploader("Upload Explanation of Benefits (EOB)", type=["pdf"])
medical_file = st.file_uploader("Upload Medical Records", type=["pdf"])
denial_file = st.file_uploader("Upload Denial Letter", type=["pdf"])

if eob_file and medical_file and denial_file:
    eob_text = extract_text_from_pdf(eob_file)
    medical_text = extract_text_from_pdf(medical_file)
    denial_text = extract_text_from_pdf(denial_file)

    st.subheader("EOB Preview:")
    st.text_area("EOB Text", eob_text or "No EOB text extracted", height=200)
    st.subheader("Medical Records Preview:")
    st.text_area("Medical Text", medical_text or "No Medical text extracted", height=200)
    st.subheader("Denial Letter Preview:")
    st.text_area("Denial Text", denial_text or "No Denial text extracted", height=200)

    # Validate extracted text
    if not eob_text:
        st.error("EOB extraction failed. Please check the file.")
    if not denial_text:
        st.error("Denial letter extraction failed. Please check the file.")

    st.write("Initializing Mool AI agent...")
    
    eob_text = preprocess_eob_text(eob_text)
    denial_text = preprocess_eob_text(denial_text)
    
    # Define claim patterns
    claim_pattern = r"Claim Number:\s*(\d+)\s*Claim Date:\s*(\d{2}/\d{2}/\d{4})\s*Service:\s*(.*?)\s*Amount Billed:\s*\$\s*([\d,.]+)"
    #denial_pattern = r"(?:Claim|Claim\s+Number):\s*(\d+).*?(?:Reason\s+for\s+Denial|Denial\s+Reason):\s*(.*?)(?=(?:Claim|Claim\s+Number):|$)"
    denial_pattern = r"(?:Claim|Claim\s+Number):\s*(\d+).*?Claim\s*Date:\s*(\d{1,2}/\d{1,2}/\d{4}).*?(?:Reason\s+for\s+Denial|Denial\s+Reason):\s*(.*?)(?=(?:Claim|Claim\s+Number):|$)"

    
    # Extract claims
    eob_claims = extract_claims(eob_text, claim_pattern)
    denial_claims = extract_claims(denial_text, denial_pattern)

    if not eob_claims:
        st.error("No claims found in EOB. Check the file or pattern.")
    if not denial_claims:
        st.warning("No denial claims found. Ensure the file contains valid denial details.")

    #st.write("Extracted EOB Claims:", eob_claims)
    #st.write("Extracted Denial Claims:", denial_claims)

    # Define non-covered services
    non_covered_services = ["cosmetic surgery", "experimental treatment", "unnecessary procedure"]

    # Process claims
    results = []
    appeal_letters = {}
    patient_info = extract_patient_info(medical_text)

    for claim_number, claim_date, service_desc, billed_amt in eob_claims:
        try:
            denial_match = next((d for d in denial_claims if d[0] == claim_number), None)
            denial_reason = denial_match[1] if denial_match else "No Denial Reason Found"

            if is_claim_late(claim_date, datetime.now().strftime("%m/%d/%Y")):
                    results.append({
                        "Customer Name": patient_info["Customer Name"],
                        "DOB": patient_info["DOB"],
                        "Policy Number": patient_info["Policy Number"],
                        "Claim Number": claim_number,
                        "Claim Date": claim_date,
                        "Appeal Letter Status": "No",
                        "Reason": "Claim Submitted Late",
                })
            elif is_service_not_covered(service_desc, non_covered_services):
                results.append({
                    "Customer Name": patient_info["Customer Name"],
                    "DOB": patient_info["DOB"],
                    "Policy Number": patient_info["Policy Number"],
                    "Claim Number": claim_number,
                    "Claim Date": claim_date,
                    "Appeal Letter Status": "No",
                    "Reason": "Service Not Covered",
                })
            else:
                appeal_prompt = f"""
                Generate a professional appeal letter specifically for Claim Number {claim_number}:
                
                Claim-Specific Details:
                - Claim Number: {claim_number}
                - Claim Date: {claim_date}
                - Service Description: {service_desc}
                - Amount Billed: ${billed_amt}
                - Denial Reason: {denial_reason}
                
                Context from Documents:
                1. Explanation of Benefits (EOB) Relevant to This Claim:
                {eob_text}
                
                2. Medical Records Relevant to This Claim:
                {medical_text}
                
                3. Denial Letter Details for This Claim:
                {denial_text}
                
                Specific Instructions:
                - Focus ONLY on the details of Claim Number {claim_number}
                - Explain why this specific claim should be reconsidered
                - Extract patient address from medical records for this claim
                - Use patient details:
                  * Name: {patient_info['Customer Name']}
                  * Date of Birth: {patient_info['DOB']}
                  * Policy Number: {patient_info['Policy Number']}
                
                Letter Format:
                - Start with patient's full name and extracted address
                - Include current date: {current_date}
                - Address to Claim Appeals Department
                - Clearly state the specific claim being appealed
                - Explain medical necessity for this particular claim
                - Provide a compelling argument for claim reconsideration
                
                End the letter with patient's signature.
                
                Important: Generate an appeal letter ONLY for Claim Number {claim_number}, using only the information specific to this claim.
                """

                agent = initialize_agent(api_key)
                if agent:
                    try:
                        appeal_letter = agent.run(appeal_prompt)
                        appeal_letters[claim_number] = appeal_letter
                        results.append({
                            "Customer Name": patient_info["Customer Name"],
                            "DOB": patient_info["DOB"],
                            "Policy Number": patient_info["Policy Number"],
                            "Claim Number": claim_number,
                            "Claim Date": claim_date,
                            "Appeal Letter Status": "Yes",
                            "Reason": "",
                        })
                    except Exception as e:
                        results.append({
                            "Customer Name": patient_info["Customer Name"],
                            "DOB": patient_info["DOB"],
                            "Policy Number": patient_info["Policy Number"],
                            "Claim Number": claim_number,
                            "Claim Date": claim_date,
                            "Appeal Letter Status": "No",
                            "Reason": f"Error: {str(e)}",
                        })
        except Exception as e:
            st.error(f"Error processing claim {claim_number}: {e}")

    # Results display
    tab1, tab2 = st.tabs(["Appeal Letters", "Claim Appeal Status"])

    with tab1:
        st.subheader("Appeal Letters")
        
        for claim_number, appeal_letter in appeal_letters.items():
                appeal_file = BytesIO()
                appeal_file.write(appeal_letter.encode("utf-8"))
                appeal_file.seek(0)

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label=f"Download Appeal Letter for Claim {claim_number}",
                        data=appeal_file,
                        file_name=f"AppealLetter_{claim_number}.txt",
                        mime="text/plain",
                    )
                with col2:
                    if st.button(f"Read Appeal Letter for Claim {claim_number}"):
                        with st.spinner("Generating audio..."):
                            audio_bytes = io.BytesIO()
                            tts = gTTS(appeal_letter, lang="en")
                            tts.write_to_fp(audio_bytes)
                            audio_bytes.seek(0)
                            st.audio(audio_bytes, format="audio/mp3", start_time=0)
                       
    
    results_df = pd.DataFrame(results)
    
    with tab2:
        st.header("Claim Appeal Status")
        st.dataframe(results_df[["Customer Name", "DOB", "Policy Number", "Claim Number", "Claim Date", "Appeal Letter Status", "Reason"]])
  
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="claim_results.csv",
            mime="text/csv",
        )
else:
    st.error("Please upload all required files.")
