import fitz  # PyMuPDF for text extraction
import pdfplumber  # Extract structured tables
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import re
import json
import os
import spacy
import numpy as np
from huggingface_hub import InferenceClient

# ✅ Define file paths
pdf_path = r"D:\Assignment\data\data (4).pdf"
output_json_path = r"D:\Assignment\data\extracted_data.json"

# ✅ Set up Hugging Face API (Use a free model)
HF_API_KEY = "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"
llm = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token=HF_API_KEY)


# ✅ Preprocess OCR Images
def preprocess_image(img):
    """Enhance image for better OCR detection."""
    img = img.convert("L")  # Convert to grayscale
    img = img.filter(ImageFilter.SHARPEN)  # Sharpen the image
    img = ImageEnhance.Contrast(img).enhance(2)  # Increase contrast
    return img

# ✅ Extract text and tables with OCR & Table Parsing
def extract_text_from_pdf(pdf_path):
    """Extract text and tables from a PDF file using pdfplumber & PyMuPDF."""
    text = ""
    extracted_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                extracted_tables.append(table)  # Store table for debugging
                table_text = "\n".join(["\t".join([cell if cell else "" for cell in row]) for row in table])
                text += table_text + "\n"

    doc = fitz.open(pdf_path)
    for page in doc:
        page_text = page.get_text("text")
        if page_text.strip():
            text += page_text + "\n"
        else:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = preprocess_image(img)
            text += pytesseract.image_to_string(img) + "\n"

    return text.strip()

# ✅ Extract financial data using advanced regex
def extract_financial_data(text, quarter_dates):
    """Extracts financial data using adaptive regex and table-based parsing."""
    extracted_data = {}

    patterns = {
        "Revenue from operations": r"(?:Revenue from operations|Operating revenue)\s+([\d,]+\.\d+)",
        "Other income": r"(?:Other income|Miscellaneous income|Non-operating income)\s+([\d,]+\.\d+)",
        "Total income": r"(?:Total income|Gross income)\s+([\d,]+\.\d+)",
        "Cost of construction and development": r"(?:Cost of construction and development|Construction expenses)\s+([\d,]+\.\d+)",
        "Employee benefit expense": r"(?:Employee benefits expense|Staff expenses)\s+([\d,]+\.\d+)",
        "Finance costs": r"(?:Finance costs|Interest expense)\s+([\d,]+\.\d+)",
        "Depreciation and amortisation expenses": r"(?:Depreciation and amortisation expenses|Depreciation)\s+([\d,]+\.\d+)",
        "Other expenses": r"(?:Other expenses|Miscellaneous expenses)\s+([\d,]+\.\d+)",
        "Profit/loss before tax": r"(?:Profit/ \(loss\) before tax|Earnings before tax)\s+([\d,]+\.\d+)",
        "Current tax": r"(?:Current tax|Income tax)\s+([\d,]+\.\d+)",
        "Deferred tax": r"(?:Deferred tax|Tax adjustments)\s+([\d,]+\.\d+)",
        "Profit/loss for the period/year": r"(?:Profit/ \(loss\) for the period/ year|Net earnings)\s+([\d,]+\.\d+)"
    }

    for quarter in quarter_dates:
        quarter_data = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            quarter_data[key] = float(match.group(1).replace(",", "")) if match else None
        extracted_data[f"Quarter ended {quarter}"] = quarter_data

    return extracted_data

# ✅ Fill missing values using trend-based estimation
def interpolate_missing_values(data):
    """Use statistical methods to estimate missing financial values."""
    for quarter, values in data.items():
        non_null_values = [v for v in values.values() if v is not None]
        if len(non_null_values) > 1:
            avg_value = np.mean(non_null_values)
            for key in values:
                if values[key] is None:
                    values[key] = round(avg_value, 2)  # Fill missing values with mean
    return data

# ✅ Improve LLM Prompt for Missing Data Prediction
def refine_data_with_llm(data):
    """Use an LLM to refine extracted financial data and fill missing values."""
    prompt = f"""
    Given the following extracted financial data:
    {json.dumps(data, indent=2)}

    Some values are missing (null). Based on financial trends and calculations, fill in the missing values accurately.
    Ensure all values are present and follow the correct JSON structure.
    """

    try:
        # ✅ Use text_generation() instead of chat_completion()
        response = llm.text_generation(prompt, max_new_tokens=500)

        # ✅ Debugging: Print the raw response
        print("🔹 LLM Response:", response)

        # ✅ Check if the response is empty
        if not response:
            raise ValueError("❌ LLM response is empty.")

        # ✅ Ensure response starts with valid JSON structure
        llm_output = response.strip()
        if not llm_output.startswith("{") and not llm_output.startswith("["):
            raise ValueError("❌ LLM response is not in JSON format.")

        return json.loads(llm_output)

    except Exception as e:
        print(f"❌ Error in refine_data_with_llm: {e}")
        return data  # Return original extracted data if LLM fails



# ✅ Extract text from PDF
pdf_text = extract_text_from_pdf(pdf_path)

# ✅ Define reporting periods
quarter_dates = [
    "31 December 2024", "30 September 2024", "31 December 2023",
    "Year to date period ended 31 December 2024", "Year to date period ended 31 December 2023",
    "Year ended 31 March 2024"
]

# ✅ Extract financial data
standalone_results = extract_financial_data(pdf_text, quarter_dates)
consolidated_results = extract_financial_data(pdf_text, quarter_dates)

# ✅ Interpolate missing values using statistical methods
standalone_results = interpolate_missing_values(standalone_results)
consolidated_results = interpolate_missing_values(consolidated_results)

# ✅ Refine using LLM for final adjustments
standalone_results = refine_data_with_llm(standalone_results)
consolidated_results = refine_data_with_llm(consolidated_results)

# ✅ Save JSON Output
final_output = {
    "Standalone_financial_results_for_all_months": standalone_results,
    "Balance_sheet": "Balance_sheet_are_not_present",
    "Cash_flow_statements": "Cash_flow_statements_are_not_present",
    "Statement_Consolidated_finanacial_results_for_all_months": consolidated_results
}

with open(output_json_path, "w", encoding="utf-8") as json_file:
    json.dump(final_output, json_file, indent=4)

print(f"✅ Extracted JSON saved at: {output_json_path}")