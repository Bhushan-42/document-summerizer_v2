import os
import uuid
import json # <-- Add json import
from flask import Flask, request, jsonify, render_template, abort
from werkzeug.utils import secure_filename
from ollama import chat

# --- Text Extraction Libraries ---
import PyPDF2
import docx # Correctly refers to python-docx library

# --- Data Handling & Visualization ---
import pandas as pd       # <-- Add pandas
import plotly.express as px # <-- Add Plotly Express
import plotly.io as pio     # <-- Add Plotly IO for JSON export

# --- Flask App Configuration ---
UPLOAD_FOLDER = 'uploads'
# --- UPDATED ALLOWED_EXTENSIONS ---
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xlsx', 'xls'} # <-- Add Excel extensions

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB upload limit

# --- Ollama Configuration ---
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.1:8b')

# --- Helper Functions ---

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    try:
        text = ""
        with open(file_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            if reader.is_encrypted:
                try:
                    reader.decrypt('')
                except Exception as decrypt_err:
                    print(f"Could not decrypt PDF {file_path}: {decrypt_err}")
                    return "Error reading PDF: File is encrypted and could not be decrypted."

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text if text else "Could not extract text from PDF (possibly image-based or empty)."
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        if "encrypted" in str(e).lower():
             return "Error reading PDF: File is encrypted."
        return f"Error reading PDF: {e}"


def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text])
        return text if text else "No text found in DOCX document."
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
        return f"Error reading DOCX: {e}"

def summarize_text_with_ollama(text_content, max_length=30000):
    if not text_content or text_content.isspace():
        return "Cannot summarize empty or whitespace-only text."

    if len(text_content) > max_length:
        print(f"Warning: Text length ({len(text_content)}) exceeds limit ({max_length}). Truncating.")
        text_content = text_content[:max_length] + "\n... [Content Truncated]"

    summarization_prompt = f"""
You are an expert text summarizer. Please provide a concise summary of the following document content.
Focus on the main points, key findings, and overall message. Avoid adding opinions or information not present in the text.
The summary should be easy to understand and capture the essence of the document.

Document Content:
---
{text_content}
---

Concise Summary:
"""
    try:
        response = chat(model=OLLAMA_MODEL, messages=[
            {'role': 'user', 'content': summarization_prompt}
        ])
        summary = response.get('message', {}).get('content', "").strip()
        return summary if summary else "AI model returned an empty summary."
    except Exception as e:
        print(f"Error calling Ollama for summarization: {e}")
        return f"Error during summarization: {e}"

# --- NEW: Function to Create Visualization from Excel ---
def create_visualization_from_excel(file_path):
    """
    Reads an Excel file and generates a simple Plotly bar chart JSON.
    Assumes:
        - First sheet is used.
        - First column is categorical (X-axis).
        - Second column is numerical (Y-axis).
    Returns:
        dict: Plotly chart JSON data or None if error.
        str: Error message or None if success.
    """
    try:
        # Read the first sheet
        df = pd.read_excel(file_path, sheet_name=0)

        if df.empty:
            return None, "Excel file is empty or has no data in the first sheet."
        if len(df.columns) < 2:
            return None, "Excel file needs at least two columns for visualization (Category, Value)."

        # Basic check if the second column looks numeric
        # You might need more robust type checking in a real application
        if not pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
             # Attempt conversion, ignore errors for non-numeric values initially
             df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
             # Check if *any* conversion worked. If not, it's likely not numeric data.
             if df.iloc[:, 1].isnull().all():
                 return None, f"The second column ('{df.columns[1]}') does not appear to contain numeric data suitable for a bar chart value axis."
             # Optional: Inform the user about skipped non-numeric rows if needed

        # Assume first column is X, second column is Y
        x_col = df.columns[0]
        y_col = df.columns[1]

        # Create a bar chart using Plotly Express
        fig = px.bar(df, x=x_col, y=y_col, title=f"Visualization of {x_col} vs {y_col}")

        # Convert the figure to a JSON string compatible with Plotly.js
        chart_json_str = pio.to_json(fig)
        chart_json_dict = json.loads(chart_json_str) # Parse string back to dict

        return chart_json_dict, None # Return JSON dict and no error

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, "Error processing file: File not found."
    except pd.errors.EmptyDataError:
         print(f"Error: No data found in Excel file {file_path}")
         return None, "Excel file is empty or contains no data."
    except IndexError:
        print(f"Error: Could not access columns in Excel file {file_path}")
        return None, "Could not access expected columns (need at least two)."
    except Exception as e:
        print(f"Error reading or visualizing Excel file {file_path}: {e}")
        # Consider more specific pandas/excel error handling if needed
        return None, f"An unexpected error occurred while processing the Excel file: {e}"

# --- Routes for HTML Pages ---
@app.route('/')
def index():
    return render_template('index.html', active_page='home')

@app.route('/summarization')
def summarization_page():
    return render_template('summarization.html', active_page='summarization')

@app.route('/about')
def about():
    return render_template('about.html', active_page='about')

@app.route('/research')
def research():
    return render_template('research.html', active_page='research')

# --- NEW: Route for Visualization Page ---
@app.route('/visualization', methods=['GET', 'POST'])
def visualization_page_and_handler():
    if request.method == 'GET':
        # Show the upload page
        return render_template('visualization.html', active_page='visualization')

    # --- POST Request Logic (File Upload and Processing) ---
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        file_extension = original_filename.rsplit('.', 1)[1].lower()

        # Ensure it's an Excel file for this endpoint
        if file_extension not in ['xlsx', 'xls']:
             print(f"Incorrect file type for visualization: {original_filename}")
             return jsonify({"error": f"File type not allowed for visualization. Please upload XLSX or XLS."}), 400

        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        try:
            file.save(file_path)
            print(f"Excel file saved temporarily to: {file_path}")

            # Generate visualization
            chart_json, error_message = create_visualization_from_excel(file_path)

            # Clean up the uploaded file
            os.remove(file_path)
            print(f"Temporary file removed: {file_path}")

            if error_message:
                 print(f"Visualization generation failed: {error_message}")
                 return jsonify({"error": error_message}), 400 # Use 400 for processing errors

            print("Visualization JSON generated successfully.")
            return jsonify({"chart_json": chart_json}) # Send Plotly JSON to frontend

        except Exception as e:
            print(f"An error occurred during visualization processing: {e}")
            # Ensure cleanup even if visualization function fails unexpectedly
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Temporary file removed after error: {file_path}")
                except OSError as remove_err:
                    print(f"Error removing file during exception handling: {remove_err}")
            return jsonify({"error": f"An unexpected server error occurred: {e}"}), 500

    else:
        print(f"File type not allowed or invalid file: {file.filename}")
        return jsonify({"error": f"File type not allowed. Please upload XLSX or XLS."}), 400


# --- API Endpoint for Summarization ---
@app.route('/summarize', methods=['POST'])
def handle_summarize():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        file_extension = original_filename.rsplit('.', 1)[1].lower()

        # --- Ensure only PDF/DOCX are handled here ---
        if file_extension not in ['pdf', 'docx']:
            print(f"Incorrect file type for summarization: {original_filename}")
            return jsonify({"error": f"File type not allowed for summarization. Please upload PDF or DOCX."}), 400

        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        try:
            file.save(file_path)
            print(f"File saved temporarily to: {file_path}")

            extracted_text = ""
            summary = "" # Initialize summary

            if file_extension == 'pdf':
                extracted_text = extract_text_from_pdf(file_path)
            elif file_extension == 'docx':
                extracted_text = extract_text_from_docx(file_path)

            # Check for extraction errors before summarizing
            if extracted_text.startswith("Error reading") or extracted_text.startswith("Could not extract"):
                summary = f"Text Extraction Failed: {extracted_text}"
            elif not extracted_text or extracted_text.isspace():
                summary = "Text extraction resulted in empty content. Cannot summarize."
            else:
                print(f"Extracted text length: {len(extracted_text)}. Summarizing...")
                summary = summarize_text_with_ollama(extracted_text)

            # Clean up the uploaded file
            os.remove(file_path)
            print(f"Temporary file removed: {file_path}")

            return jsonify({"summary": summary})

        except Exception as e:
            print(f"An error occurred during summarization processing: {e}")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError as remove_err:
                    print(f"Error removing file during exception handling: {remove_err}")
            return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

    else:
        # This case should now only trigger if allowed_file fails for pdf/docx
        print(f"File type not allowed (Summarize Endpoint): {file.filename}")
        return jsonify({"error": f"File type not allowed. Please upload PDF or DOCX."}), 400


# --- Run the App ---
if __name__ == '__main__':
    # <-- Updated print message to reflect added functionality -->
    print(f"Starting Flask Document Summarizer & Visualizer server (PDF, DOCX, XLSX, XLS) using Ollama model '{OLLAMA_MODEL}'...")
    app.run(host='0.0.0.0', port=5001, debug=True) # Using port 5001 as in original code