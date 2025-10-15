import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
import subprocess
import tempfile
import base64
import uuid
from pathlib import Path
import time
from datetime import datetime
import json
import reportlab
import sys

# Load environment variables
load_dotenv()

# Initialize the model
@st.cache_resource
def load_model():
    return GoogleGenerativeAI(
        model='gemini-2.5-flash',
        temperature=st.session_state.get('temperature', 0.5),
        google_api_key=os.getenv('GOOGLE_API_KEY')
    )

def generate_pdf_code(prompt, style_preference="", include_examples=True):
    """Generate Python code for PDF creation using LLM"""
    
    style_instructions = ""
    if style_preference:
        style_instructions = f" Style preferences: {style_preference}"
    
    examples = ""
    if include_examples:
        examples = """
        Important: Use reportlab for PDF generation. Follow these guidelines:
        - Always use Canvas from reportlab.pdfgen
        - Set page size to A4 (595.27, 841.89)
        - Use proper margins (50-100 units)
        - Include proper error handling
        - Save with a unique filename using timestamp
        - Use professional fonts and colors
        """
    
    template = PromptTemplate(
        template="""You are a highly skilled Python developer specialized in generating PDF files using reportlab.
        Your task is to write complete and correct Python code that generates a beautiful PDF based on the user's instructions.

        Requirements:
        {examples}
        {style_instructions}

        User Prompt: {prompt}

        Return only the python code without any explanations or markdown formatting.
        Make sure the code is complete and can run independently.""",
    )
    
    parser = StrOutputParser()
    model = load_model()
    chain = template | model | parser
    
    try:
        tempCode = str(chain.invoke({
            'prompt': prompt,
            'style_instructions': style_instructions,
            'examples': examples
        }))
        # Clean the code
        tempCode = tempCode.replace("```python", "").replace("```", "").strip()
        return tempCode
    except Exception as e:
        st.error(f"Error generating code: {str(e)}")
        return None

def create_pdf_from_code(code):
    """Execute the generated code to create PDF"""
    try:
        # Create a temporary file with unique name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file_path = f.name
        
        # Execute the code with timeout
        result = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Clean up the temporary Python file
        os.unlink(temp_file_path)
        
        if result.returncode != 0:
            error_msg = result.stderr
            # Try to provide more helpful error messages
            if "ModuleNotFoundError" in error_msg:
                error_msg += "\n\n💡 Tip: The code requires additional libraries. Please install them using: pip install reportlab Pillow"
            st.error(f"Error executing code: {error_msg}")
            return None, error_msg
            
        return result, None
    except subprocess.TimeoutExpired:
        error_msg = "PDF generation timed out. The code might be stuck in an infinite loop."
        st.error(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"Error creating PDF: {str(e)}"
        st.error(error_msg)
        return None, error_msg

def find_pdf_files(directory='.'):
    """Find PDF files in the current directory"""
    pdf_files = list(Path(directory).glob('*.pdf'))
    # Sort by modification time, newest first
    pdf_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return pdf_files

def get_pdf_download_link(pdf_path, filename):
    """Generate a download link for the PDF file"""
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    b64_pdf = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}" style="\
            background: linear-gradient(45deg, #4CAF50, #45a049); \
            color: white; \
            padding: 14px 20px; \
            text-align: center; \
            text-decoration: none; \
            display: inline-block; \
            border-radius: 8px; \
            font-size: 16px; \
            font-weight: bold; \
            margin: 10px 0; \
            border: none; \
            cursor: pointer; \
            transition: all 0.3s ease;">\
            📥 Download PDF</a>'
    return href

def validate_code_safety(code):
    """Basic safety check for generated code"""
    dangerous_patterns = [
        "os.system", "subprocess.call", "eval(", "exec(", "__import__",
        "open(", "shutil.rmtree", "rm -", "del ", "import os"
    ]
    
    warnings = []
    for pattern in dangerous_patterns:
        if pattern in code:
            warnings.append(f"Potential security concern: {pattern}")
    
    return warnings

def save_to_history(prompt, code, success, filename=None):
    """Save generation attempt to history"""
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    history_entry = {
        'timestamp': datetime.now().isoformat(),
        'prompt': prompt,
        'code': code,
        'success': success,
        'filename': filename
    }
    
    st.session_state.history.insert(0, history_entry)
    # Keep only last 10 entries
    st.session_state.history = st.session_state.history[:10]

def main():
    st.set_page_config(
        page_title="AI PDF Generator Pro",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            background: linear-gradient(45deg, #1f77b4, #ff7f0e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
        .feature-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            color: white;
            margin: 10px 0;
        }
        .download-section {
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, #e8f4fd 0%, #d1e7ff 100%);
            border-radius: 15px;
            margin: 20px 0;
            border: 2px dashed #1f77b4;
        }
        .success-message {
            padding: 15px;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 10px;
            color: #155724;
            margin: 10px 0;
        }
        .stProgress > div > div > div > div {
            background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'generated_code' not in st.session_state:
        st.session_state.generated_code = None
    if 'pdf_generated' not in st.session_state:
        st.session_state.pdf_generated = False
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI-Powered PDF Generator Pro</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model settings
        st.subheader("Model Configuration")
        temperature = st.slider("Creativity Temperature", 0.1, 1.0, 0.5, 0.1,
                               help="Higher values make output more creative, lower values more deterministic")
        st.session_state.temperature = temperature
        
        # Style preferences
        st.subheader("Style Preferences")
        style_preference = st.selectbox(
            "Document Style",
            ["Professional", "Modern", "Minimal", "Elegant", "Creative", "Custom"]
        )
        
        if style_preference == "Custom":
            custom_style = st.text_input("Describe your custom style:")
            style_preference = custom_style
        
        # Advanced options
        with st.expander("Advanced Options"):
            include_examples = st.checkbox("Include coding guidelines", value=True)
            auto_download = st.checkbox("Auto-download PDF", value=False)
            show_debug = st.checkbox("Show debug information", value=False)
        
        # Examples section
        st.subheader("📚 Quick Examples")
        examples = {
            "Professional Resume": "Create a modern professional resume with sections for education, work experience, skills, and contact information. Use a clean layout with proper spacing.",
            "Business Invoice": "Generate a professional invoice template with company header, itemized products/services, subtotal, tax, and total amount. Include payment terms and due date.",
            "Project Report": "Create a comprehensive project report with cover page, table of contents, executive summary, methodology, results, and conclusion sections.",
            "Event Certificate": "Design an elegant certificate template with decorative border, centered title, recipient name, description, and signature lines.",
            "Marketing Brochure": "Generate a tri-fold brochure with attractive headings, bullet points, and placeholders for images. Use a modern color scheme."
        }
        
        for name, example in examples.items():
            if st.button(f"📄 {name}", key=name):
                st.session_state.prompt_text = example
                st.rerun()
        
        # History section
        if 'history' in st.session_state and st.session_state.history:
            st.subheader("📋 Generation History")
            for i, entry in enumerate(st.session_state.history[:5]):
                with st.expander(f"{entry['timestamp'][:16]} - {entry['prompt'][:50]}..."):
                    st.write(f"Status: {'✅ Success' if entry['success'] else '❌ Failed'}")
                    if st.button(f"Retry #{i+1}", key=f"retry_{i}"):
                        st.session_state.prompt_text = entry['prompt']
                        st.rerun()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Prompt input with enhanced features
        st.subheader("📝 Describe Your PDF")
        prompt = st.text_area(
            "What kind of PDF would you like to generate?",
            height=150,
            placeholder="Example: Create a professional invoice PDF with company header, item table, tax calculation, and total amount...",
            value=st.session_state.get('prompt_text', ''),
            help="Be as specific as possible about content, layout, and style"
        )
        
        # Additional requirements
        col1a, col1b = st.columns(2)
        with col1a:
            page_size = st.selectbox("Page Size", ["A4", "Letter", "Legal", "A3"])
        with col1b:
            orientation = st.selectbox("Orientation", ["Portrait", "Landscape"])
        
        # Generate button with progress
        generate_btn = st.button("🚀 Generate PDF", type="primary", use_container_width=True)
        
        # Manual code editing option
        if st.session_state.get('generated_code'):
            with st.expander("✏️ Edit Generated Code (Advanced)"):
                edited_code = st.text_area(
                    "Modify the generated code if needed:",
                    value=st.session_state.generated_code,
                    height=300
                )
                if st.button("🔄 Run Edited Code"):
                    st.session_state.generated_code = edited_code
                    # Rerun with edited code
    
    with col2:
        st.subheader("💡 Tips & Best Practices")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; color: white;'>
        <h4 style='color: white;'>🎯 For Best Results:</h4>
        <ul style='color: white;'>
        <li><b>Be specific</b> about content structure</li>
        <li><b>Mention</b> required sections (headers, tables, lists)</li>
        <li><b>Specify</b> color schemes and fonts</li>
        <li><b>Include</b> any branding requirements</li>
        <li><b>Describe</b> the document's purpose</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        **Supported Features:**
        ✅ Tables & Lists
        ✅ Images & Logos  
        ✅ Multiple Pages
        ✅ Headers & Footers
        ✅ Different Fonts
        ✅ Colors & Styling
        """)
    
    # Processing
    if generate_btn and prompt:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Generate code
        status_text.text("🤖 Generating PDF code...")
        progress_bar.progress(25)
        
        code = generate_pdf_code(prompt, style_preference, include_examples)
        
        if code:
            # Safety check
            safety_warnings = validate_code_safety(code)
            if safety_warnings and show_debug:
                for warning in safety_warnings:
                    st.warning(warning)
            
            st.session_state.generated_code = code
            
            # Display generated code
            with st.expander("🔍 View Generated Code", expanded=show_debug):
                st.code(code, language='python')
            
            status_text.text("✅ Code generated successfully! Creating PDF...")
            progress_bar.progress(50)
            
            # Step 2: Create PDF
            with st.spinner("📄 Generating PDF document..."):
                result, error = create_pdf_from_code(code)
                
                if result:
                    progress_bar.progress(75)
                    
                    # Find generated PDF files
                    pdf_files = find_pdf_files()
                    
                    if pdf_files:
                        pdf_file = pdf_files[0]
                        progress_bar.progress(100)
                        status_text.text("✅ PDF generated successfully!")
                        
                        # Success section
                        st.balloons()
                        
                        # Save to history
                        save_to_history(prompt, code, True, pdf_file.name)
                        
                        # Enhanced download section
                        st.markdown('<div class="download-section">', unsafe_allow_html=True)
                        st.markdown("### 🎉 Your PDF is Ready!")
                        
                        col_d1, col_d2, col_d3 = st.columns(3)
                        
                        with col_d1:
                            st.markdown(get_pdf_download_link(pdf_file, f"document_{uuid.uuid4().hex[:8]}.pdf"), unsafe_allow_html=True)
                        
                        with col_d2:
                            # PDF preview (first page as image would be better, but simplified)
                            with open(pdf_file, "rb") as f:
                                pdf_bytes = f.read()
                            
                            st.download_button(
                                label="📥 Download PDF",
                                data=pdf_bytes,
                                file_name=f"generated_document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        
                        with col_d3:
                            if st.button("🔄 Generate Another", use_container_width=True):
                                st.session_state.prompt_text = ""
                                st.session_state.generated_code = None
                                st.rerun()
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # File info
                        file_size = pdf_file.stat().st_size / 1024  # KB
                        st.info(f"**File Info:** {pdf_file.name} | {file_size:.1f} KB | Generated at {datetime.now().strftime('%H:%M:%S')}")
                        
                        # Clean up old files (keep last 3)
                        for old_pdf in pdf_files[3:]:
                            try:
                                old_pdf.unlink()
                            except:
                                pass
                    else:
                        status_text.text("❌ PDF generation failed")
                        save_to_history(prompt, code, False)
                        st.error("PDF file was not generated. Please try again with a more specific prompt.")
                else:
                    progress_bar.progress(100)
                    status_text.text("❌ PDF creation failed")
                    save_to_history(prompt, code, False)
                    st.error("Failed to create PDF. Please try again with a different prompt.")
        else:
            progress_bar.progress(100)
            status_text.text("❌ Code generation failed")
            st.error("Failed to generate code. Please check your API configuration and try again.")
    
    elif generate_btn and not prompt:
        st.warning("⚠️ Please enter a description for the PDF you want to generate.")
    
    # Footer
    st.markdown("---")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f2:
        st.markdown(
            "<div style='text-align: center; color: #666;'>"
            "Powered by Google Gemini AI & Streamlit<br>"
            "Generated PDFs are automatically cleaned up after download"
            "</div>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":

    main()




