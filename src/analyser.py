"""
analyser.py
AI Document Analyser — Core Analysis Engine

Extracts text from uploaded documents and uses an LLM to produce
structured insights: summary, decisions, action items, risks, and sentiment.

Supports: PDF, DOCX, TXT
Author: Aryan Patil
"""

import os
import json
import PyPDF2
import docx
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL  = "gpt-4o"


def extract_text(file_path: str) -> str:
    """
    Extracts plain text from a PDF, DOCX, or TXT file.

    @param  file_path   str     Absolute or relative path to the document file
    @return str                 Extracted plain text content of the document;
                                returns empty string if extraction fails or file type unsupported
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        text = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text)

    elif ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    else:
        print(f"Unsupported file type: {ext}")
        return ""


def truncate_text(text: str, max_chars: int = 12000) -> str:
    """
    Truncates document text to fit within the model's context window.
    Preserves the beginning of the document where key information typically appears.

    @param  text        str     Full extracted document text
    @param  max_chars   int     Maximum character count to send to the model (default: 12000)
    @return str                 Truncated text, with a note appended if truncation occurred
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[Document truncated for analysis. Showing first 12,000 characters.]"


def analyse_document(file_path: str) -> dict:
    """
    Runs the full analysis pipeline on a document file.
    Extracts text, sends it to the LLM with a structured prompt,
    and parses the response into a structured insights dictionary.

    @param  file_path   str     Path to the document to analyse
    @return dict                Structured insights with keys:
                                  - summary (str):           2-3 sentence executive summary
                                  - key_decisions (list):    Major decisions found in the document
                                  - action_items (list):     Action items with owner and deadline if present
                                  - risks_identified (list): Risks or concerns mentioned
                                  - sentiment (str):         Overall tone of the document
                                  - error (str):             Present only if analysis failed
    """
    print(f"Extracting text from: {file_path}")
    text = extract_text(file_path)

    if not text.strip():
        return {"error": "Could not extract text from the document. Check the file format."}

    text = truncate_text(text)

    prompt = f"""
You are an expert document analyst. Analyse the document below and return a JSON object with exactly these keys:

- "summary": A 2-3 sentence executive summary of the document's main purpose and conclusions.
- "key_decisions": A list of strings — the major decisions, conclusions, or directives in the document.
- "action_items": A list of objects with "owner", "action", and "deadline" (use null if not specified).
- "risks_identified": A list of strings — any risks, concerns, or issues mentioned or implied.
- "sentiment": One of: "positive", "negative", "neutral", "cautiously optimistic", "urgent".

Return ONLY valid JSON. No preamble, no markdown, no explanation.

DOCUMENT:
{text}
"""

    print("Sending to LLM for analysis...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1500
    )

    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Attempt to extract JSON if model added surrounding text
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            result = {"error": "Model returned unparseable output.", "raw": raw}

    return result


def answer_question(file_path: str, question: str) -> str:
    """
    Answers a specific natural language question about a document.
    Uses the same text extraction pipeline and sends the question
    alongside the document content to the LLM.

    @param  file_path   str     Path to the document to query
    @param  question    str     The natural language question to answer
    @return str                 The model's answer as a plain text string;
                                returns an error message string if extraction fails
    """
    text = extract_text(file_path)
    if not text.strip():
        return "Could not extract text from the document."

    text = truncate_text(text)

    prompt = f"""
Answer the following question about the document below.
Be concise and factual. If the answer is not in the document, say so clearly.

QUESTION: {question}

DOCUMENT:
{text}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyser.py <path_to_document> [question]")
        sys.exit(1)

    path = sys.argv[1]

    if len(sys.argv) >= 3:
        q = " ".join(sys.argv[2:])
        print(f"\nQuestion: {q}")
        print(answer_question(path, q))
    else:
        result = analyse_document(path)
        print(json.dumps(result, indent=2))
