# AI Document Analyser

Upload any document (PDF, DOCX, TXT) and get structured insights powered by an LLM. Built because I got tired of reading 40-page reports to find the three things that actually matter.

## What it does

- Extracts key decisions, action items, and risks from any document
- Generates an executive summary in plain English
- Answers questions about the document in natural language
- Exports findings as structured JSON or markdown

## Stack
Python · OpenAI API · LangChain · PyPDF2 · python-docx · Flask

## Quickstart
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
python src/app.py
```

Then open `http://localhost:5000` and upload a document.

## Example output
```json
{
  "summary": "The board meeting covered three strategic decisions...",
  "key_decisions": ["Approved $2M budget for Q3 expansion", "..."],
  "action_items": [{"owner": "CFO", "action": "Submit revised forecast by Friday"}],
  "risks_identified": ["Supply chain dependency on single vendor"],
  "sentiment": "cautiously optimistic"
}
```
