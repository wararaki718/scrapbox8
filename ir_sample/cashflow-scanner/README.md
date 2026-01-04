# CashFlow Scanner

This project extracts cash flow data from financial reports of listed companies to calculate a proprietary dividend safety score. It is designed as a standalone sample within the `ir_sample/` directory.

## Project Purpose

The primary goal is to automate the analysis of financial statements (in text or PDF format) to assess the sustainability of a company's dividend payments from the perspective of a veteran investor.

## Features

- **Data Extraction**: Uses Google's Gemini API with JSON Mode to accurately extract financial figures.
- **Data Validation**: Employs Pydantic to ensure the integrity of the extracted data.
- **PDF Parsing**: Integrates PyMuPDF to handle financial reports in PDF format.
- **Custom Logic**: Calculates a dividend safety score based on a unique formula.

### Calculation Logic

1.  **Free Cash Flow (Free CF)**
    ```
    Free CF = Cash Flow from Operating Activities + Cash Flow from Investing Activities
    ```
2.  **Dividend Safety Score**
    ```
    Score = Free CF / Total Dividend Payout
    ```
3.  **Judgment Criteria**
    - **Score >= 1.2**: "安全 (Safe)" - The company generates sufficient cash to cover its dividends.
    - **Score < 1.2**: "減配警戒 (Dividend Cut Warning)" - The company may face challenges in sustaining its dividend payments.

## Tech Stack

- **Language**: Python 3.11+
- **LLM**: Google Gemini API (`google-generativeai`)
- **Data Validation**: Pydantic (`pydantic`)
- **PDF Parsing**: PyMuPDF (`pymupdf`)

## Setup

1.  **Clone the repository and navigate to the project directory:**
    ```bash
    git clone <repository-url>
    cd <repository-path>/ir_sample/cashflow-scanner
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Set up your Google API Key:**
    You need to set your Google Gemini API key as an environment variable.
    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY"
    ```

## How to Run

Execute the `main.py` script to run the analysis on the sample data. The script includes a dummy financial report for demonstration purposes.

```bash
python main.py
```

sample url
- https://global.toyota/pages/global_toyota/ir/financial-results/2025_4q_summary_en.pdf

The script will:
1.  Initialize the `CashFlowScanner`.
2.  Extract financial data from the dummy report.
3.  Calculate the free cash flow and dividend safety score.
4.  Print the results and the final judgment.
