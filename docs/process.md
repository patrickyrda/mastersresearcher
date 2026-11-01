# How it works

A short walkthrough of what happens between "here's a list of schools" and "here's a dashboard comparing them".
For the detailed diagrams, see [architecture.md](architecture.md).

## The idea

Comparing finance master's programs means digging through dozens of school websites, PDFs and brochures to find
the same handful of facts: requirements, deadlines, tuition, salaries, employment rate, class profile. This project
automates that, and keeps a quote and a source link for every value so nothing has to be taken on faith.

It runs for free: the only paid-looking piece is the Gemini API, and it uses the free tier with a hard daily budget.

## The process

### 1. Pick the programs
`programs.yaml` lists the programs (US, France, UK) with their main page URLs. You can add extra pages (an
employment report, a fees page) when the crawler misses something.

### 2. Crawl the school sites
The crawler starts from each program's page and follows the links that look relevant (admissions, fees, careers...),
while skipping unrelated sections like MBA or news pages. It respects `robots.txt`, waits between requests, and
caches everything. HTML and PDFs are converted to clean text. Sites that block bots can be handled by saving the
pages by hand into `manual/`.

### 3. Build a local search index
The text is split into small chunks and turned into embeddings with a small multilingual model that runs on the CPU.
This makes it possible to quickly find the passages that talk about, say, tuition or salaries. No API calls here.

### 4. Extract the facts with an LLM
For each group of fields, the most relevant chunks are sent to Gemini, which answers in a fixed JSON format with
the value, the exact quote it came from and the source. Each quote is then checked against the original page:
if it can't be found word for word, the value is flagged ⚠️ instead of trusted.

A rate limiter and a daily budget keep usage inside the free tier. When the budget runs out, the run stops cleanly
and picks up where it left off the next day.

### 5. Explore the results
A Streamlit dashboard shows:
- **Compare**: all programs side by side
- **Detail**: one program, every value with its quote and link
- **Ask**: free-form questions answered from the collected pages, with citations

## Stack

Python · httpx · trafilatura · PyMuPDF · fastembed · NumPy · Gemini API · Typer · Streamlit · pytest
