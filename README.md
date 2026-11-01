# Finance Masters Research

I built this side project while looking into finance master's programs. The information I needed (requirements,
documents, deadlines, tuition, salaries, employment rate, class profile) was spread across dozens of school websites,
often several pages deep or buried in PDFs. Checking every site by hand was slow, so this project collects it all in
one place.

It crawls the official pages of chosen programs in the US, France and the UK, pulls out the key facts with an LLM, and
shows them side by side in a local dashboard. Every value keeps a quote and a link to its source, so it can be checked.

## How it works
1. **Crawl.** A polite crawler (it respects `robots.txt`, waits between requests and caches pages) starts from each
   program's page in `programs.yaml` and follows the relevant links. HTML and PDFs are turned into clean text.
2. **Index.** The text is split into chunks and embedded locally.
3. **Extract.** For each group of fields, the most relevant chunks go to Gemini, which returns structured JSON with the
   value and the exact quote it came from. Each quote is then checked against the original page, and a value is
   flagged ⚠️ if its quote can't be found word for word.
4. **Explore.** A Streamlit dashboard has three tabs: **Compare** (all programs side by side), **Detail** (one program,
   with sources) and **Ask** (free-form questions).

More detail and diagrams are in [docs/process.md](docs/process.md) and [docs/architecture.md](docs/architecture.md).

### A simple RAG
Both the extraction and the **Ask** tab use a small retrieval-augmented generation setup, built without a vector database:
- Pages are split into ~350-word chunks with some overlap.
- Chunks are embedded on the CPU with a multilingual MiniLM model (fastembed), because many of the sites are in French.
- Vectors are stored as plain NumPy arrays, and retrieval is cosine similarity, with a small boost for the page type
  that fits the question (for example, an employment report for salary questions).
- The top chunks are sent to the model as numbered sources. It may only answer from them, and it cites them as `[n]`.

## Not done yet
- **Some sites block crawlers.** MIT, Oxford, Berkeley, Columbia and UCLA return HTTP 403. For now, their pages have to
  be saved by hand into `manual/<program-id>/`.
- **Some fields need a manual check.** A few values come back *not found* or flagged ⚠️. They need the right page added
  to `extra_urls` in `programs.yaml`, or a check by hand.

## Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env     # then paste your free key from https://aistudio.google.com/apikey
```

## Use
```powershell
python -m pipeline.cli check-seeds              # are the program URLs reachable? (free)
python -m pipeline.cli run --program hec-mif    # crawl → index → extract one program
python -m pipeline.cli run --all                # everything (resumable; may span a few days of free quota)
python -m pipeline.cli usage                    # Gemini requests/tokens used today
python -m pipeline.cli report                   # missing / unverified fields per program
streamlit run app/streamlit_app.py              # dashboard
pytest                                          # tests (mocked LLM, no network, no quota)
```
  
## Model and limits
- The LLM is the **Gemini API free tier**, so the project costs nothing to run. The key comes from Google AI Studio, in
  a project without billing, so it can never be charged. Embeddings run locally.
- Extraction uses `gemini-3.8-flash` and the Ask chat uses `gemini-3.5-flash-lite`. Each model has its own free daily
  quota. If one is overloaded or out of quota, a list of other free models is tried in turn.
- A rate limiter and a daily request/token budget stop a run cleanly before any limit is hit. Every step is saved to
  disk, so the next run continues where the last one stopped. One program takes about 5 requests.
- `crawl` and `index` never call the LLM. Only `extract`, `run` and the **Ask** tab use quota.
- All of this is set in `.env` (see `.env.example`). The real free-tier limits are shown at
  https://aistudio.google.com/rate-limit.

## Built with AI
This project was developed using [Claude Code](https://claude.com/claude-code).
