# Architecture

## 1. System overview

What talks to what. Everything except the Gemini API runs locally and is free.

```mermaid
flowchart LR
    subgraph IN["Inputs"]
        YAML["programs.yaml<br/>seed URLs, extra_urls"]
        MAN["manual/{id}/<br/>pages saved by hand"]
        ENV[".env<br/>GEMINI_API_KEY, limits"]
    end

    subgraph WEB["Internet"]
        SITES["School websites<br/>HTML + PDF"]
        GEM["Gemini API - free tier<br/>gemini-3.8-flash / 3.5-flash-lite"]
    end

    subgraph ING["Ingestion - local, no LLM"]
        FETCH["Fetcher<br/>httpx, robots.txt, delay, cache"]
        CRAWL["Crawler<br/>best-first, scoping, dedupe"]
        TEXT["Text extraction<br/>trafilatura, lxml, PyMuPDF"]
    end

    subgraph IDX["Knowledge base - local"]
        CHUNK["Chunker<br/>350 words, 60 overlap"]
        EMB["Embedder<br/>fastembed, multilingual MiniLM"]
        STORE[("Vector index<br/>numpy + chunks.json")]
    end

    subgraph LLM["LLM layer"]
        RET["Retriever<br/>cosine similarity + boost"]
        EXT["Extractor<br/>5 field groups, JSON mode"]
        VER["Quote verifier<br/>anti-hallucination"]
        QA["Q&A - RAG<br/>answers with citations"]
        GUARD["Rate limiter + daily request/token budget<br/>per model"]
    end

    subgraph OUT["Outputs"]
        REC[("data/extracted/{id}.json<br/>facts + quotes + sources")]
        USAGE[("data/usage.json")]
        CLI["CLI - Typer"]
        DASH["Dashboard - Streamlit<br/>Compare, Detail, Ask"]
    end

    YAML --> CRAWL
    MAN --> TEXT
    CRAWL --> FETCH
    FETCH <--> SITES
    FETCH --> TEXT
    TEXT --> CRAWL
    CRAWL --> CHUNK --> EMB --> STORE
    STORE --> RET
    RET --> EXT
    RET --> QA
    EXT --> GUARD
    QA --> GUARD
    ENV --> GUARD
    GUARD <--> GEM
    GUARD --> USAGE
    EXT --> VER --> REC
    REC --> DASH
    QA --> DASH
    USAGE --> DASH
    CLI --> CRAWL
    CLI --> EXT
    DASH -. "re-run button" .-> CRAWL
```

## 2. Pipeline stages and data on disk

Every stage writes its output to disk, so later runs skip stages that are already done.

```mermaid
flowchart TD
    A["programs.yaml"] -->|crawl| B[("data/raw/_cache/*<br/>raw HTML/PDF")]
    B --> C[("data/raw/{id}/docs.json<br/>clean text + doc_type")]
    M["manual/{id}/*"] --> C
    C -->|index| D[("data/index/{id}/<br/>vectors.npy + chunks.json")]
    D -->|extract - uses Gemini| E[("data/extracted/{id}.json")]
    E --> F["Dashboard"]
    D -->|ask - uses Gemini| F

    classDef free fill:#e6f4ea,stroke:#34a853,color:#000
    classDef quota fill:#fdecea,stroke:#ea4335,color:#000
    class B,C,D free
    class E quota
```

Green = free local steps. Red = uses the Gemini free quota.

## 3. Crawler decisions

How the crawler decides whether to follow a link.

```mermaid
flowchart TD
    L["Link found on a page"] --> S{"Same site?"}
    S -- no --> X["Skip"]
    S -- yes --> N{"Negative word?<br/>mba, minor, summer, news..."}
    N -- yes --> X
    N -- no --> P{"Inside the program's section?<br/>path prefix or URL slug"}
    P -- yes --> K1{"score >= 2?"}
    K1 -- yes --> Q["Push to priority queue"]
    K1 -- no --> X
    P -- no --> SIB{"Sibling program?<br/>same parent path"}
    SIB -- yes --> X
    SIB -- no --> K2{"score >= 5 and shallow URL<br/>and fewer than 5 off-section pages?"}
    K2 -- yes --> Q2["Push as leaf page<br/>(its links are not followed)"]
    K2 -- no --> X
```

## 4. Extracting one field group

```mermaid
flowchart TD
    G["Field group, e.g. outcomes"] --> R["Retrieve top 5 chunks of this program<br/>query embedding x chunk vectors"]
    R --> P["Build prompt<br/>numbered sources + JSON template"]
    P --> B{"Budget and rate limit OK?"}
    B -- "daily request/token budget reached" --> STOP["Stop cleanly<br/>resume tomorrow"]
    B -- "per-minute limit" --> W["Wait"] --> B
    B -- ok --> C["Gemini call<br/>temperature 0, JSON output, low thinking"]
    C -- "429, short retryDelay" --> W
    C -- "429, daily quota or long retryDelay" --> STOP
    C -- ok --> J["Parse JSON + coerce types<br/>'€78,500' → 78500"]
    J --> V{"Quote found verbatim<br/>in the source page?"}
    V -- yes --> OK["Fact verified ✔"]
    V -- no --> FL["Fact flagged ⚠️<br/>hidden from Compare by default"]
    OK --> S[("Save record<br/>checkpoint")]
    FL --> S
```

## 5. Summarized sequence

A full run for one program, followed by a question in the dashboard.

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant CLI as CLI / Dashboard
    participant CR as Crawler + Fetcher
    participant W as School website
    participant IX as Indexer + Embedder
    participant EX as Extractor + Verifier
    participant RL as Rate limiter / Budget
    participant G as Gemini API

    U->>CLI: run --program hec-mif
    CLI->>CR: crawl(seed)
    loop best-first, up to 20 pages
        CR->>W: GET page (robots.txt checked, 1s delay)
        W-->>CR: HTML / PDF
        CR->>CR: extract text, score links, dedupe
    end
    CR-->>CLI: docs.json

    CLI->>IX: index(docs)
    IX->>IX: chunk + embed locally
    IX-->>CLI: vectors + chunks

    loop 5 field groups
        CLI->>EX: extract group
        EX->>IX: retrieve top-5 chunks
        IX-->>EX: relevant chunks
        EX->>RL: check budget, wait for slot
        alt budget available
            RL->>G: generate_content (JSON output)
            G-->>RL: JSON answer + token usage
            RL-->>EX: answer (usage recorded)
            EX->>EX: validate, verify quotes
            EX-->>CLI: facts saved (checkpoint)
        else budget reached
            RL-->>CLI: stop cleanly, resume tomorrow
        end
    end

    U->>CLI: open dashboard, ask a question
    CLI->>IX: retrieve top-8 chunks
    IX-->>CLI: chunks
    CLI->>RL: check chat-model budget
    RL->>G: question + sources
    G-->>CLI: answer with [n] citations
    CLI-->>U: answer + source links
```
