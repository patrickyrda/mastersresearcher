"""Dashboard: streamlit run app/streamlit_app.py"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import MissingKeyError, settings  # noqa: E402
from pipeline.extract import load_record  # noqa: E402
from pipeline.index import Index, get_embedder  # noqa: E402
from pipeline.ratelimit import BudgetExceeded, UsageTracker  # noqa: E402
from pipeline.schema import FIELD_GROUPS, Fact, ProgramRecord, load_programs  # noqa: E402

# Rough fixed conversion, only for sorting/filtering salaries across countries
TO_EUR = {"EUR": 1.0, "GBP": 1.17, "USD": 0.90}
FLAGS = {"US": "🇺🇸", "FR": "🇫🇷", "UK": "🇬🇧"}

st.set_page_config(page_title="Finance Masters Research", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Loading local embedding model…")
def embedder():
    return get_embedder(settings)


def load_all() -> tuple[list, dict[str, ProgramRecord]]:
    seeds = load_programs()
    records = {s.id: r for s in seeds if (r := load_record(s.id))}
    return seeds, records


def show(f: Fact, include_unverified: bool):
    if f.value is None or (not f.verified and not include_unverified):
        return None
    return f.value


def to_eur(amount, currency) -> float | None:
    if not isinstance(amount, (int, float)) or not isinstance(currency, str):
        return None
    rate = TO_EUR.get(currency.strip().upper()[:3])
    return round(amount * rate) if rate else None


seeds, records = load_all()
seed_by_id = {s.id: s for s in seeds}

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Pipeline")
    tracker = UsageTracker(settings.daily_token_budget, settings.daily_request_budget)
    st.caption(f"Gemini requests today (free tier, budget {settings.daily_request_budget}/model/day)")
    for role, model in [("Extraction", settings.extract_model), ("Chat", settings.chat_model)]:
        used = tracker.used(model)
        st.progress(min(used["requests"] / max(settings.daily_request_budget, 1), 1.0),
                    text=f"{role} · {model}: {used['requests']} req · {used['tokens']:,} tokens")
    st.caption("🔑 GEMINI_API_KEY " + ("is set" if settings.gemini_api_key else "is **missing** — see .env.example"))
    st.caption(f"{len(records)}/{len(seeds)} programs extracted")

    st.divider()
    target = st.selectbox("Re-run pipeline for", [s.id for s in seeds],
                          format_func=lambda i: f"{seed_by_id[i].school} — {seed_by_id[i].program}")
    force = st.checkbox("Force (redo crawl + extraction)", help="Uses Gemini quota again")
    if st.button("Run", width="stretch"):
        from pipeline.cli import run_program

        log = io.StringIO()
        with st.spinner("Crawling, indexing, extracting… (can take a few minutes)"):
            try:
                with contextlib.redirect_stdout(log):
                    run_program(seed_by_id[target], force=force)
                st.success("Done")
            except (BudgetExceeded, MissingKeyError) as e:
                st.error(str(e))
        st.code(log.getvalue()[-3000:] or "(no output)")

st.title("🎓 Finance Masters Research")
tab_compare, tab_detail, tab_ask = st.tabs(["Compare", "Program detail", "Ask"])

# ---------------- Compare ----------------
with tab_compare:
    if not records:
        st.info("No programs extracted yet. Run `python -m pipeline.cli run --program hec-mif` "
                "or use the sidebar.")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    countries = c1.multiselect("Country", ["FR", "UK", "US"], default=["FR", "UK", "US"])
    no_gmat = c2.checkbox("GMAT/GRE not required")
    min_salary = c3.number_input("Min salary (≈EUR)", 0, 300_000, 0, step=5_000)
    include_unverified = c4.checkbox("Show unverified values", help="Values whose quote wasn't found in the source")

    rows = []
    for s in seeds:
        r = records.get(s.id) or ProgramRecord(seed=s)
        g = lambda grp, name: show(r.fact(grp, name), include_unverified)  # noqa: E731
        salary, cur = g("outcomes", "avg_salary"), g("outcomes", "salary_currency")
        deadlines = g("admissions", "deadlines")
        rows.append({
            "Country": f"{FLAGS[s.country]} {s.country}", "School": s.school, "Program": s.program,
            "City": s.city, "Ranking": s.ranking,
            "Duration": g("overview", "duration"),
            "Tuition": g("overview", "tuition"), "Tuition cur.": g("overview", "tuition_currency"),
            "Salary": salary, "Salary cur.": cur, "Salary ≈EUR": to_eur(salary, cur),
            "Employed %": g("outcomes", "employment_rate"),
            "GMAT/GRE": g("admissions", "gmat_gre_policy"),
            "English test": g("admissions", "english_test_minimum"),
            "Deadlines": " | ".join(deadlines) if isinstance(deadlines, list) else deadlines,
            "Class size": g("class_profile", "class_size"),
            "% intl": g("class_profile", "pct_international"),
            "_country": s.country, "_extracted": s.id in records,
        })
    df = pd.DataFrame(rows)
    df = df[df["_country"].isin(countries)]
    if no_gmat:
        policy = df["GMAT/GRE"].fillna("").str.lower()
        df = df[policy.str.contains("optional|waive|not required|no gmat|not mandatory")]
    if min_salary:
        df = df[df["Salary ≈EUR"].fillna(0) >= min_salary]
    st.dataframe(
        df.drop(columns=["_country", "_extracted"]), hide_index=True, width="stretch",
        column_config={
            "Tuition": st.column_config.NumberColumn(format="%,.0f"),
            "Salary": st.column_config.NumberColumn(format="%,.0f"),
            "Salary ≈EUR": st.column_config.NumberColumn(format="€%,.0f", help="Fixed approx. rates"),
            "Employed %": st.column_config.NumberColumn(format="%.0f%%"),
        },
    )
    st.caption("Empty cell = not found in the crawled pages (or unverified). Check *Program detail* for sources; "
               "add missing pages to `extra_urls` in programs.yaml.")

# ---------------- Program detail ----------------
with tab_detail:
    pid = st.selectbox("Program", [s.id for s in seeds], key="detail",
                       format_func=lambda i: f"{FLAGS[seed_by_id[i].country]} {seed_by_id[i].school} — {seed_by_id[i].program}"
                       + ("" if i in records else "  (not extracted)"))
    s = seed_by_id[pid]
    r = records.get(pid)
    st.subheader(f"{s.program} — {s.school}")
    st.write(" · ".join(filter(None, [s.city, s.country, s.ranking])) + "  ·  " +
             " ".join(f"[program page]({u})" for u in s.urls))
    if not r:
        st.info("Not extracted yet — use the sidebar to run the pipeline for this program.")
    else:
        st.caption(f"Extracted {r.extracted_at}")
        for group_name, group in FIELD_GROUPS.items():
            st.markdown(f"#### {group_name.replace('_', ' ').title()}")
            for name, (_, desc) in group["fields"].items():
                f = r.fact(group_name, name)
                label = name.replace("_", " ").capitalize()
                if f.value is None:
                    st.markdown(f"**{label}:** <span style='opacity:.5'>not found</span>", unsafe_allow_html=True)
                    continue
                value = ", ".join(f.value) if isinstance(f.value, list) else f.value
                if isinstance(value, float) and value.is_integer():
                    value = f"{int(value):,}"
                mark = "" if f.verified else " ⚠️"
                with st.expander(f"**{label}:** {value}{mark}"):
                    if not f.verified:
                        st.warning("Quote not found verbatim in the source — double-check this value.")
                    if f.quote:
                        st.markdown(f"> {f.quote}")
                    if f.source_url and f.source_url.startswith("http"):
                        st.markdown(f"Source: [{f.source_url}]({f.source_url})")
                    elif f.source_url:
                        st.markdown(f"Source: `{f.source_url}` (saved by hand)")
                    st.caption(desc)

# ---------------- Ask ----------------
with tab_ask:
    available = Index.available()
    if not available:
        st.info("No indexed pages yet — run the pipeline first.")
    else:
        scope = st.selectbox("Search in", ["All programs"] + available,
                             format_func=lambda i: i if i == "All programs" else
                             f"{seed_by_id[i].school} — {seed_by_id[i].program}" if i in seed_by_id else i)
        st.session_state.setdefault("chat", [])
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if question := st.chat_input("e.g. Which UK programs don't require the GMAT?"):
            st.session_state.chat.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                try:
                    from pipeline.llm import GeminiClient
                    from pipeline.qa import answer

                    with st.spinner("Searching…"):
                        text, chunks = answer(question, Index.load(available), embedder(), GeminiClient(settings),
                                              settings, seed_by_id,
                                              program_id=None if scope == "All programs" else scope)
                    sources = "\n".join(f"{i}. {c['url']}" for i, c in enumerate(chunks, 1))
                    text = f"{text}\n\n<details><summary>Sources</summary>\n\n{sources}\n</details>"
                except (BudgetExceeded, MissingKeyError) as e:
                    text = f"⚠️ {e}"
                st.markdown(text, unsafe_allow_html=True)
            st.session_state.chat.append({"role": "assistant", "content": text})
