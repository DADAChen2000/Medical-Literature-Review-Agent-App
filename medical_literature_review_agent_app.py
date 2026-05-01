import json
import os
import re
import time
import textwrap
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "your_email@example.com")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

if not OPENAI_API_KEY:
    st.warning("请先在 .env 中配置 OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def schema_obj(name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "name": name,
        "strict": True,
        "schema": schema,
    }


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```json\s*|```$", "", text, flags=re.S)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("模型没有返回合法 JSON")
    return json.loads(match.group(0))


def llm_json(system_prompt: str, user_prompt: str, schema_name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """优先使用 Responses API 的结构化输出；失败时回退到 JSON 文本解析。"""
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={"format": schema_obj(schema_name, schema)},
        )
        return json.loads(resp.output_text)
    except Exception:
        fallback_prompt = f"""
请严格返回 JSON，不要输出 Markdown，不要解释。
JSON Schema 参考：
{json.dumps(schema, ensure_ascii=False)}

任务：
{user_prompt}
"""
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": fallback_prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return extract_json(resp.choices[0].message.content or "")


PICO_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "research_question": {"type": "string"},
        "pico": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "P_population": {"type": "string"},
                "I_exposure_intervention": {"type": "string"},
                "C_comparator": {"type": "string"},
                "O_outcomes": {"type": "string"},
                "S_study_design": {"type": "string"},
            },
            "required": ["P_population", "I_exposure_intervention", "C_comparator", "O_outcomes", "S_study_design"],
        },
        "search_terms_zh": {"type": "array", "items": {"type": "string"}},
        "search_terms_en": {"type": "array", "items": {"type": "string"}},
        "pubmed_query": {"type": "string"},
        "inclusion_criteria": {"type": "array", "items": {"type": "string"}},
        "exclusion_criteria": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "research_question",
        "pico",
        "search_terms_zh",
        "search_terms_en",
        "pubmed_query",
        "inclusion_criteria",
        "exclusion_criteria",
    ],
}

SCREEN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": ["include", "exclude", "uncertain"]},
        "reason": {"type": "string"},
        "study_object": {"type": "string"},
        "method": {"type": "string"},
        "sample_size": {"type": "string"},
        "outcomes": {"type": "string"},
        "main_findings": {"type": "string"},
        "evidence_gap_or_note": {"type": "string"},
    },
    "required": [
        "decision",
        "reason",
        "study_object",
        "method",
        "sample_size",
        "outcomes",
        "main_findings",
        "evidence_gap_or_note",
    ],
}

OUTLINE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "review_title": {"type": "string"},
        "core_argument": {"type": "string"},
        "outline": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "section": {"type": "string"},
                    "key_points": {"type": "array", "items": {"type": "string"}},
                    "supporting_evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["section", "key_points", "supporting_evidence"],
            },
        },
        "manual_review_flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["review_title", "core_argument", "outline", "manual_review_flags"],
}


def build_pico(topic: str) -> Dict[str, Any]:
    system = "你是循证医学和医学文献综述助手，擅长 PICO/PECO 拆解、检索式设计和纳排标准制定。"
    user = f"""
研究主题：{topic}
请完成：
1. 将主题拆解为 PICO/PECO 或适合综述的研究问题；
2. 给出中英文关键词；
3. 生成可用于 PubMed 的英文检索式；
4. 生成初步纳入和排除标准。
要求：医学研究场景，结果要可直接用于文献初筛。
"""
    return llm_json(system, user, "PICOResult", PICO_SCHEMA)


def pubmed_search(query: str, max_results: int = 20) -> List[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results,
        "sort": "relevance",
        "email": NCBI_EMAIL,
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    r = requests.get(PUBMED_SEARCH_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("esearchresult", {}).get("idlist", [])


def parse_article(article_node: ET.Element) -> Dict[str, str]:
    pmid = article_node.findtext(".//PMID", default="")
    title = " ".join(article_node.findtext(".//ArticleTitle", default="").split())

    abstract_parts = []
    for abs_node in article_node.findall(".//Abstract/AbstractText"):
        label = abs_node.attrib.get("Label")
        text = " ".join("".join(abs_node.itertext()).split())
        if text:
            abstract_parts.append(f"{label}: {text}" if label else text)
    abstract = "\n".join(abstract_parts)

    journal = article_node.findtext(".//Journal/Title", default="")
    year = (
        article_node.findtext(".//PubDate/Year")
        or article_node.findtext(".//ArticleDate/Year")
        or ""
    )

    authors = []
    for author in article_node.findall(".//Author")[:6]:
        last = author.findtext("LastName", default="")
        fore = author.findtext("ForeName", default="")
        name = " ".join([fore, last]).strip()
        if name:
            authors.append(name)

    return {
        "pmid": pmid,
        "title": title,
        "authors": "; ".join(authors),
        "journal": journal,
        "year": year,
        "abstract": abstract,
        "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
    }


def pubmed_fetch(pmids: List[str]) -> List[Dict[str, str]]:
    if not pmids:
        return []
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "email": NCBI_EMAIL,
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    r = requests.get(PUBMED_FETCH_URL, params=params, timeout=40)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    return [parse_article(node) for node in root.findall(".//PubmedArticle")]


def screen_article(article: Dict[str, str], pico: Dict[str, Any]) -> Dict[str, Any]:
    system = "你是医学文献综述初筛助手。你只做初筛和信息抽取，不替代研究者判断。"
    article_text = textwrap.shorten(
        f"标题：{article['title']}\n摘要：{article['abstract']}",
        width=2600,
        placeholder="...",
    )
    user = f"""
研究问题：{pico['research_question']}
PICO：{json.dumps(pico['pico'], ensure_ascii=False)}
纳入标准：{json.dumps(pico['inclusion_criteria'], ensure_ascii=False)}
排除标准：{json.dumps(pico['exclusion_criteria'], ensure_ascii=False)}

待筛选文献：
{article_text}

请判断 include/exclude/uncertain，并提取：研究对象、方法、样本量、结局指标、主要发现。
如果摘要信息不足，请 decision 设为 uncertain，并说明需人工复核。
"""
    result = llm_json(system, user, "ScreeningResult", SCREEN_SCHEMA)
    return {**article, **result}


def generate_outline(topic: str, pico: Dict[str, Any], evidence_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    system = "你是医学综述写作助手，擅长根据证据表生成综述框架和研究空白。"
    compact_rows = []
    for row in evidence_rows[:12]:
        compact_rows.append({
            "title": row.get("title", ""),
            "year": row.get("year", ""),
            "method": row.get("method", ""),
            "sample_size": row.get("sample_size", ""),
            "outcomes": row.get("outcomes", ""),
            "main_findings": row.get("main_findings", ""),
            "decision": row.get("decision", ""),
        })
    user = f"""
研究主题：{topic}
研究问题：{pico['research_question']}
PICO：{json.dumps(pico['pico'], ensure_ascii=False)}
证据表摘要：{json.dumps(compact_rows, ensure_ascii=False)}

请生成综述标题、核心论点、分级综述框架，并列出需要人工复核的问题。
"""
    return llm_json(system, user, "ReviewOutline", OUTLINE_SCHEMA)


st.set_page_config(page_title="医学文献综述辅助 Agent", layout="wide")
st.title("医学文献综述辅助 Agent")
st.caption("功能：PICO 拆解 → 检索式生成 → PubMed 检索 → 文献初筛 → 证据表 → 综述框架")

with st.sidebar:
    st.header("参数设置")
    max_results = st.slider("PubMed 检索数量", 5, 50, 15, 5)
    sleep_sec = st.slider("每篇筛选间隔秒数", 0.0, 2.0, 0.2, 0.1)
    st.write(f"当前模型：`{OPENAI_MODEL}`")
    st.warning("该工具仅用于科研初筛，所有纳排结论和综述内容必须人工复核。")

topic = st.text_area(
    "输入研究主题",
    value="老年非肌层浸润性膀胱癌术后患者自我管理困难的影响因素和护理干预研究",
    height=90,
)

run = st.button("开始运行 Agent", type="primary")

if run:
    if not topic.strip():
        st.error("请先输入研究主题")
        st.stop()

    with st.status("Agent 正在运行", expanded=True) as status:
        st.write("1/5 正在拆解 PICO 并生成检索式...")
        pico = build_pico(topic)
        st.subheader("PICO / 检索策略")
        st.json(pico)

        st.write("2/5 正在检索 PubMed...")
        pmids = pubmed_search(pico["pubmed_query"], max_results=max_results)
        st.write(f"检索到 {len(pmids)} 条记录")

        st.write("3/5 正在获取摘要...")
        articles = pubmed_fetch(pmids)
        st.write(f"成功获取 {len(articles)} 篇摘要")

        st.write("4/5 正在逐篇初筛并抽取证据...")
        rows = []
        progress = st.progress(0)
        for i, article in enumerate(articles, start=1):
            try:
                rows.append(screen_article(article, pico))
            except Exception as e:
                rows.append({
                    **article,
                    "decision": "uncertain",
                    "reason": f"模型处理失败，需人工复核：{e}",
                    "study_object": "",
                    "method": "",
                    "sample_size": "",
                    "outcomes": "",
                    "main_findings": "",
                    "evidence_gap_or_note": "",
                })
            progress.progress(i / max(len(articles), 1))
            time.sleep(sleep_sec)

        df = pd.DataFrame(rows)
        st.subheader("证据表")
        display_cols = [
            "decision", "reason", "title", "year", "journal", "study_object",
            "method", "sample_size", "outcomes", "main_findings",
            "evidence_gap_or_note", "pubmed_url"
        ]
        st.dataframe(df[display_cols], use_container_width=True)

        st.write("5/5 正在生成综述框架...")
        included_or_uncertain = [r for r in rows if r.get("decision") in ["include", "uncertain"]]
        outline = generate_outline(topic, pico, included_or_uncertain)
        st.subheader("综述框架")
        st.json(outline)

        status.update(label="Agent 运行完成", state="complete")

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下载证据表 CSV",
        data=csv,
        file_name="medical_literature_evidence_table.csv",
        mime="text/csv",
    )

    st.markdown("### 可复制的综述框架")
    st.markdown(f"**题目：** {outline['review_title']}")
    st.markdown(f"**核心论点：** {outline['core_argument']}")
    for item in outline["outline"]:
        st.markdown(f"#### {item['section']}")
        st.markdown("**要点：**")
        for p in item["key_points"]:
            st.markdown(f"- {p}")
        st.markdown("**支撑证据：**")
        for ev in item["supporting_evidence"]:
            st.markdown(f"- {ev}")

    if outline.get("manual_review_flags"):
        st.markdown("### 需人工复核")
        for flag in outline["manual_review_flags"]:
            st.markdown(f"- {flag}")
