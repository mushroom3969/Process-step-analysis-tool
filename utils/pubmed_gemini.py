"""
PubMed 搜尋 + Gemini 文獻分析工具
"""

import json
import re
import time
import urllib.parse as uparse
import urllib.request as ureq


GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_BASE = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


def pubmed_search(query: str, max_results: int = 5) -> list[str]:
    """查詢 PubMed，回傳 PMID 列表。"""
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&retmode=json&retmax={max_results}&term={uparse.quote(query)}"
    )
    try:
        with ureq.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        return data["esearchresult"].get("idlist", [])
    except Exception:
        return []


def pubmed_fetch_abstracts(pmids: list[str]) -> list[dict]:
    """依 PMID 清單抓取摘要，回傳文章資訊列表。"""
    if not pmids:
        return []
    fetch_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pubmed&id={','.join(pmids)}&rettype=abstract&retmode=xml"
    )
    try:
        with ureq.urlopen(fetch_url, timeout=15) as r:
            xml = r.read().decode("utf-8")
    except Exception:
        return []

    def _clean(s: str | None) -> str:
        return re.sub(r"<[^>]+>", "", s).strip() if s else ""

    articles = []
    for block in re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml, re.DOTALL):
        pmid    = (re.search(r"<PMID[^>]*>(\d+)</PMID>", block) or [None, "?"])[1]
        title   = _clean((re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", block, re.DOTALL) or [None, None])[1])
        abstract= _clean((re.search(r"<AbstractText[^>]*>(.*?)</AbstractText>", block, re.DOTALL) or [None, None])[1])
        year    = (re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", block, re.DOTALL) or [None, "?"])[1]
        journal = _clean((re.search(r"<ISOAbbreviation>(.*?)</ISOAbbreviation>", block) or [None, None])[1])
        articles.append({
            "pmid": pmid,
            "title": title or "No title",
            "abstract": (abstract or "No abstract")[:800],
            "year": year,
            "journal": journal or "?",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return articles


def build_search_queries_with_gemini(
    features: list[str],
    target: str,
    context: str,
    api_key: str,
) -> list[tuple[str, str]]:
    """
    用 Gemini 將參數名稱轉換為 PubMed 搜尋關鍵字。
    失敗時退回簡易規則式關鍵字。
    回傳 [(param_name, query), ...] 列表。
    """
    feat_list = "\n".join(f"- {f}" for f in features)
    prompt = (
        "You are a bioprocess scientist. Convert each process parameter name "
        "into a concise PubMed search query (3-6 words) using standard scientific terminology. "
        f"Process context: {context} Target: {target} "
        f"Parameters:\n{feat_list}\n"
        "Also add 2 broad topic queries. "
        'Reply ONLY with a JSON array like: [{"param":"X","query":"HIC protein yield"}] '
        "No markdown, no explanation."
    )
    try:
        url = f"{GEMINI_BASE}?key={api_key}"
        payload = json.dumps({
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 1000, "temperature": 0.1},
        }).encode("utf-8")
        req = ureq.Request(url, data=payload,
                           headers={"Content-Type": "application/json"}, method="POST")
        with ureq.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        raw = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        raw = re.sub(r"^```[\w]*\n?|```$", "", raw).strip()
        parsed = json.loads(raw)
        qs = [(item["param"], item["query"]) for item in parsed
              if "param" in item and "query" in item]
        if qs:
            return qs
    except Exception:
        pass

    # 退回規則式
    proc_kw = context.split(",")[0].strip() if context else "bioprocess"
    qs = []
    for feat in features:
        clean = re.sub(r"\(.*?\)", "", feat).replace("_", " ").replace("-", " ")
        words = [w for w in clean.split() if len(w) > 3]
        kw = " ".join(words[:4])
        if kw:
            qs.append((feat, f"{kw} {proc_kw} chromatography yield"))
    qs.append(("Overall", f"{proc_kw} yield optimization process parameters"))
    return qs


def call_gemini(api_key: str, prompt: str, max_tokens: int = 6000) -> str:
    """呼叫 Gemini API 並回傳文字回應。"""
    url = f"{GEMINI_BASE}?key={api_key}"
    payload = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2},
    }).encode("utf-8")
    req = ureq.Request(url, data=payload,
                       headers={"Content-Type": "application/json"}, method="POST")
    with ureq.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    return result["candidates"][0]["content"]["parts"][0]["text"]


def search_pubmed_for_features(
    features: list[str],
    target: str,
    context: str,
    api_key: str,
    max_papers: int = 3,
    progress_callback=None,
) -> dict[str, dict]:
    """
    完整搜尋流程：Gemini 建立查詢 → PubMed 搜尋 → 抓取摘要。
    回傳 {feature: {"query": str, "articles": list}} 字典。
    """
    queries = build_search_queries_with_gemini(features, target, context, api_key)
    all_articles: dict[str, dict] = {}

    for i, (feat, query) in enumerate(queries):
        pmids = pubmed_search(query, max_results=max_papers)
        if pmids:
            arts = pubmed_fetch_abstracts(pmids)
            all_articles[feat] = {"query": query, "articles": arts}
        time.sleep(0.35)
        if progress_callback:
            progress_callback(i + 1, len(queries), feat)

    return all_articles


def build_literature_prompt(
    all_articles: dict,
    important_features: list[str],
    target: str,
    context: str,
    lang: str = "繁體中文",
) -> tuple[str, list[str]]:
    """
    組裝 Gemini 文獻分析 prompt。
    回傳 (prompt_str, ref_list)
    """
    lit_context = ""
    ref_list: list[str] = []
    ref_idx = 1

    for feat, data in all_articles.items():
        lit_context += f"\n\n=== Parameter: {feat} ===\n"
        for art in data["articles"]:
            lit_context += (
                f"[{ref_idx}] {art['title']} ({art['journal']}, {art['year']}, PMID:{art['pmid']})\n"
                f"Abstract: {art['abstract']}\n\n"
            )
            ref_list.append(
                f"[{ref_idx}] {art['title']}. {art['journal']} ({art['year']}). "
                f"PMID: {art['pmid']}. {art['url']}"
            )
            ref_idx += 1

    lang_inst = (
        "Respond in Traditional Chinese (繁體中文)."
        if lang == "繁體中文"
        else "Respond in English."
    )
    feat_list_str = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(important_features))

    prompt = (
        "You are an expert bioprocess scientist. Analyze the following real PubMed literature "
        "to explain why the listed process parameters are important predictors of the target variable.\n\n"
        f"Process Context: {context}\n"
        f"Target Variable: {target}\n\n"
        f"Important Parameters:\n{feat_list_str}\n\n"
        f"=== REAL PUBMED LITERATURE ===\n{lit_context}\n\n"
        "STRICT RULES:\n"
        "1. ONLY cite papers from the list above using [number]\n"
        "2. Do NOT invent papers\n"
        "3. If evidence is weak or absent, say so\n"
        "4. End with a Reference List with PMID and URL\n"
        f"5. {lang_inst}\n\n"
        "Output:\n"
        "## 總覽\n(collective effect on target)\n\n"
        "## 各參數分析\n(each param: mechanism + [ref])\n\n"
        "## 研究缺口\n(what is NOT covered by found literature)\n\n"
        "## 參考文獻\n(all cited with PMID + URL)"
    )
    return prompt, ref_list
