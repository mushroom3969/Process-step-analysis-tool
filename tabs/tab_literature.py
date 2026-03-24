"""Tab 7 — 文獻佐證分析（PubMed + Gemini）"""
import sys, os as _os
_dir = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_dir)
for _p in [_dir, _root, _os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import re
import time
import streamlit as st

from utils import (
    search_pubmed_for_features,
    call_gemini,
    build_literature_prompt,
)


def _get_api_key() -> str:
    import os
    return st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))


def _render_article_list(all_articles: dict):
    st.markdown("#### 📄 找到的論文")
    for feat, data in all_articles.items():
        with st.expander(
            f"**{feat}** — {len(data['articles'])} 篇 | 搜尋：`{data['query']}`"
        ):
            for art in data["articles"]:
                st.markdown(
                    f"**[{art['title']}]({art['url']})**  \n"
                    f"_{art['journal']}_ ({art['year']}) · PMID: `{art['pmid']}`  \n"
                    + art["abstract"] + "..."
                )
                st.markdown("---")


def _render_analysis_result(params: dict, ref_list: list[str], response: str, lang: str):
    st.markdown("---")
    st.markdown("### 📄 分析結果")
    st.caption(
        f"目標：`{params.get('target', '')}` ｜ "
        f"參數數：{len(params.get('features', []))} ｜ "
        f"引用論文：{len(ref_list)} 篇"
    )
    st.markdown(response)

    if ref_list:
        st.markdown("### 🔗 快速論文連結")
        for ref in ref_list:
            url_m = re.search(r"https://pubmed[^\s]+", ref)
            if url_m:
                label = ref.split(url_m.group())[0].rstrip(". ")
                st.markdown(f"- {label} [🔗 PubMed]({url_m.group()})")

    # 匯出
    st.markdown("---")
    export_md = (
        "# 文獻佐證分析報告\n"
        f"生成時間：{time.strftime('%Y-%m-%d %H:%M')}\n\n"
        f"## 製程背景\n{params.get('context', '')}\n\n"
        f"## 目標變數\n{params.get('target', '')}\n\n"
        "## 分析參數\n"
        + "\n".join(f"- {f}" for f in params.get("features", []))
        + "\n\n---\n\n"
        "## AI 分析結果（基於 PubMed 真實文獻）\n\n"
        + response
        + "\n\n---\n\n## 所有搜尋到的論文\n\n"
        + "\n".join(ref_list)
    )
    st.download_button(
        "📥 下載完整報告（含文獻出處）",
        data=export_md.encode("utf-8"),
        file_name="pubmed_literature_report.md",
        mime="text/markdown",
        key="download_lit",
    )

    # 追問
    st.markdown("### 💬 追問")
    follow_up = st.text_area(
        "針對以上分析，想進一步了解？",
        placeholder="例：Loading capacity 對 yield 的非線性效應，文獻中有哪些實驗數據？",
        key="lit_followup_q",
    )
    if st.button("📨 送出追問", key="lit_followup_btn"):
        if follow_up.strip():
            api_key = _get_api_key()
            with st.spinner("思考中..."):
                try:
                    lang_inst = (
                        "Respond in Traditional Chinese."
                        if lang == "繁體中文"
                        else "Respond in English."
                    )
                    fu_prompt = (
                        f"Based on this previous analysis:\n\n{response}\n\n"
                        f"User follow-up: {follow_up}\n\n"
                        "Answer using ONLY the cited literature. "
                        f"If more papers are needed, say so clearly.\n{lang_inst}"
                    )
                    reply = call_gemini(api_key, fu_prompt, max_tokens=2000)
                    st.markdown("#### 💬 回覆")
                    st.markdown(reply)
                except Exception as e:
                    st.error(f"追問失敗：{e}")


def render():
    st.header("📚 文獻佐證分析")
    st.markdown("""
    <div class="info-box">
    自動從 <b>PubMed</b> 搜尋真實論文 → 抓取摘要 → Gemini 基於真實文獻整理分析。
    所有引用都附 PMID 連結，可直接追蹤原始論文。
    </div>
    """, unsafe_allow_html=True)

    # ── Step 1：參數設定 ──────────────────────────────────────
    st.markdown("### Step 1：設定分析參數")
    col_la, col_lb = st.columns(2)

    with col_la:
        target_var_lit = st.text_input(
            "🎯 目標變數（Y）",
            value=st.session_state.get("target_col", ""),
            placeholder="例：phenyl chromatography_Yield Rate (%)",
        )
        process_context = st.text_input(
            "🧪 製程背景",
            value="rhG-CSF protein purification, Phenyl Hydrophobic Interaction Chromatography",
        )
        max_papers_per_feat = st.slider("每個參數搜尋論文數", 1, 5, 3, key="lit_max_papers")

    with col_lb:
        # 從 RF / PLS 自動帶入重要特徵
        auto_features: list[str] = []
        if st.session_state.get("fi_perm_df") is not None:
            auto_features = st.session_state["fi_perm_df"]["Feature"].head(8).tolist()
        if st.session_state.get("pls_vip_df") is not None:
            vip_top = (
                st.session_state["pls_vip_df"]
                [st.session_state["pls_vip_df"]["VIP"] >= 1.0]["Feature"]
                .head(8)
                .tolist()
            )
            auto_features = list(dict.fromkeys(auto_features + vip_top))

        important_features_text = st.text_area(
            "📌 重要參數（每行一個）",
            value="\n".join(auto_features[:6]) if auto_features else "",
            height=200,
        )
        lang_lit = st.radio("輸出語言", ["繁體中文", "English"], horizontal=True, key="lit_lang")

    important_features = [
        f.strip() for f in important_features_text.strip().split("\n") if f.strip()
    ]

    # ── Step 2：PubMed 搜尋 ───────────────────────────────────
    st.markdown("### Step 2：搜尋 PubMed 文獻")

    if not important_features or not target_var_lit.strip():
        st.warning("請填入目標變數與重要參數。")
        return

    if st.button("🔎 搜尋 PubMed 論文", key="run_pubmed"):
        api_key = _get_api_key()

        prog = st.progress(0, text="搜尋中...")

        def _progress(done, total, feat):
            prog.progress(done / total, text=f"搜尋中：{feat[:40]}...")

        with st.spinner("🤖 Gemini 正在將參數名稱轉換為 PubMed 搜尋關鍵字..."):
            all_articles = search_pubmed_for_features(
                important_features, target_var_lit, process_context,
                api_key, max_papers_per_feat, progress_callback=_progress,
            )

        prog.empty()
        st.session_state["pubmed_results"] = all_articles
        total = sum(len(v["articles"]) for v in all_articles.values())
        st.success(f"✅ 共找到 {total} 篇論文！")

    if not st.session_state.get("pubmed_results"):
        return

    all_articles = st.session_state["pubmed_results"]
    _render_article_list(all_articles)

    # ── Step 3：Gemini 分析 ───────────────────────────────────
    st.markdown("### Step 3：AI 基於真實文獻分析")

    if st.button("🧠 開始 AI 文獻分析", type="primary", key="run_lit_gemini"):
        api_key = _get_api_key()
        if not api_key:
            st.error("找不到 GEMINI_API_KEY，請在 Streamlit Secrets 設定。")
            return

        prompt, ref_list = build_literature_prompt(
            all_articles, important_features, target_var_lit, process_context, lang_lit
        )

        with st.spinner("🧠 Gemini 正在基於真實文獻分析（約 30-60 秒）..."):
            try:
                ai_response = call_gemini(api_key, prompt)
                st.session_state["lit_response"] = ai_response
                st.session_state["lit_ref_list"] = ref_list
                st.session_state["lit_params"] = {
                    "target": target_var_lit,
                    "features": important_features,
                    "context": process_context,
                }
                st.success("✅ 分析完成！")
            except Exception as e:
                import traceback
                st.error(f"Gemini 呼叫失敗：{e}")
                st.code(traceback.format_exc())

    # ── 顯示分析結果 ─────────────────────────────────────────
    if st.session_state.get("lit_response"):
        _render_analysis_result(
            params=st.session_state.get("lit_params", {}),
            ref_list=st.session_state.get("lit_ref_list", []),
            response=st.session_state["lit_response"],
            lang=lang_lit,
        )
