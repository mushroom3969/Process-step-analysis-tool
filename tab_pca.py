"""Tab — PCA 分析（Hotelling T² + 貢獻分析）"""
import sys, os as _os
_dir = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_dir)
for _p in [_dir, _root, _os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from pca import pca
from scipy.stats import f as f_dist
from utils import extract_number


def _ht2_threshold(alpha, n, p):
    f_crit = f_dist.ppf(1 - alpha, p, n - p)
    return (p * (n - 1) * (n + 1)) / (n * (n - p)) * f_crit


def render(selected_process_df):
    st.header("PCA 主成分分析")
    _cd = st.session_state.get("clean_df")
    work_df = _cd if _cd is not None else selected_process_df

    if work_df is None:
        st.info("請先在側欄選擇製程步驟。")
        return

    numeric_options = work_df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_options) < 2:
        st.warning("有效數值欄位不足 2 個。"); return

    stored_target = st.session_state.get("target_col")
    default_idx   = numeric_options.index(stored_target) if stored_target in numeric_options else 0

    with st.expander("⚙️ PCA 設定", expanded=True):
        pc1, pc2, pc3 = st.columns(3)
        target_col_pca = pc1.selectbox("排除的目標欄位（Y）", numeric_options,
                                        index=default_idx, key="pca_target")
        n_components   = pc2.slider("最大主成分數", 2, min(15, len(numeric_options)-1), 5)
        alpha_pca      = pc3.select_slider("Hotelling T² 顯著水準 α",
                                            [0.01, 0.05, 0.10], value=0.05)

    if st.button("🧩 執行 PCA", key="run_pca"):
        try:
            exclude   = [c for c in ["BatchID", target_col_pca] if c in work_df.columns]
            X_pca     = work_df.drop(columns=exclude, errors="ignore").select_dtypes(include=["number"]).dropna(axis=1)
            if X_pca.shape[1] < 2:
                st.error("有效欄位不足 2 個。"); return

            labels      = work_df["BatchID"].values if "BatchID" in work_df.columns else np.arange(len(work_df))
            x_scaled    = StandardScaler().fit_transform(X_pca)
            n_comp      = min(n_components, X_pca.shape[1] - 1)

            with st.spinner("PCA 計算中..."):
                model   = pca(n_components=n_comp, alpha=alpha_pca, detect_outliers=["ht2","spe"])
                results = model.fit_transform(x_scaled)

            st.session_state.update({
                "pca_model": model, "pca_results": results,
                "pca_X": X_pca, "pca_x_scaled": x_scaled,
                "pca_labels": labels, "pca_feat": X_pca.columns.tolist(),
            })
            st.success("✅ PCA 完成！")
        except Exception as e:
            st.error(f"PCA 失敗：{e}")

    if st.session_state.get("pca_model") is None:
        return

    model      = st.session_state["pca_model"]
    results    = st.session_state["pca_results"]
    X_pca      = st.session_state["pca_X"]
    x_scaled   = st.session_state["pca_x_scaled"]
    labels     = st.session_state["pca_labels"]
    feat_names = st.session_state["pca_feat"]
    ev         = model.results["explained_var"]
    vr         = model.results["variance_ratio"]
    scores     = results["PC"].values
    loadings   = model.results["loadings"].values
    n_pc       = scores.shape[1]

    cols_m = st.columns(min(n_pc, 5))
    for i, c in enumerate(cols_m):
        c.metric(f"PC{i+1} 累計解釋", f"{ev[i]*100:.1f}%", delta=f"+{vr[i]*100:.1f}%")

    subtabs = st.tabs(["📊 Scree & Scatter","🔵 Biplot","🚨 Hotelling T² 異常偵測","🔬 單筆貢獻分析"])

    with subtabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Scree Plot**")
            fig, _ = model.plot(); st.pyplot(fig); plt.close()
        with c2:
            st.markdown("**Score Scatter**")
            fig, _ = model.scatter(SPE=True, HT2=True); st.pyplot(fig); plt.close()

    with subtabs[1]:
        bp1, bp2 = st.columns(2)
        n_feat_bi = bp1.slider("載荷向量數", 3, min(15, len(feat_names)), 6, key="bi_nfeat")
        pc_opts   = list(range(n_pc))
        pc_x = bp2.selectbox("X 軸 PC", pc_opts, index=0, format_func=lambda x: f"PC{x+1}", key="bi_pcx")
        pc_y = bp2.selectbox("Y 軸 PC", pc_opts, index=min(1,n_pc-1), format_func=lambda x: f"PC{x+1}", key="bi_pcy")
        fig, _ = model.biplot(n_feat=n_feat_bi, PC=[pc_x,pc_y], legend=True, SPE=True, HT2=True)
        st.pyplot(fig); plt.close()
        topfeat = model.results.get("topfeat")
        if topfeat is not None:
            tf = topfeat.copy()
            tf["feature_name"] = tf["feature"].apply(
                lambda x: feat_names[int(x)] if str(x).isdigit() and int(x) < len(feat_names) else str(x))
            st.dataframe(tf, width="stretch", hide_index=True)

    with subtabs[2]:
        st.markdown("#### Hotelling T² — 各 Batch 異常程度")
        pca_show_mean = st.checkbox("顯示 T² 平均線", value=True, key="pca_show_mean_ht2")
        ht2_vals    = np.sum((scores**2)/ev, axis=1)
        n_obs       = x_scaled.shape[0]
        thres_68    = _ht2_threshold(0.32, n_obs, n_pc)
        thres_95    = _ht2_threshold(0.05, n_obs, n_pc)
        thres_99    = _ht2_threshold(0.01, n_obs, n_pc)

        m1,m2,m3 = st.columns(3)
        m1.metric("超過 68% 閾值", f"{(ht2_vals>thres_68).sum()} 批")
        m2.metric("超過 95% 閾值", f"{(ht2_vals>thres_95).sum()} 批")
        m3.metric("超過 99% 閾值", f"{(ht2_vals>thres_99).sum()} 批")

        idx_sorted = np.argsort([extract_number(str(b)) for b in labels])
        bar_colors = ["#e84855" if v>thres_99 else "#f4a261" if v>thres_95
                      else "#e9c46a" if v>thres_68 else "#2e86ab"
                      for v in ht2_vals[idx_sorted]]

        fig, ax = plt.subplots(figsize=(14,5))
        ax.bar(range(len(labels)), ht2_vals[idx_sorted], color=bar_colors, alpha=0.85, width=0.7)
        for th, col, lbl in [(thres_68,"#e9c46a",f"68% ({thres_68:.1f})"),
                              (thres_95,"#f4a261",f"95% ({thres_95:.1f})"),
                              (thres_99,"#e84855",f"99% ({thres_99:.1f})")]:
            ax.axhline(th, color=col, linestyle="--", lw=1.5, label=lbl)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([str(labels[i])[-6:] for i in idx_sorted], rotation=90, fontsize=7)
        ax.set_ylabel("Hotelling T²"); ax.set_title("Hotelling T² per Batch")
        ax.legend(title="Confidence Level"); ax.grid(axis="y", linestyle="--", alpha=0.4)
        if pca_show_mean:
            mu_ht2 = float(np.mean(ht2_vals))
            ax.axhline(mu_ht2, color="#7209b7", linewidth=1.2, linestyle=":",
                       label=f"Mean T²={mu_ht2:.2f}", alpha=0.85)
            ax.legend(title="Confidence Level")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        ht2_df = pd.DataFrame({"Batch": labels, "T²": ht2_vals.round(3)})
        ht2_df["Status"] = ht2_df["T²"].apply(
            lambda v: "🔴 >99%" if v>thres_99 else "🟠 >95%" if v>thres_95
            else "🟡 >68%" if v>thres_68 else "🟢 Normal")
        st.dataframe(ht2_df.sort_values("T²", ascending=False).reset_index(drop=True),
                     width="stretch", hide_index=True)

    with subtabs[3]:
        st.markdown("#### 選擇要分析的 Batch")
        batch_opts = [str(b) for b in labels]
        sel_batch  = st.selectbox("選擇 Batch", batch_opts, key="pca_sel_batch")
        sample_i   = batch_opts.index(sel_batch)
        view_mode  = st.radio("分析模式",
                               ["所有 PC 的特徵貢獻（總 T²）","單一 PC 的特徵貢獻"],
                               horizontal=True, key="pca_view_mode")
        top_n = st.slider("顯示前 N 個特徵", 5, min(30, len(feat_names)), 15, key="pca_top_contrib")

        if view_mode == "所有 PC 的特徵貢獻（總 T²）":
            contribs = np.zeros(loadings.shape[1])
            for a in range(n_pc):
                contribs += (scores[sample_i,a]/ev[a]) * loadings[a,:] * x_scaled[sample_i,:]
            df_c = pd.DataFrame({"Feature": feat_names, "Contribution": contribs})
            df_c = df_c.reindex(pd.Series(contribs).abs().sort_values(ascending=False).index).reset_index(drop=True).head(top_n)
            fig, ax = plt.subplots(figsize=(12, max(5, top_n*0.4)))
            ax.barh(df_c["Feature"], df_c["Contribution"],
                    color=["#e84855" if v>0 else "#2e86ab" for v in df_c["Contribution"]], alpha=0.85)
            ax.axvline(0, color="black", lw=1)
            ax.set_title(f"Total T² Contribution — {sel_batch}", fontsize=13)
            ax.invert_yaxis(); ax.grid(axis="x", linestyle="--", alpha=0.5)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.dataframe(df_c, width="stretch", hide_index=True)
            # PC decomposition
            t2_per_pc = (scores[sample_i,:]**2)/ev
            fig2, ax2 = plt.subplots(figsize=(8,4))
            ax2.bar([f"PC{j+1}" for j in range(n_pc)], t2_per_pc, color="#2e86ab", alpha=0.8)
            for bar, val in zip(ax2.patches, t2_per_pc):
                ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                         f"{val:.2f}", ha="center", va="bottom", fontsize=9)
            ax2.set_title(f"PC-wise T² Decomposition — {sel_batch}")
            ax2.grid(axis="y", linestyle="--", alpha=0.4)
            plt.tight_layout(); st.pyplot(fig2); plt.close()
        else:
            sel_pc = st.selectbox("選擇 PC", list(range(n_pc)),
                                   format_func=lambda x: f"PC{x+1}", key="pca_sel_pc")
            pc_c  = (scores[sample_i,sel_pc]/ev[sel_pc]) * loadings[sel_pc,:] * x_scaled[sample_i,:]
            df_pc = pd.DataFrame({"Feature": feat_names, "Contribution": pc_c})
            df_pc = df_pc.reindex(pd.Series(pc_c).abs().sort_values(ascending=False).index).reset_index(drop=True).head(top_n)
            fig, ax = plt.subplots(figsize=(12, max(5, top_n*0.4)))
            ax.barh(df_pc["Feature"], df_pc["Contribution"],
                    color=["#e84855" if v>0 else "#2e86ab" for v in df_pc["Contribution"]], alpha=0.85)
            ax.axvline(0, color="black", lw=1)
            ax.set_title(f"PC{sel_pc+1} Contribution — {sel_batch}", fontsize=13)
            ax.invert_yaxis(); ax.grid(axis="x", linestyle="--", alpha=0.5)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.dataframe(df_pc, width="stretch", hide_index=True)
