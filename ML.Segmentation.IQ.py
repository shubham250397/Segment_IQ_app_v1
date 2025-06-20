import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, dendrogram
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ─── GLOBAL STYLING ───
st.set_page_config(page_title="Segmentation IQ", layout="wide")
st.markdown("""
<style>
.banner {background:#fff;padding:1rem;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.1);margin-bottom:1rem;text-align:center;}
.card {background:#f9f9f9;padding:1rem;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,0.08);margin-bottom:1rem;}
.tab-desc {color:#555;margin-bottom:1rem;}
</style>
""", unsafe_allow_html=True)

# ─── BANNER ───
st.markdown("""
<div class="banner">
  <h1>🧠 Segmentation IQ</h1>
  <p><em>Clustering & Interpretability, end-to-end</em></p>
</div>
""", unsafe_allow_html=True)

# ─── SESSION STATE INIT ───
for key in (
    "df_raw","feat_cols","df_proc","df_sel","dropped_corr",
    "X_pca","X_red","model","k","labels","ctrs_pca","ctrs_orig",
    "metrics","run_done","suggested_k","scenarios"
):
    if key not in st.session_state:
        st.session_state[key] = [] if key=="scenarios" else None

# ─── TABS ───
tab1, tab2, tab3 = st.tabs([
  "🛠 Configure & Run",
  "🔍 Interpret & Save",
  "📊 Compare Scenarios"
])

# ─── TAB 1: CONFIGURE & RUN ───
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="tab-desc">Upload CSV, preprocess, then pick & run a clustering model.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("CSV with `Candidate_ID` + numeric features", type="csv")
    if not uploaded:
        st.info("Awaiting data upload…")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        df = pd.read_csv(uploaded)
        if "Candidate_ID" not in df.columns:
            st.error("❌ Missing `Candidate_ID`."); st.markdown('</div>', unsafe_allow_html=True); st.stop()

        feats = [c for c in df.columns if c!="Candidate_ID"]
        if not feats or not all(np.issubdtype(df[c].dtype,np.number) for c in feats):
            st.error("❌ All other columns must be numeric."); st.markdown('</div>', unsafe_allow_html=True); st.stop()

        # store raw
        st.session_state.df_raw = df.copy()
        st.session_state.feat_cols = feats

        # 1) Impute
        with st.spinner("Imputing missing values…"):
            imp = SimpleImputer(strategy="mean")
            arr = imp.fit_transform(df[feats])
            df_proc = pd.DataFrame(arr,columns=feats)
        st.session_state.df_proc = df_proc

        # 2) Log-transform if skewed & outside [0,1]
        skewed=[]
        with st.spinner("Log-transforming skewed…"):
            for c in feats:
                col = df_proc[c]
                if not (col.min()>=0 and col.max()<=1) and col.skew()>1:
                    df_proc[c] = np.log1p(col - col.min() + 1e-3)
                    skewed.append(c)

        # 3) Scale to [0,1]
        with st.spinner("Scaling…"):
            mms = MinMaxScaler()
            df_proc[feats] = mms.fit_transform(df_proc[feats])

        # 4) Drop correlated
        corr_thr = st.slider("Corr threshold to drop features",0.80,0.99,0.95,0.01)
        corr = df_proc.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape),1).astype(bool))
        dropped = [col for col in upper.columns if upper[col].max()>corr_thr]
        df_sel = df_proc.drop(columns=dropped)
        st.session_state.dropped_corr = dropped

        # 5) VarianceThreshold
        vt_thr = st.slider("Variance threshold",0.0,0.1,0.0,0.01)
        vt = VarianceThreshold(vt_thr)
        kept = df_sel.columns[vt.fit(df_sel).get_support()].tolist()
        df_sel = pd.DataFrame(vt.transform(df_sel),columns=kept)
        st.session_state.df_sel = df_sel

        # Show selection
        st.subheader("🧪 Feature Selection")
        with st.expander("Dropped vs Kept"):
            st.write(f"Dropped (corr>{corr_thr}): {dropped}")
            st.write(f"Kept  (var>{vt_thr}): {kept}")

        # PCA
        with st.spinner("Running PCA…"):
            pca = PCA()
            Xp = pca.fit_transform(df_sel)
            evr = pca.explained_variance_ratio_
            cum = np.cumsum(evr)
            n_comp = int(np.argmax(cum>=0.95)+1)
        st.session_state.X_pca = Xp
        st.session_state.X_red = Xp[:,:n_comp]

        st.markdown("---")
        st.subheader(f"📈 PCA retained {n_comp} components (≥95% var)")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(1,len(evr)+1)),y=evr,name="Explained Var"))
        fig.add_trace(go.Scatter(x=list(range(1,len(evr)+1)),y=cum,mode="lines+markers",name="Cumulative"))
        fig.update_layout(xaxis_title="Component",yaxis_title="Variance",plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

        # Model selection
        st.markdown("---")
        st.subheader("2. Select & Evaluate Model")
        model = st.selectbox("Algorithm",["K-Means","Gaussian Mixture","Hierarchical","DBSCAN"])
        st.session_state.model = model

        # Model-specific evaluation
        if model == "K-Means":
            ks=list(range(2,11)); wcss=[]; sils=[]
            with st.spinner("Elbow & Silhouette…"):
                for k_ in ks:
                    km = KMeans(n_clusters=k_,random_state=42).fit(st.session_state.X_red)
                    wcss.append(km.inertia_)
                    labels_ = km.labels_
                    unique,counts = np.unique(labels_,return_counts=True)
                    if len(unique)>1 and np.all(counts>=2):
                        sils.append(silhouette_score(st.session_state.X_red,labels_))
                    else:
                        sils.append(np.nan)
            c1,c2 = st.columns(2)
            c1.plotly_chart(px.line(x=ks,y=wcss,markers=True,labels={"x":"K","y":"WCSS"}))
            c2.plotly_chart(px.line(x=ks,y=sils,markers=True,labels={"x":"K","y":"Silhouette"}))
            best_k = ks[int(np.nanargmax(sils))]
            st.info(f"Suggested K by silhouette: {best_k}")
            st.session_state.suggested_k = best_k

        elif model == "Gaussian Mixture":
            ks=list(range(2,11)); aics=[]; bics=[]
            with st.spinner("AIC & BIC…"):
                for k_ in ks:
                    gm = GaussianMixture(n_components=k_,random_state=42).fit(st.session_state.X_red)
                    aics.append(gm.aic(st.session_state.X_red))
                    bics.append(gm.bic(st.session_state.X_red))
            c1,c2 = st.columns(2)
            c1.plotly_chart(px.line(x=ks,y=aics,markers=True,labels={"x":"K","y":"AIC"}))
            c2.plotly_chart(px.line(x=ks,y=bics,markers=True,labels={"x":"K","y":"BIC"}))

        elif model == "Hierarchical":
            with st.spinner("Dendrogram…"):
                Z = linkage(st.session_state.X_red,method="ward")
                fig,ax = plt.subplots(figsize=(8,3))
                dendrogram(Z,ax=ax,no_labels=True)
                ax.set_title("Dendrogram")
                st.pyplot(fig)

        else:  # DBSCAN
            from sklearn.neighbors import NearestNeighbors
            nbr = NearestNeighbors(n_neighbors=5).fit(st.session_state.X_red)
            dist,_ = nbr.kneighbors(st.session_state.X_red)
            d5 = np.sort(dist[:,4])
            st.subheader("k-distance (k=5)")
            st.line_chart(d5)

        # K slider if needed
        if model in ["K-Means","Gaussian Mixture","Hierarchical"]:
            k = st.slider("Number of clusters (K)",2,10,st.session_state.suggested_k or 4)
        else:
            k = None
        st.session_state.k = k

        # Clear & Restart
        if st.button("🔄 Clear & Restart"):
            for k in list(st.session_state.keys()):
                if k!="scenarios": del st.session_state[k]
            st.experimental_rerun()

        # Run clustering
        if st.button("🚀 Run Clustering"):
            prog = st.progress(0)
            Xr = st.session_state.X_red

            with st.spinner("Fitting model…"):
                if model == "K-Means":
                    m = KMeans(n_clusters=k,random_state=42)
                    labs = m.fit_predict(Xr); ctrs_pca = m.cluster_centers_
                elif model == "Gaussian Mixture":
                    m = GaussianMixture(n_components=k,random_state=42)
                    labs = m.fit_predict(Xr); ctrs_pca = m.means_
                elif model == "Hierarchical":
                    m = AgglomerativeClustering(n_clusters=k)
                    labs = m.fit_predict(Xr)
                    ctrs_pca = np.vstack([Xr[labs==i].mean(axis=0) for i in range(k)])
                else:
                    m = DBSCAN(eps=1.0,min_samples=5)
                    labs = m.fit_predict(Xr)
                    uniq = sorted(set(labs)-{-1})
                    ctrs_pca = np.vstack([Xr[labs==i].mean(axis=0) for i in uniq])
                prog.progress(40)

            # Assemble results
            df_out = st.session_state.df_raw.copy()
            df_out["Cluster"] = labs
            tsne = TSNE(n_components=2,random_state=42)
            t2 = tsne.fit_transform(Xr)
            df_out["tSNE1"],df_out["tSNE2"] = t2[:,0],t2[:,1]
            df_out["PC1"],df_out["PC2"] = st.session_state.X_pca[:,0],st.session_state.X_pca[:,1]

            st.session_state.labels = labs
            st.session_state.ctrs_pca = ctrs_pca
            st.session_state.ctrs_orig = pd.DataFrame(st.session_state.df_sel)\
                                          .groupby(pd.Series(labs,name="Cluster"))\
                                          .mean().sort_index().values
            st.session_state.res = df_out
            prog.progress(60)

            # Compute evaluation metrics safely
            with st.spinner("Computing metrics…"):
                unique,counts = np.unique(labs,return_counts=True)
                # Silhouette
                sil = (silhouette_score(Xr,labs)
                       if len(unique)>1 and np.all(counts>=2) else np.nan)
                # Calinski-Harabasz
                ch = (calinski_harabasz_score(Xr,labs)
                      if len(unique)>1 else np.nan)
                # Davies-Bouldin
                db = (davies_bouldin_score(Xr,labs)
                      if len(unique)>1 else np.nan)
            met = pd.DataFrame({
                "Metric": ["Silhouette","Calinski-Har","Davies-B"],
                "Score": [sil,ch,db]
            })
            st.session_state.metrics = met
            prog.progress(80)

            st.success("✅ Clustering Complete")
            st.markdown('</div>', unsafe_allow_html=True)

            # Distribution donut
            st.subheader("Cluster Distribution")
            dist = df_out["Cluster"].value_counts().reset_index()
            dist.columns=["Cluster","Count"]
            dist["Percent"] = dist["Count"]/len(df_out)*100
            figd = px.pie(dist,names="Cluster",values="Count",hole=0.6,title="Count & %")
            st.plotly_chart(figd,use_container_width=True)

            # Scatter plots
            c1,c2 = st.columns(2)
            c1.subheader("2D PCA"); c1.plotly_chart(
                px.scatter(df_out,x="PC1",y="PC2",color=df_out["Cluster"].astype(str),
                           hover_data=["Candidate_ID"]),use_container_width=True)
            c2.subheader("2D t-SNE"); c2.plotly_chart(
                px.scatter(df_out,x="tSNE1",y="tSNE2",color=df_out["Cluster"].astype(str),
                           hover_data=["Candidate_ID"]),use_container_width=True)

            # Metrics table & quality
            st.subheader("📈 Evaluation Metrics")
            st.dataframe(met,use_container_width=True)
            if not np.isnan(sil):
                quality = ("👍 Good" if sil>=0.5 else
                           "⚠️ Moderate" if sil>=0.25 else "❌ Poor")
                st.markdown(f"**Silhouette** = {sil:.2f} → {quality}")
            else:
                st.markdown("**Silhouette**: unable to compute (singleton/one cluster).")

            st.session_state.run_done = True

# ─── TAB 2: INTERPRET & SAVE ───
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="tab-desc">Dive into distances & feature‐contribution, then save your scenario.</div>', unsafe_allow_html=True)
    if not st.session_state.run_done:
        st.info("▶️ Run clustering first.")
    else:
        df_sel = st.session_state.df_sel
        labs   = st.session_state.labels
        ctrs   = st.session_state.ctrs_orig
        raw    = st.session_state.df_raw
        w1,w2  = 0.51,0.49
        K      = ctrs.shape[0]

        def compute_cluster_metrics(df_sel,labs,ctrs):
            recs=[]
            for idx, row in df_sel.iterrows():
                cid = raw.at[idx,"Candidate_ID"]
                assigned = labs[idx]
                vec = row.values
                d=[]; f=[]
                # Manhattan distance + feature-contribution
                for c in range(K):
                    mask = (~np.isnan(vec)) & (~np.isnan(ctrs[c]))
                    dist = np.nansum(np.abs(vec[mask]-ctrs[c][mask])) if mask.any() else np.inf
                    d.append(dist)
                    valid=0; contrib=0
                    for i,v in enumerate(vec):
                        if not np.isnan(v):
                            valid+=1
                            diff_sq = (v-ctrs[c,i])**2 if not np.isnan(ctrs[c,i]) else np.inf
                            if all(diff_sq <= (v-ctrs[oc,i])**2 for oc in range(K) if oc!=c):
                                contrib+=1
                    f.append((contrib/valid*100) if valid>0 else np.nan)
                rec={"Candidate_ID":cid,"Current_Cluster":assigned}
                for c in range(K):
                    rec[f"Distance_C{c+1}"] = d[c]
                    rec[f"FC_C{c+1}"]       = f[c]
                recs.append(rec)
            dfm=pd.DataFrame(recs)
            # ranks 1..K
            for c in range(K):
                dfm[f"Rank_Distance_C{c+1}"] = dfm[f"Distance_C{c+1}"]\
                    .rank(method="min",ascending=True).astype(int)
                dfm[f"Rank_FC_C{c+1}"]       = dfm[f"FC_C{c+1}"]\
                    .rank(method="min",ascending=False).astype(int)
                dfm[f"Combined_Rank_C{c+1}"]=(
                    w1*dfm[f"Rank_Distance_C{c+1}"] +
                    w2*dfm[f"Rank_FC_C{c+1}"]
                )
            combs=[f"Combined_Rank_C{i+1}" for i in range(K)]
            dfm["Suggested_Cluster"] = dfm[combs]\
                .idxmin(axis=1).str.extract(r"(\d+)").astype(int)
            dfm["cluster_match"] = dfm["Current_Cluster"]==dfm["Suggested_Cluster"]
            return dfm

        cm = compute_cluster_metrics(df_sel,labs,ctrs)
        st.subheader("🔬 Distance & Feature-Contribution")
        st.dataframe(cm,use_container_width=True)

        sel = st.selectbox("Select Candidate",cm["Candidate_ID"].tolist())
        prow = cm[cm["Candidate_ID"]==sel].iloc[0]
        combs = [f"Combined_Rank_C{i+1}" for i in range(K)]
        vals = prow[combs].values
        labs_str=[str(i+1) for i in range(K)]
        fig=go.Figure([go.Bar(
            x=labs_str,y=vals,
            marker_color=["green" if v==vals.min() else "steelblue" for v in vals]
        )])
        fig.update_layout(
            title=f"Combined Ranks for {sel}",
            xaxis_title="Cluster", yaxis_title="Rank",
            plot_bgcolor="white"
        )
        st.plotly_chart(fig,use_container_width=True)
        st.markdown(f"**Cluster Match:** {'✅' if prow['cluster_match'] else '❌'}")

        name = st.text_input("Scenario Name",value=f"Scenario {len(st.session_state.scenarios)+1}")
        if st.button("💾 Save Scenario"):
            sc = {
                "name":name,
                "model":st.session_state.model,
                "k":st.session_state.k,
                "silhouette":float(st.session_state.metrics.loc[0,"Score"]),
                "calinski":float(st.session_state.metrics.loc[1,"Score"]),
                "davies":float(st.session_state.metrics.loc[2,"Score"])
            }
            st.session_state.scenarios.append(sc)
            st.success(f"Saved `{name}`")
    st.markdown('</div>', unsafe_allow_html=True)

# ─── TAB 3: COMPARE SCENARIOS ───
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="tab-desc">Compare saved clustering scenarios side-by-side.</div>', unsafe_allow_html=True)
    scs = st.session_state.scenarios
    if not scs:
        st.info("No scenarios saved.")
    else:
        dfsc = pd.DataFrame(scs)
        st.dataframe(dfsc,use_container_width=True)
        pick = st.multiselect("Compare",dfsc["name"].tolist(),default=dfsc["name"][:2])
        if len(pick)>1:
            cmp = dfsc[dfsc["name"].isin(pick)].set_index("name")
            st.subheader("Scenario Metrics")
            st.bar_chart(cmp[["silhouette","calinski","davies"]])
        if st.button("🗑 Clear All"):
            st.session_state.scenarios=[]
            st.success("Cleared scenarios.")
    st.markdown('</div>', unsafe_allow_html=True)
