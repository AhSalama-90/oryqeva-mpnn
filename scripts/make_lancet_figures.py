"""Lancet-style figures (Liberation Sans = Helvetica metric clone) for OryqevaMPNN workbook.
Usage: python make_lancet_figures.py <workbook.xlsx> <outdir>"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
SEED=42; np.random.seed(SEED)
X, OUT = sys.argv[1], sys.argv[2]
avail={f.name for f in fm.fontManager.ttflist}
FONT=next(f for f in ["Helvetica","Arial","Liberation Sans"] if f in avail)
plt.rcParams.update({"font.family":FONT,"font.size":9,"axes.linewidth":0.7,"svg.fonttype":"none",
  "pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,"xtick.major.width":0.7,
  "ytick.major.width":0.7,"legend.frameon":False,"figure.dpi":150})
# Lancet palette
NAVY,RED,GREEN,TEAL,PURP,GREY="#00468B","#ED0000","#42B540","#0099B4","#925E9F","#ADB6B6"
def save(fig,name):
    fig.savefig(f"{OUT}/svg/{name}.svg",bbox_inches="tight")
    fig.savefig(f"{OUT}/png/{name}.png",dpi=900,bbox_inches="tight",facecolor="white")
    plt.close(fig)
def fp(p): return "<0·001" if p<0.001 else f"{p:.3f}".replace(".","·")
def fn(x): return f"{x:.3f}".replace(".","·").replace("-","−")
def forest(ax_labels, est, lo, hi, ptxt_cols, headers, title_x, fname, groups=None, pooled=None, xlim=None, sizes=None, figw=7.8):
    est,lo,hi=[np.asarray(v,float) for v in (est,lo,hi)]
    n=len(est); rows=n+(1 if pooled else 0)
    fig=plt.figure(figsize=(figw,0.26*rows+1.3))
    gs=fig.add_gridspec(1,3,width_ratios=[2.5,2.6,2.3],wspace=0.02)
    axl,axp,axr=[fig.add_subplot(g) for g in gs]
    y=np.arange(rows)[::-1]+0.0
    for a in (axl,axr): a.axis("off"); a.set_ylim(-1,rows+0.2)
    axp.set_ylim(-1,rows+0.2)
    for s in ("left","top","right"): axp.spines[s].set_visible(False)
    axp.set_yticks([]); axp.axvline(0,color="black",lw=0.7)
    lo_,hi_=xlim
    for i in range(n):
        yy=y[i]
        axp.plot([lo[i],hi[i]],[yy,yy],color="black",lw=0.9,solid_capstyle="butt")
        axp.scatter(est[i],yy,marker="s",s=(sizes[i] if sizes is not None else 22),color=NAVY,zorder=3,linewidths=0)
        axl.text(0,yy,ax_labels[i],va="center",ha="left")
        axr.text(0,yy,ptxt_cols[i],va="center",ha="left")
    if pooled:
        yy=y[-1]-0.0; e,l,h,lab,rt=pooled
        axp.fill([l,e,h,e],[yy,yy+0.32,yy,yy-0.32],color=RED,zorder=3,lw=0)
        axl.text(0,yy,lab,va="center",ha="left",fontweight="bold"); axr.text(0,yy,rt,va="center",ha="left",fontweight="bold")
        axp.axhline(yy+0.65,color=GREY,lw=0.5)
    axp.set_xlim(lo_,hi_); axp.set_xlabel(title_x)
    axl.text(0,rows+0.0,headers[0],fontweight="bold",va="bottom"); axr.text(0,rows+0.0,headers[1],fontweight="bold",va="bottom")
    axl.set_xlim(0,1); axr.set_xlim(0,1)
    return fig,axp
def t_ci(d,alpha=0.05):
    d=np.asarray(d,float); m=d.mean(); se=d.std(ddof=1)/np.sqrt(len(d)); t=stats.t.ppf(1-alpha/2,len(d)-1); return m,m-t*se,m+t*se

ct=pd.read_excel(X,sheet_name="Contrast Tests"); ts=pd.read_excel(X,sheet_name="Target Summary")
ps=pd.read_excel(X,sheet_name="Prediction Scores"); loo=pd.read_excel(X,sheet_name="Leave One Out")
av=pd.read_excel(X,sheet_name="Assay Validation"); meta={}

# ---- Fig 1: contrasts forest
ct=ct.copy(); lab={"SFT_vs_ProteinMPNN":"SFT vs ProteinMPNN","Preference + auxiliary SFT_vs_ProteinMPNN":"Pref + aux SFT vs ProteinMPNN",
 "Final (preference + SFT)_vs_ProteinMPNN":"Final vs ProteinMPNN","Final (preference + SFT)_vs_SFT":"Final vs SFT",
 "Final (preference + SFT)_vs_Preference + auxiliary SFT":"Final vs Pref + aux SFT","SFT_vs_Preference + auxiliary SFT":"SFT vs Pref + aux SFT"}
L=[lab[c] for c in ct.Comparison]; r=[f"{fn(m)} ({fn(a)} to {fn(b)})   {fp(p)}" for m,a,b,p in zip(ct["mean difference"],ct["95% CI lower"],ct["95% CI upper"],ct["Paired t p"])]
fig,ax=forest(L,ct["mean difference"],ct["95% CI lower"],ct["95% CI upper"],r,["Comparison (n=21)","Difference (95% CI)     p"],
  "Mean paired difference in i_pTM (first − second model)",None,xlim=(-0.08,0.18),sizes=[40]*6)
ax.text(-0.075,-0.55,"← Second higher",fontsize=7.5,ha="left"); ax.text(0.175,-0.55,"First higher →",fontsize=7.5,ha="right")
save(fig,"fig01_forest_paired_contrasts")

# ---- Fig 2: per-target forest Final vs ProteinMPNN, Welch CI from 5 designs/arm
g=ps.pivot_table(index="Target",columns="Model",values="i_pTM",aggfunc=list)
rows=[]
for t,rw in g.iterrows():
    a=np.array(rw["Final"],float); b=np.array(rw["ProteinMPNN"],float)
    d=a.mean()-b.mean(); va,vb=a.var(ddof=1)/len(a),b.var(ddof=1)/len(b); se=np.sqrt(va+vb)
    df=(va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1)); h=stats.t.ppf(.975,df)*se
    rows.append((t,d,d-h,d+h,len(a),len(b),1/se**2))
f2=pd.DataFrame(rows,columns=["target","diff","lo","hi","n_final","n_base","w"]).sort_values("diff",ascending=False)
chk=ts.set_index("Target")["Final minus baseline"]; assert np.allclose(f2.set_index("target")["diff"],chk.loc[f2.target].values,atol=1e-6)
f2.to_csv(f"{OUT}/tables/fig02_per_target_welch.csv",index=False)
pm=ct.loc[ct.Comparison.str.startswith("Final (preference + SFT)_vs_ProteinMPNN"),["mean difference","95% CI lower","95% CI upper"]].iloc[0]
w=f2.w.values; sz=20+60*(w/w.max())
r=[f"{fn(d)} ({fn(l)} to {fn(h)})" for d,l,h in zip(f2["diff"],f2.lo,f2.hi)]
fig,ax=forest(list(f2.target),f2["diff"],f2.lo,f2.hi,r,["Target (5 designs per arm)","Difference (95% CI)"],
  "Final − ProteinMPNN mean i_pTM",None,xlim=(-0.6,0.8),sizes=sz,
  pooled=(pm.iloc[0],pm.iloc[1],pm.iloc[2],"Overall (paired, 21 targets)",f"{fn(pm.iloc[0])} ({fn(pm.iloc[1])} to {fn(pm.iloc[2])})"))
save(fig,"fig02_forest_per_target_final_vs_proteinmpnn")

# ---- Fig 3: leave-one-out forest (t-CI from the 21 target-level paired differences)
d21=(ts.set_index("Target")["Final minus baseline"]); rows=[]
for t in d21.index:
    m,l,h=t_ci(d21.drop(t)); rows.append((t,m,l,h))
f3=pd.DataFrame(rows,columns=["removed","m","lo","hi"]).merge(loo.rename(columns={"removed_target":"removed"}),on="removed")
assert np.allclose(f3.m,f3.mean_difference,atol=1e-6)
f3=f3.sort_values("m"); f3.to_csv(f"{OUT}/tables/fig03_leave_one_out.csv",index=False)
r=[f"{fn(m)} ({fn(l)} to {fn(h)})   {fp(p)}" for m,l,h,p in zip(f3.m,f3.lo,f3.hi,f3.t_p)]
fig,ax=forest(["Excluding "+t for t in f3.removed],f3.m,f3.lo,f3.hi,r,["Target removed (n=20)","Difference (95% CI)     p"],
  "Final − ProteinMPNN mean i_pTM, leave-one-target-out",None,xlim=(-0.02,0.2),sizes=[22]*len(f3),
  pooled=(pm.iloc[0],pm.iloc[1],pm.iloc[2],"All 21 targets",f"{fn(pm.iloc[0])} ({fn(pm.iloc[1])} to {fn(pm.iloc[2])})   0·015"))
save(fig,"fig03_forest_leave_one_out")

# ---- Fig 4: ROC + PR, experimental binder_final vs predicted-structure confidence scores
av["y"]=av.binder_final.astype(bool).astype(int)
cols=[c for c in av if c.startswith("ipsae_min_")]; names={c:c.replace("ipsae_min_","") for c in cols}
pal=[NAVY,RED,GREEN,TEAL,PURP,"#FDAF91","#AD002A",GREY,"#000000","#7A7A7A"]
fig,axs=plt.subplots(1,3,figsize=(7.2,2.9),gridspec_kw={"wspace":0.42})
res=[]
for c,col in zip(cols,pal):
    s=av[["y",c]].dropna(); au=roc_auc_score(s.y,s[c]); ap=average_precision_score(s.y,s[c])
    res.append((names[c],len(s),int(s.y.sum()),au,ap,s.y.mean()))
    fpr,tpr,_=roc_curve(s.y,s[c]); pr,rc,_=precision_recall_curve(s.y,s[c])
    axs[0].plot(fpr,tpr,color=col,lw=1.3,label=f"{names[c]} ({au:.3f})")
    axs[1].plot(rc,pr,color=col,lw=1.3)
R=pd.DataFrame(res,columns=["predictor","n","n_binders","ROC_AUC","AP","prevalence"]).sort_values("ROC_AUC",ascending=False); R.to_csv(f"{OUT}/tables/fig04_roc_pr_metrics.csv",index=False)
axs[0].plot([0,1],[0,1],color="black",lw=0.6,ls=(0,(4,3)))
axs[0].set(xlabel="1 − Specificity",ylabel="Sensitivity",xlim=(0,1),ylim=(0,1.01)); axs[0].set_aspect("equal")
prev=av.y.mean(); axs[1].axhline(prev,color="black",lw=0.6,ls=(0,(4,3)))
axs[1].set(xlabel="Recall",ylabel="Precision",xlim=(0,1),ylim=(0,1.02)); axs[1].set_aspect("equal")
axs[1].text(0.98,prev-0.02,f"Prevalence = {prev:.3f}",ha="right",va="top",fontsize=7)
h,l=axs[0].get_legend_handles_labels(); 
order=np.argsort([-float(x.split("(")[1][:-1]) for x in l]); h=[h[i] for i in order]; l=[l[i] for i in order]
axs[2].axis("off"); axs[2].legend(h,l,title="Structure predictor\n(ipSAE-min; ROC-AUC)",loc="center left",fontsize=7.5,title_fontsize=7.5,alignment="left",bbox_to_anchor=(-0.05,0.5))
for a,t in zip(axs[:2],"AB"): a.text(-0.3,1.06,t,transform=a.transAxes,fontsize=12,fontweight="bold")
save(fig,"fig04_roc_pr_binder_discrimination")

# ---- Fig 5: forest of ROC-AUC with bootstrap CI per predictor (ipSAE and sc_DockQ)
rows=[]
rng=np.random.default_rng(SEED)
for kind,pref in (("ipSAE-min","ipsae_min_"),("sc-DockQ","sc_dockq_")):
    for c in [c for c in av if c.startswith(pref)]:
        s=av[["y",c]].dropna(); yv,sv=s.y.values,s[c].values; a0=roc_auc_score(yv,sv); bs=[]
        for _ in range(2000):
            i=rng.integers(0,len(s),len(s))
            if yv[i].min()==yv[i].max(): continue
            bs.append(roc_auc_score(yv[i],sv[i]))
        rows.append((kind,c.replace(pref,""),len(s),a0,*np.percentile(bs,[2.5,97.5])))
f5=pd.DataFrame(rows,columns=["score","predictor","n","auc","lo","hi"]); f5.to_csv(f"{OUT}/tables/fig05_auc_bootstrap.csv",index=False)
f5=f5.sort_values(["score","auc"],ascending=[True,False]).reset_index(drop=True)
r=[f"{a:.3f} ({l:.3f} to {h:.3f})   n={n}".replace(".","·") for a,l,h,n in zip(f5.auc,f5.lo,f5.hi,f5.n)]
labs=[f"{s}: {p}" for s,p in zip(f5.score,f5.predictor)]
fig,ax=forest(labs,f5.auc,f5.lo,f5.hi,r,["Score: predictor","ROC-AUC (95% bootstrap CI)"],"ROC-AUC for experimental binder status",None,xlim=(0.4,1.0),sizes=[28]*len(f5))
ax.axvline(0.5,color=GREY,lw=0.8,ls=(0,(4,3))); ax.axvline(0,color="white",lw=0)
save(fig,"fig05_forest_auc_by_predictor")

# ---- Fig 6: per-target paired means (dumbbell) all four models
order=ts.sort_values("Final minus baseline")["Target"].tolist(); T=ts.set_index("Target")
fig,ax=plt.subplots(figsize=(3.6,4.6)); y=np.arange(len(order))
ax.hlines(y,T.loc[order,"ProteinMPNN mean"],T.loc[order,"Final mean"],color=GREY,lw=1)
ax.scatter(T.loc[order,"ProteinMPNN mean"],y,s=18,color=GREY,label="ProteinMPNN",zorder=3,linewidths=0)
ax.scatter(T.loc[order,"Final mean"],y,s=18,color=NAVY,label="Final",zorder=3,linewidths=0)
ax.set_yticks(y); ax.set_yticklabels(order); ax.set_xlabel("Mean i_pTM (5 designs per target)"); ax.legend(loc="lower right",fontsize=8)
save(fig,"fig06_dumbbell_target_means")
import json; json.dump({"font":FONT,"seed":SEED,"n_assay_rows":int(len(av)),"n_binders":int(av.y.sum())},open(f"{OUT}/tables/run_metadata.json","w"))
print(FONT); print(R.to_string())
