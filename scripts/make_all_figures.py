"""Lancet-style main (F1-F6) and supplementary (S1-S9) figures for OryqevaMPNN. Seed 20261105.
Font: Helvetica -> Arial -> Liberation Sans (metric-compatible)."""
import sys, re, json, itertools, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score
# NOTE: this script depends on load.py, which referenced a private analysis session file and was intentionally excluded. See README for data provenance.
SEED=20261105; rng=np.random.default_rng(SEED)
OUT="./results/figures"  # relative path, adjust as needed
F = None  # TODO: replace with path to your own cleaned preference-pair dataset (see data/train_pairs_clean.json)
FONT=next(f for f in ["Helvetica","Arial","Liberation Sans"] if f in {x.name for x in fm.fontManager.ttflist})
plt.rcParams.update({"font.family":FONT,"font.size":8,"axes.linewidth":0.7,"svg.fonttype":"none","axes.spines.top":False,
 "axes.spines.right":False,"xtick.major.width":0.7,"ytick.major.width":0.7,"xtick.major.size":3,"ytick.major.size":3,"legend.frameon":False,
 "axes.labelsize":8.5,"xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":7.5})
NAVY,RED,GREEN,TEAL,PURP,PEACH,CRIM,GREY="#00468B","#ED0000","#42B540","#0099B4","#925E9F","#FDAF91","#AD002A","#ADB6B6"
MODELS=["ProteinMPNN","SFT","Preference+auxSFT","Final"]; MC=[GREY,TEAL,PURP,NAVY]
SEQ=LinearSegmentedColormap.from_list("lan",["#F4F7FB","#7FA3CC",NAVY])
def fn(x,d=3): return f"{x:.{d}f}".replace("-","−").replace(".","·")
def fp(p): return "<0·001" if p<0.001 else fn(p)
def save(fig,name):
    fig.savefig(f"{OUT}/svg/{name}.svg",bbox_inches="tight"); fig.savefig(f"{OUT}/png/{name}.png",dpi=600,bbox_inches="tight",facecolor="white"); plt.close(fig); print("saved",name)
def lab(ax,L,dx=-0.16,dy=1.04): ax.text(dx,dy,L,transform=ax.transAxes,fontsize=11,fontweight="bold",va="bottom")
def tci(d):
    d=np.asarray(d,float); m=d.mean(); h=stats.t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)); return m,m-h,m+h
def wilson(k,n,z=1.96):
    p=k/n; c=(p+z*z/2/n)/(1+z*z/n); h=z*np.sqrt(p*(1-p)/n+z*z/4/n/n)/(1+z*z/n); return p,c-h,c+h
def forest_ax(ax,labels,est,lo,hi,xlabel,null=0.0,colors=None,txt=True,ms=4,xlim=None):
    est,lo,hi=[np.asarray(v,float) for v in (est,lo,hi)]; y=np.arange(len(est))[::-1]
    ax.axvline(null,color="black",lw=0.7)
    for i in range(len(est)):
        c=colors[i] if colors is not None else NAVY
        ax.plot([lo[i],hi[i]],[y[i]]*2,color="black",lw=0.9); ax.plot(est[i],y[i],"s",color=c,ms=ms,mew=0)
        if txt: ax.text(1.02,y[i],f"{fn(est[i])} ({fn(lo[i])} to {fn(hi[i])})",transform=ax.get_yaxis_transform(),va="center",fontsize=6.8)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlabel(xlabel); ax.spines["left"].set_visible(False); ax.tick_params(axis="y",length=0)
    if xlim: ax.set_xlim(*xlim)
# ------------------------------------------------------------------ data
ts=pd.read_excel(F,sheet_name="Target Summary").set_index("Target"); td=pd.read_excel(F,sheet_name="Target Diagnostics").set_index("target")
ps=pd.read_excel(F,sheet_name="Prediction Scores"); gen=pd.read_excel(F,sheet_name="Generated Sequences")
ct=pd.read_excel(F,sheet_name="Contrast Tests"); loo=pd.read_excel(F,sheet_name="Leave One Out"); res=pd.read_excel(F,sheet_name="Residue Composition")
thr=pd.read_excel(F,sheet_name="Threshold Crossing"); leak=pd.read_excel(F,sheet_name="Leakage Audit Revised").set_index("Target")
av=pd.read_excel(F,sheet_name="Assay Validation"); lo_=pd.read_excel(F,sheet_name="Label Outcomes")
sj=pd.read_excel(F,sheet_name="Statistics JSON").set_index("MetricKeyPath")["Value"]
clean=load.unpack("Clean Pairs 3680"); raw_n=40040
T=list(ts.index); d=(ts["Final minus baseline"]).loc[T]; base=ts["ProteinMPNN mean"]; fin=ts["Final mean"]
M={m:ts[c] for m,c in zip(MODELS,["ProteinMPNN mean","SFT mean","Preference+auxSFT mean","Final mean"])}
ghost=lambda ax: None
# ================================================================== F1
fig=plt.figure(figsize=(7.2,5.6)); gs=fig.add_gridspec(2,2,hspace=0.55,wspace=0.45)
ax=fig.add_subplot(gs[0,0]); lab(ax,"A")
vals=[raw_n,len(clean)]; ax.barh([1,0],vals,color=[GREY,NAVY],height=0.45)
for y,v in zip([1,0],vals): ax.text(v+600,y,f"{v:,}",va="center",fontsize=8)
ax.set_yticks([1,0]); ax.set_yticklabels(["Unfiltered\npreference pairs","Retained\npairs"]); ax.set_xlim(0,50000); ax.set_xlabel("Preference pairs (n)")
ax.text(0.98,0.5,f"{len(clean)/raw_n*100:.1f}% retained",transform=ax.transAxes,ha="right",fontsize=7.5)
ax=fig.add_subplot(gs[0,1]); lab(ax,"B")
sc=pd.crosstab(clean.Pair_target,clean.Preferred_source); order=clean.Pair_target.value_counts().index
bottom=np.zeros(len(order)); cols=[NAVY,RED,GREEN,TEAL,PURP,PEACH]
for c,col in zip(sc.columns,cols): v=sc.loc[order,c].values; ax.barh(np.arange(len(order))[::-1],v,left=bottom,color=col,height=0.65,label=c); bottom+=v
ax.set_yticks(np.arange(len(order))[::-1]); ax.set_yticklabels(order,fontsize=6.5); ax.set_xlabel("Retained pairs (n) by preferred-sequence source"); ax.legend(title="Source",loc="lower right",fontsize=7,title_fontsize=7)
ax=fig.add_subplot(gs[1,0]); lab(ax,"C"); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,6)
def box(ax,x,y,w,h,t,c=NAVY,fc="white",fs=7.2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="square,pad=0",ec=c,fc=fc,lw=0.9)); ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs)
def arr(ax,x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=7,lw=0.8,color="black"))
box(ax,3.4,4.5,3.2,1.1,"Pretrained ProteinMPNN\n(baseline)",GREY)
for x_,t_,c_ in [(0.0,"Protein-\nMPNN",GREY),(2.6,"SFT",TEAL),(5.2,"Pref. +\naux. SFT",PURP),(7.8,"Final\n(pref.+SFT)",NAVY)]:
    box(ax,x_,2.2,2.2,1.2,t_,c_,fs=6.2)
box(ax,1.2,0.1,7.6,1.0,f"Training inputs: {len(clean):,} cross-context preference pairs and\nlabelled binder / non-binder designs",GREY,fs=6.8)
ax.text(5,5.9,"Compared model conditions (schematic; see Methods)",ha="center",fontsize=7)
for x_ in (1.1,3.7,6.3,8.9): arr(ax,5,4.5,x_,3.4)
ax=fig.add_subplot(gs[1,1]); lab(ax,"D"); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,6)
box(ax,0.0,3.8,3.2,1.6,"21 benchmark\ntargets",fs=7); box(ax,3.7,3.8,2.5,1.6,"4 models",fs=7); box(ax,6.7,3.8,3.3,1.6,"5 sequences per\ntarget–model",fs=7)
box(ax,1.0,0.5,8.0,1.7,"420 designs scored by predicted i_pTM\n(statistical unit: complex, n = 21)",NAVY,fs=7); arr(ax,1.6,3.8,3.0,2.2); arr(ax,4.95,3.8,5.0,2.2); arr(ax,8.3,3.8,7.0,2.2)
save(fig,"F1_data_curation_and_workflow")
# ================================================================== F2
fig=plt.figure(figsize=(7.2,6.0)); gs=fig.add_gridspec(2,2,hspace=0.5,wspace=0.5)
ax=fig.add_subplot(gs[0,0]); lab(ax,"A")
for t in T: ax.plot([0,1],[base[t],fin[t]],color=(NAVY if d[t]>0 else RED),lw=0.8,alpha=0.7)
ax.plot([0,1],[base.mean(),fin.mean()],color="black",lw=2); ax.set_xticks([0,1]); ax.set_xticklabels(["ProteinMPNN","Final"]); ax.set_xlim(-0.2,1.2); ax.set_ylabel("Target-mean i_pTM")
ax=fig.add_subplot(gs[0,1]); lab(ax,"B"); ds=d.sort_values(); ax.barh(np.arange(21),ds.values,color=[NAVY if v>0 else RED for v in ds.values],height=0.6)
ax.set_yticks(np.arange(21)); ax.set_yticklabels(ds.index,fontsize=6.3); ax.axvline(0,color="black",lw=0.7); ax.set_xlabel("Final − ProteinMPNN (Δ i_pTM)")
ax=fig.add_subplot(gs[1,0]); lab(ax,"C"); m,l,h=tci(d); bl,bh=json.loads(sj["primary.CI95_bootstrap_percentile"])
forest_ax(ax,["t interval","Bootstrap percentile"],[m,m],[l,bl],[h,bh],"Mean Δ i_pTM (n = 21 complexes)",xlim=(-0.02,0.18),txt=False)
ax.set_xlim(-0.02,0.2); ax.set_ylim(-0.7,1.7); ax.set_title(f"Mean {fn(m)}, 95% CI {fn(l)} to {fn(h)}\nPaired t p = {fp(float(sj['primary.p_t']))}; Wilcoxon p = {fp(float(sj['primary.p_wilcoxon']))}",fontsize=7)
ax=fig.add_subplot(gs[1,1]); lab(ax,"D"); pos=d>0
ax.scatter(base[pos],d[pos],s=16,color=NAVY,label=f"Increase (n = {pos.sum()})",lw=0); ax.scatter(base[~pos],d[~pos],s=16,color=RED,label=f"Decrease (n = {(~pos).sum()})",lw=0)
rho,p=stats.spearmanr(base,d); ax.axhline(0,color="black",lw=0.6,ls=(0,(4,3))); ax.set_xlabel("ProteinMPNN target-mean i_pTM"); ax.set_ylabel("Δ i_pTM"); ax.legend(loc="upper right")
ax.text(0.03,0.05,f"Spearman ρ = {fn(rho)}, p = {fp(p)}",transform=ax.transAxes,fontsize=7)
save(fig,"F2_primary_effect"); print("n_pos",pos.sum(),"rho",rho,p)
# ================================================================== F3
fig=plt.figure(figsize=(7.2,6.0)); gs=fig.add_gridspec(2,2,hspace=0.55,wspace=0.7)
ax=fig.add_subplot(gs[0,0]); lab(ax,"A")
X=np.arange(4)
for t in T: ax.plot(X,[M[m][t] for m in MODELS],color=GREY,lw=0.6,alpha=0.7)
mm=[M[m].mean() for m in MODELS]; ax.plot(X,mm,"-",color="black",lw=1.6); [ax.plot(i,v,"o",color=c,ms=6,mew=0) for i,(v,c) in enumerate(zip(mm,MC))]
ax.set_xticks(X); ax.set_xticklabels(["Protein\nMPNN","SFT","Pref+aux\nSFT","Final"],fontsize=7); ax.set_ylabel("Target-mean i_pTM")
ax=fig.add_subplot(gs[0,1]); lab(ax,"B",dx=-0.5)
names={"SFT_vs_ProteinMPNN":"SFT − ProteinMPNN","Preference + auxiliary SFT_vs_ProteinMPNN":"Pref+aux − ProteinMPNN","Final (preference + SFT)_vs_ProteinMPNN":"Final − ProteinMPNN","Final (preference + SFT)_vs_SFT":"Final − SFT","Final (preference + SFT)_vs_Preference + auxiliary SFT":"Final − Pref+aux","SFT_vs_Preference + auxiliary SFT":"SFT − Pref+aux"}
forest_ax(ax,[names[c] for c in ct.Comparison],ct["mean difference"],ct["95% CI lower"],ct["95% CI upper"],"Paired Δ i_pTM",xlim=(-0.08,0.17),txt=True,ms=4.5)
ax=fig.add_subplot(gs[1,0]); lab(ax,"C"); dd=(ts["Final mean"]-ts["SFT mean"]).sort_values(); m,l,h=tci(dd)
ax.barh(np.arange(21),dd.values,color=[NAVY if v>0 else RED for v in dd.values],height=0.6); ax.set_yticks(np.arange(21)); ax.set_yticklabels(dd.index,fontsize=6.3); ax.axvline(0,color="black",lw=0.7)
ax.axvspan(l,h,color=GREY,alpha=0.3,lw=0); ax.axvline(m,color="black",lw=0.8,ls=(0,(4,3))); ax.set_xlabel("Final − SFT (Δ i_pTM)"); ax.text(0.98,0.04,f"Mean {fn(m)}\n(95% CI {fn(l)} to {fn(h)}\nshaded; crosses 0)",transform=ax.transAxes,fontsize=6.8,ha="right")
ax=fig.add_subplot(gs[1,1]); lab(ax,"D",dx=-0.5)
rk=[float(sj[k]) for k in [i for i in sj.index if i.startswith("friedman.mean_rank")]]
print("friedman ranks",rk, sj["friedman.Q"],sj["friedman.p"],sj["friedman.kendall_W"])
ax.barh(np.arange(4)[::-1],rk,color=MC,height=0.5); ax.set_yticks(np.arange(4)[::-1]); ax.set_yticklabels(["ProteinMPNN","SFT","Pref+aux SFT","Final"]); ax.set_xlabel("Mean rank across 21 targets (1 = lowest)")
ax.text(0.98,1.02,f"Friedman Q = {float(sj['friedman.Q']):.2f}\np = {fp(float(sj['friedman.p']))}, W = {float(sj['friedman.kendall_W']):.2f}",transform=ax.transAxes,ha="right",va="bottom",fontsize=7)
save(fig,"F3_four_model_comparison")
# ================================================================== F4
fig=plt.figure(figsize=(7.2,6.2)); gs=fig.add_gridspec(2,2,hspace=0.55,wspace=0.65)
ax=fig.add_subplot(gs[0,0]); lab(ax,"A"); idn=leak["IdentityPercent"].loc[T]; rho,p=stats.spearmanr(idn,d)
ax.scatter(idn,d,s=18,color=NAVY,lw=0); [ax.annotate(t,(idn[t],d[t]),fontsize=5.8,xytext=(2,2),textcoords="offset points") for t in T]
ax.axhline(0,color="black",lw=0.6,ls=(0,(4,3))); ax.set_xlabel("Max. archived target-sequence identity (%)"); ax.set_ylabel("Δ i_pTM (Final − ProteinMPNN)")
ax.text(0.97,0.05,f"Spearman ρ = {fn(rho)}, p = {fp(p)}\nn = {len(T)} complexes",transform=ax.transAxes,ha="right",fontsize=7); print("sim rho",rho,p)
ax=fig.add_subplot(gs[0,1]); lab(ax,"B",dx=-0.55); lo_s=loo.set_index("removed_target").loc[T]
rows=[tci(d.drop(t)) for t in T]; o=np.argsort([r[0] for r in rows]); 
forest_ax(ax,[f"− {T[i]}" for i in o],[rows[i][0] for i in o],[rows[i][1] for i in o],[rows[i][2] for i in o],"Mean Δ i_pTM, target removed",txt=False,ms=3,xlim=(-0.01,0.2)); ax.tick_params(axis="y",labelsize=5.8)
m,l,h=tci(d); ax.axvspan(l,h,color=GREY,alpha=0.3,lw=0)
ax=fig.add_subplot(gs[1,0]); lab(ax,"C"); ths=[100,55,52,50,48,46,45,44,42]; ths=sorted(set([100]+[int(np.floor(x)) for x in sorted(idn.values)[-8:]]),reverse=True); out=[]
for t0 in [100,52,50,48,46,45,44,42,40.5]:
    keep=idn[idn<t0].index if t0<100 else idn.index
    if len(keep)>=8: mm_=tci(d.loc[keep]); out.append((t0,len(keep),*mm_))
lbl=[("All" if o_[0]==100 else f"<{o_[0]:g}%")+f" (n={o_[1]})" for o_ in out]
forest_ax(ax,lbl,[o_[2] for o_ in out],[o_[3] for o_ in out],[o_[4] for o_ in out],"Mean Δ i_pTM after excluding identity ≥ threshold",txt=False,xlim=(-0.03,0.2))
ax=fig.add_subplot(gs[1,1]); lab(ax,"D",dx=-0.55)
srt=d.sort_values(ascending=False); sc_=[("All 21",tci(d)),("Without 5U8R",tci(d.drop("5U8R"))),("Without 3 largest",tci(d.drop(srt.index[:3])))]
forest_ax(ax,[a for a,_ in sc_],[b[0] for _,b in sc_],[b[1] for _,b in sc_],[b[2] for _,b in sc_],"Mean Δ i_pTM",xlim=(-0.02,0.2),txt=True,ms=5)
print("top3",list(srt.index[:3]),"share",sj.get("three_largest_share_of_net_difference"),"JSON removed3 mean",sj.get("removed_three_largest_positive.mean_difference"),sc_[2][1][0])
save(fig,"F4_generalization_robustness")
# ================================================================== diversity (Hamming) from sequences
def mism(seqs): return np.mean([sum(a!=b for a,b in zip(x,y))/len(x) for x,y in itertools.combinations(seqs,2)])
gen["Model"]=gen["Model"].astype(str); mod_map=dict(zip(sorted(gen.Model.unique()),sorted(gen.Model.unique()))); print(gen.Model.unique())
div=gen.groupby(["Target","Model"])["Sequence"].apply(lambda s:mism(list(s))).unstack(); print(div.mean())
# ================================================================== F5
fig=plt.figure(figsize=(7.2,7.2)); gs=fig.add_gridspec(2,2,hspace=0.45,wspace=0.5)
ax=fig.add_subplot(gs[0,0]); lab(ax,"A"); r=res.sort_values("mean_target_ppt"); sig=r.t_p_bh_20<0.05
ax.barh(np.arange(20),r.mean_target_ppt,color=[NAVY if s else GREY for s in sig],height=0.6); ax.set_yticks(np.arange(20)); ax.set_yticklabels(r.residue); ax.axvline(0,color="black",lw=0.7)
ax.set_xlabel("Mean change (Final − ProteinMPNN), percentage points"); ax.text(0.97,0.55,"Navy: BH-adjusted\np < 0·05",transform=ax.transAxes,ha="right",fontsize=7); ax.tick_params(axis="y",labelsize=6.5)
ax=fig.add_subplot(gs[0,1]); lab(ax,"B"); cd,hd=td.charged_fraction_delta.loc[T],td.hydrophobic_fraction_delta.loc[T]
for i,(v,c,n) in enumerate([(cd,NAVY,"Charged"),(hd,RED,"Hydrophobic")]):
    ax.scatter(np.full(21,i)+rng.uniform(-0.12,0.12,21),v,s=14,color=c,lw=0,alpha=0.8); m,l,h=tci(v); ax.plot([i,i],[l,h],color="black",lw=1.2); ax.plot(i,m,"D",color="black",ms=5)
ax.axhline(0,color="black",lw=0.6,ls=(0,(4,3))); ax.set_xticks([0,1]); ax.set_xticklabels(["Charged","Hydrophobic"]); ax.set_ylabel("Fraction change per target")
ax=fig.add_subplot(gs[1,0]); lab(ax,"C")
for j,mname in enumerate(sorted(div.columns, key=lambda x:["ProteinMPNN","SFT","Preference","Final"].index(next(k for k in ["ProteinMPNN","SFT","Preference","Final"] if x.startswith(k))) if any(x.startswith(k) for k in ["ProteinMPNN","SFT","Preference","Final"]) else 9)):
    v=div[mname]; ax.scatter(np.full(21,j)+rng.uniform(-0.12,0.12,21),v,s=12,color=MC[j],lw=0,alpha=0.8); ax.plot([j-0.25,j+0.25],[v.mean()]*2,color="black",lw=1.5)
ax.set_xticks(range(4)); ax.set_xticklabels(["Protein\nMPNN","SFT","Pref+aux\nSFT","Final"],fontsize=7); ax.set_ylabel("Within-target sequence diversity\n(normalised pairwise mismatch)")
dv=div.iloc[:,:]; 
ax=fig.add_subplot(gs[1,1]); lab(ax,"D"); rho,p=stats.spearmanr(cd,d); ax.scatter(cd,d,s=18,color=NAVY,lw=0); ax.axhline(0,color="black",lw=0.6,ls=(0,(4,3))); ax.set_xlabel("Charged-fraction change"); ax.set_ylabel("Δ i_pTM")
ax.text(0.03,0.05,f"Spearman ρ = {fn(rho)}, p = {fp(p)}",transform=ax.transAxes,fontsize=7); print("charged rho",rho,p)
save(fig,"F5_composition_diversity")
# ================================================================== F6
av["y"]=av.binder_final.astype(bool).astype(int); lo_["in608"]=lo_.full_name.isin(av.full_name); print("label overlap",lo_.in608.sum())
fig=plt.figure(figsize=(7.2,6.2)); gs=fig.add_gridspec(2,2,hspace=0.55,wspace=0.5)
ax=fig.add_subplot(gs[0,0]); lab(ax,"A")
npos=(lo_.label=="positive").sum(); nneg=len(lo_)-npos; sb=av.y.sum(); sn=len(av)-sb
ax.barh([1,0],[npos,sb],color=NAVY,height=0.45,label="Binder / positive"); ax.barh([1,0],[nneg,sn],left=[npos,sb],color=GREY,height=0.45,label="Non-binder / negative")
ax.set_yticks([1,0]); ax.set_yticklabels([f"Label dataset\n(n = {len(lo_):,})",f"Assay-validation\nsubset (n = {len(av)})"]); ax.set_xlabel("Designs (n)"); ax.legend(loc="center right",fontsize=6.5,bbox_to_anchor=(1.0,0.5))
ax=fig.add_subplot(gs[0,1]); lab(ax,"B"); s=av[av.adaptyv_binding.isin(["binder","non_binder"])&av.twist_binding.isin(["binder","non_binder"])]
print(av.twist_binding.value_counts(dropna=False).to_dict())
tb=s.twist_binding; cm=pd.crosstab(s.adaptyv_binding,tb).reindex(index=["binder","non_binder"],columns=["binder","non_binder"]).fillna(0).astype(int)
ax.imshow(cm.values,cmap=SEQ); [ax.text(j,i,str(cm.values[i,j]),ha="center",va="center",fontsize=9,color=("white" if cm.values[i,j]>cm.values.max()/2 else "black")) for i in range(2) for j in range(2)]
ax.set_xticks([0,1]); ax.set_xticklabels(["Binder","Non-binder"]); ax.set_yticks([0,1]); ax.set_yticklabels(["Binder","Non-binder"]); ax.set_xlabel("Twist label"); ax.set_ylabel("Adaptyv label"); ax.spines[:].set_visible(False)
ax.set_title(f"Designs tested by both vendors (n = {cm.values.sum()})",fontsize=7.5)
def cluster_auc(df,col,B=2000):
    s=df[["y",col,"target"]].dropna(); a0=roc_auc_score(s.y,s[col]); groups=[g for _,g in s.groupby("target")]; bs=[]
    for _ in range(B):
        idx=rng.integers(0,len(groups),len(groups)); b=pd.concat([groups[i] for i in idx])
        if b.y.nunique()==2: bs.append(roc_auc_score(b.y,b[col]))
    return a0,*np.percentile(bs,[2.5,97.5]),len(s)
cols=[c for c in av if c.startswith(("ipsae_min_","sc_dockq_"))]; AU={c:cluster_auc(av,c) for c in cols}
pd.DataFrame(AU,index=["auc","lo","hi","n"]).T.to_csv(f"{OUT}/tables/F6C_cluster_bootstrap_auc.csv")
ax=fig.add_subplot(gs[1,0]); lab(ax,"C"); top=sorted([c for c in cols if c.startswith("ipsae")],key=lambda c:-AU[c][0])[:4]
for c,col in zip(top,[NAVY,RED,GREEN,PURP]):
    s=av[["y",c]].dropna(); f,t,_=roc_curve(s.y,s[c]); a=AU[c]; ax.plot(f,t,color=col,lw=1.4,label=f"{c.replace('ipsae_min_','')} {a[0]:.3f} ({a[1]:.3f}–{a[2]:.3f})")
ax.plot([0,1],[0,1],color="black",lw=0.6,ls=(0,(4,3))); ax.set_xlabel("1 − Specificity"); ax.set_ylabel("Sensitivity"); ax.set_aspect("equal"); ax.legend(title="ipSAE-min: AUC (95% CI)",loc="upper center",bbox_to_anchor=(0.5,-0.22),fontsize=6.3,title_fontsize=6.5)
ax=fig.add_subplot(gs[1,1]); lab(ax,"D"); g=lo_.groupby("in608").label.agg(k=lambda x:(x=="positive").sum(),n="size")
xs=[]; 
for i,(flag,nm) in enumerate([(True,"In assay\nsubset"),(False,"Remaining\npopulation")]):
    k,n=g.loc[flag,"k"],g.loc[flag,"n"]; p_,l,h=wilson(k,n); ax.bar(i,p_,color=[NAVY,GREY][i],width=0.45); ax.plot([i,i],[l,h],color="black",lw=1); ax.text(i,h+0.02,f"{k}/{n}",ha="center",fontsize=7)
ax.set_xticks([0,1]); ax.set_xticklabels(["In assay\nsubset","Remaining\npopulation"]); ax.set_ylabel("Positive-label rate (Wilson 95% CI)"); ax.set_ylim(0,1)
_,pf=stats.fisher_exact([[g.loc[True,"k"],g.loc[True,"n"]-g.loc[True,"k"]],[g.loc[False,"k"],g.loc[False,"n"]-g.loc[False,"k"]]]); ax.text(0.5,0.95,("Fisher exact p < 0·001" if pf<0.001 else f"Fisher exact p = {fn(pf)}"),transform=ax.transAxes,ha="center",fontsize=7)
save(fig,"F6_experimental_provenance_roc")
# ================================================================== S1
cl=clean.copy(); cl["lp"]=cl.Preferred_sequence.str.len(); cl["ld"]=cl.Dispreferred_sequence.str.len()
camp=lambda s:"single" if "single_target" in s else "multi"; dm=lambda s:"opus" if s.startswith("opus") else "mythos"
cl["same_len"]=cl.lp==cl.ld; cl["same_target_set"]=cl.Preferred_name.map(camp)==cl.Dispreferred_name.map(camp); cl["same_model"]=cl.Preferred_name.map(dm)==cl.Dispreferred_name.map(dm)
chg=lambda s:sum(c in"DEKR" for c in s)/len(s); hyd=lambda s:sum(c in"AVILMFWC" for c in s)/len(s)
fig=plt.figure(figsize=(7.2,5.8)); gs=fig.add_gridspec(2,2,hspace=0.5,wspace=0.45)
ax=fig.add_subplot(gs[0,0]); lab(ax,"A"); ax.hist(cl.lp-cl.ld,bins=60,color=NAVY); ax.set_xlabel("Binder length, preferred − dispreferred (aa)"); ax.set_ylabel("Pairs (n)"); ax.text(0.97,0.95,f"Equal length: {cl.same_len.mean()*100:.1f}%\nn = {len(cl):,} pairs",transform=ax.transAxes,ha="right",va="top",fontsize=7)
ax=fig.add_subplot(gs[0,1]); lab(ax,"B"); fr=[cl.same_len.mean(),cl.same_model.mean(),cl.same_target_set.mean()]; ax.barh([2,1,0],fr,color=NAVY,height=0.45)
ax.set_yticks([2,1,0]); ax.set_yticklabels(["Same binder\nlength","Same design\nmodel","Same campaign\n(single/multi)"]); ax.set_xlim(0,1); ax.set_xlabel("Fraction of retained pairs"); [ax.text(v+0.02,y,f"{v:.2f}",va="center",fontsize=7) for v,y in zip(fr,[2,1,0])]
for k,(nm,fx,L) in enumerate([("Charged fraction (D, E, K, R)",chg,"C"),("Hydrophobic fraction (A, V, I, L, M, F, W, C)",hyd,"D")]):
    ax=fig.add_subplot(gs[1,k]); lab(ax,L); a=cl.Preferred_sequence.map(fx); b=cl.Dispreferred_sequence.map(fx); ax.scatter(b,a,s=3,color=NAVY,alpha=0.25,lw=0,rasterized=True)
    lim=[0,max(a.max(),b.max())*1.05]; ax.plot(lim,lim,color="black",lw=0.6,ls=(0,(4,3))); ax.set_xlabel(f"Dispreferred: {nm}",fontsize=7); ax.set_ylabel("Preferred"); ax.set_aspect("equal")
    ax.text(0.04,0.93,f"Mean Δ = {fn(a.mean()-b.mean())}",transform=ax.transAxes,fontsize=7)
save(fig,"S1_preference_pair_inconsistency")
# S2
fig=plt.figure(figsize=(7.2,5.6)); gs=fig.add_gridspec(1,2,width_ratios=[1.6,1],wspace=0.9)
ax=fig.add_subplot(gs[0]); lab(ax,"A"); hm=pd.crosstab(clean.Pair_target,clean.Preferred_source); hm=hm.loc[clean.Pair_target.value_counts().index]
im=ax.imshow(hm.values,cmap=SEQ,aspect="auto"); ax.set_xticks(range(hm.shape[1])); ax.set_xticklabels(hm.columns,rotation=30,ha="right"); ax.set_yticks(range(hm.shape[0])); ax.set_yticklabels(hm.index,fontsize=6.8)
[ax.text(j,i,str(v),ha="center",va="center",fontsize=6,color=("white" if v>hm.values.max()/2 else "black")) for i,r_ in enumerate(hm.values) for j,v in enumerate(r_) if v>0]
ax.spines[:].set_visible(False); cb=fig.colorbar(im,ax=ax,fraction=0.04,pad=0.02); cb.set_label("Pairs (n)"); cb.outline.set_visible(False)
ax=fig.add_subplot(gs[1]); lab(ax,"B"); pc=clean.Pair_target.value_counts(); ax.barh(np.arange(len(pc))[::-1],pc.values,color=NAVY,height=0.6); ax.set_yticks(np.arange(len(pc))[::-1]); ax.set_yticklabels(pc.index,fontsize=6.8); ax.set_xlabel("Retained pairs (n)")
ax.text(0.97,0.05,f"Max/min = {pc.max()/pc.min():.1f}×",transform=ax.transAxes,ha="right",fontsize=7)
save(fig,"S2_dataset_imbalance")
# S3
nperm=2**21; B=20000; bs=np.array([d.values[rng.integers(0,21,21)].mean() for _ in range(B)])
signs=((np.arange(nperm)[:,None]>>np.arange(21))&1)*2-1; perm=(signs*np.abs(d.values)[None,:]).mean(1); obs=d.mean(); pexact=(np.abs(perm)>=abs(obs)-1e-12).mean()
fig,axs=plt.subplots(1,3,figsize=(7.2,2.6),gridspec_kw={"wspace":0.45})
axs[0].hist(bs,bins=50,color=NAVY); lo2,hi2=np.percentile(bs,[2.5,97.5]); [axs[0].axvline(v,color="black",lw=0.7,ls=(0,(4,3))) for v in (lo2,hi2)]; axs[0].set_xlabel("Bootstrap mean Δ i_pTM"); axs[0].set_ylabel("Resamples"); axs[0].set_title(f"95% CI {fn(lo2)} to {fn(hi2)}; B = {B:,}",fontsize=6.5)
axs[1].hist(perm,bins=60,color=GREY); axs[1].axvline(obs,color=RED,lw=1.2); axs[1].axvline(-obs,color=RED,lw=1.2,ls=(0,(4,3))); axs[1].set_xlabel("Sign-flip null mean Δ i_pTM"); axs[1].set_ylabel("Permutations"); axs[1].set_title(f"Exact two-sided p = {fn(pexact)} (2²¹ sign flips)",fontsize=6.5)
axs[2].hist(d,bins=np.linspace(-0.15,0.4,12),color=NAVY); axs[2].axvline(0,color="black",lw=0.6); axs[2].set_xlabel("Per-target Δ i_pTM"); axs[2].set_ylabel("Targets (n)")
for a,L in zip(axs,"ABC"): lab(a,L,dx=-0.3)
save(fig,"S3_robustness_bootstrap_permutation"); print("perm p",pexact,"json",sj.get("primary.p_perm_exact"))
# S4
fig=plt.figure(figsize=(7.2,5.8)); gs=fig.add_gridspec(1,2,width_ratios=[1.2,1],wspace=0.45)
ax=fig.add_subplot(gs[0]); lab(ax,"A"); order=ts["Final minus baseline"].sort_values(ascending=False).index; mat=np.array([[M[m][t] for m in MODELS] for t in order])
im=ax.imshow(mat,cmap=SEQ,aspect="auto",vmin=0,vmax=1); ax.set_xticks(range(4)); ax.set_xticklabels(["Protein\nMPNN","SFT","Pref+aux\nSFT","Final"],fontsize=6.8); ax.set_yticks(range(21)); ax.set_yticklabels(order,fontsize=6.5)
[ax.text(j,i,f"{mat[i,j]:.2f}",ha="center",va="center",fontsize=5.8,color=("white" if mat[i,j]>0.55 else "black")) for i in range(21) for j in range(4)]
ax.spines[:].set_visible(False); cb=fig.colorbar(im,ax=ax,fraction=0.05,pad=0.02); cb.set_label("Target-mean i_pTM"); cb.outline.set_visible(False)
ax=fig.add_subplot(gs[1]); lab(ax,"B"); rk=pd.DataFrame(mat,index=order,columns=MODELS).rank(axis=1)
for t in order: ax.plot(range(4),rk.loc[t],color=GREY,lw=0.6,alpha=0.7)
ax.plot(range(4),rk.mean(),"-o",color=NAVY,lw=1.8,ms=5,mew=0); ax.set_xticks(range(4)); ax.set_xticklabels(["Protein\nMPNN","SFT","Pref+aux\nSFT","Final"],fontsize=6.8); ax.set_yticks([1,2,3,4]); ax.set_ylabel("Within-target rank (1 = lowest)")
save(fig,"S4_model_consistency")
# S5
fig,axs=plt.subplots(1,3,figsize=(7.2,3.6),gridspec_kw={"wspace":0.7})
for a,(b,f,nm,L) in zip(axs,[("base_best_of_five","final_best_of_five","Best-of-five i_pTM","A"),("base_worst_of_five","final_worst_of_five","Worst-of-five i_pTM","B"),("within_sd_baseline","within_sd_final","Within-target SD of i_pTM","C")]):
    dif=(td[f]-td[b]).loc[T]; m,l,h=tci(dif); o=dif.sort_values().index
    a.barh(np.arange(21),dif.loc[o].values,color=[NAVY if v>0 else RED for v in dif.loc[o].values],height=0.6); a.set_yticks(np.arange(21)); a.set_yticklabels(o,fontsize=5.8); a.axvline(0,color="black",lw=0.7)
    a.axvspan(l,h,color=GREY,alpha=0.3,lw=0); a.set_xlabel(f"Δ {nm}\n(Final − ProteinMPNN)",fontsize=7.3); a.set_title(f"mean {fn(m)}\n({fn(l)} to {fn(h)})",fontsize=6.8); lab(a,L,dx=-0.45,dy=1.15)
save(fig,"S5_best_worst_variability")
# S6
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),gridspec_kw={"wspace":0.35,"width_ratios":[1.3,1]})
x=np.arange(len(thr)); c=[("both_yes",NAVY,"Above in both"),("up_crossings",GREEN,"Crossed up"),("down_crossings",RED,"Crossed down"),("both_no",GREY,"Below in both")]; bt=np.zeros(len(thr))
for k,col,nm in c: axs[0].bar(x,thr[k],bottom=bt,color=col,width=0.55,label=nm); bt+=thr[k].values
axs[0].set_xticks(x); axs[0].set_xticklabels([f"≥ {t}" for t in thr.threshold]); axs[0].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True)); axs[0].set_xlabel("Target-mean i_pTM threshold"); axs[0].set_ylabel("Targets (n = 21)"); axs[0].legend(ncol=2,fontsize=6.5,loc="upper center",bbox_to_anchor=(0.5,1.28)); lab(axs[0],"A",dx=-0.2,dy=1.2)
axs[1].plot(x,thr.baseline_at_or_above,"-o",color=GREY,ms=5,label="ProteinMPNN"); axs[1].plot(x,thr.final_at_or_above,"-o",color=NAVY,ms=5,label="Final")
axs[1].set_xticks(x); axs[1].set_xticklabels([f"≥ {t}" for t in thr.threshold]); axs[1].set_ylabel("Targets at or above threshold"); axs[1].legend(); lab(axs[1],"B",dx=-0.25,dy=1.2)
axs[1].text(0.5,-0.32,"McNemar exact p (exploratory): "+", ".join(fp(p) for p in thr.mcnemar_exact_p_exploratory),transform=axs[1].transAxes,ha="center",fontsize=6.3)
save(fig,"S6_threshold_sensitivity")
# S7
AA="ACDEFGHIKLMNPQRSTVWY"; base_g=gen[gen.Model=="ProteinMPNN"]; fin_g=gen[gen.Model.str.startswith("Final")]
def fr_(g): 
    o={}
    for t,s in g.groupby("Target")["Sequence"]:
        j="".join(s); o[t]=[j.count(a)/len(j)*100 for a in AA]
    return pd.DataFrame(o,index=list(AA)).T
dfa=fr_(fin_g)-fr_(base_g); rows=[]
for a in AA:
    v=dfa[a]; m,l,h=tci(v); rows.append((a,m,l,h,stats.ttest_1samp(v,0).pvalue))
r7=pd.DataFrame(rows,columns=["aa","m","lo","hi","p"])
def bh(p):
    p=np.asarray(p); o=np.argsort(p); r=p[o]*len(p)/(np.arange(len(p))+1); r=np.minimum.accumulate(r[::-1])[::-1]; q=np.empty_like(p); q[o]=np.minimum(r,1); return q
r7["q"]=bh(r7.p); chk=res.set_index("residue").loc[r7.aa]; print("S7 mean match",np.abs(chk.mean_target_ppt.values-r7.m.values).max(),"q match",np.abs(chk.t_p_bh_20.values-r7.q.values).max())
r7=r7.sort_values("m"); fig,ax=plt.subplots(figsize=(4.6,5.6))
forest_ax(ax,r7.aa,r7.m,r7.lo,r7.hi,"Change in residue composition, Final − ProteinMPNN (percentage points)",colors=[RED if q<0.05 else GREY for q in r7.q],txt=False,ms=4.5); 
for i,(q) in enumerate(r7.q.values[::-1]): ax.text(1.02,i,("q < 0·001" if q<0.001 else f"q = {fn(q)}"),transform=ax.get_yaxis_transform(),va="center",fontsize=6.8)
ax.text(0.02,0.02,"Red: BH-FDR q < 0·05; n = 21 targets",transform=ax.transAxes,fontsize=7); save(fig,"S7_amino_acid_forest")
# S8
ev=pd.read_excel(F,sheet_name="Training Log Events"); dp=[];sf=[];best=[]
for fnm,gdf in ev.groupby("LogFilename"):
    for t in gdf.EventText:
        m_=re.search(r"Epoch (\d+): total_loss=([\d\.eE\-]+)",str(t)) or re.search(r"Epoch (\d+): DPO loss = ([\d\.eE\-]+)",str(t))
        if m_ and not str(t).startswith("SFT"): dp.append((fnm,int(m_.group(1)),float(m_.group(2))))
        m_=re.search(r"SFT Epoch (\d+): avg_loss=([\d\.eE\-]+)",str(t))
        if m_: sf.append((fnm,int(m_.group(1)),float(m_.group(2))))
        m_=re.search(r"Best: (round|step) (\d+) \(heldout_score=([\-\d\.eE]+|inf|-inf)\)",str(t))
        if m_: best.append((fnm,m_.group(1),int(m_.group(2)),float(m_.group(3))))
dp=pd.DataFrame(dp,columns=["log","ep","loss"]); sf=pd.DataFrame(sf,columns=["log","ep","loss"]); bt_=pd.DataFrame(best,columns=["log","unit","n","score"]); print(len(dp),len(sf),len(bt_))
fig,axs=plt.subplots(1,3,figsize=(7.2,2.9),gridspec_kw={"wspace":0.5})
for a,(df,nm,col,L) in zip(axs[:2],[(dp,"DPO-stage loss",NAVY,"A"),(sf,"SFT-stage average loss",TEAL,"B")]):
    for _,g in df.groupby("log"): a.plot(g.ep,g.loss,color=col,lw=0.7,alpha=0.6)
    a.set_xlabel("Epoch"); a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True)); a.set_ylabel(nm); lab(a,L,dx=-0.3); a.text(0.97,0.95,f"{df.log.nunique()} log"+("s" if df.log.nunique()>1 else ""),transform=a.transAxes,ha="right",va="top",fontsize=7)
f_=bt_[np.isfinite(bt_.score)]; axs[2].scatter(f_.n,f_.score,s=12,color=PURP,lw=0); axs[2].set_xlabel("Selected checkpoint (round/step index)"); axs[2].set_ylabel("Held-out score"); lab(axs[2],"C",dx=-0.3); axs[2].text(0.97,0.95,f"{len(f_)} finite scores\n({len(bt_)-len(f_)} non-finite omitted)",transform=axs[2].transAxes,ha="right",va="top",fontsize=6.5)
save(fig,"S8_training_history")
# S9
g9=av.groupby("target").y.agg(k="sum",n="size").assign(**{}).sort_values("k"); g9["r"]=g9.k/g9.n
fig=plt.figure(figsize=(7.2,5.6)); gs=fig.add_gridspec(2,2,hspace=0.55,wspace=0.6)
ax=fig.add_subplot(gs[0,0]); lab(ax,"A"); W=[wilson(k,n) for k,n in zip(g9.k,g9.n)]
forest_ax(ax,[f"{t} ({n})" for t,n in zip(g9.index,g9.n)],[w[0] for w in W],[w[1] for w in W],[w[2] for w in W],"Binder rate (Wilson 95% CI)",null=av.y.mean(),txt=False,xlim=(0,1))
ax=fig.add_subplot(gs[0,1]); lab(ax,"B"); s=av[av.adaptyv_binding.isin(["binder","non_binder"])&av.twist_binding.isin(["binder","non_binder"])].copy(); s["tw"]=s.twist_binding
dis=s.groupby("target").apply(lambda x:pd.Series({"disc":(x.adaptyv_binding!=x.tw).sum(),"n":len(x)})); dis=dis[dis.n>0].sort_values("disc"); ax.barh(np.arange(len(dis)),dis.disc,color=RED,height=0.6)
ax.set_yticks(np.arange(len(dis))); ax.set_yticklabels([f"{t} ({int(n)})" for t,n in zip(dis.index,dis.n)],fontsize=6.5); ax.set_xlabel("Vendor-discordant designs (n)")
ax=fig.add_subplot(gs[1,:]); lab(ax,"C",dx=-0.07); A=pd.DataFrame(AU,index=["auc","lo","hi","n"]).T.sort_values("auc"); A["lbl"]=[("ipSAE-min: " if c.startswith("ipsae") else "sc-DockQ: ")+c.split("_",2)[-1] if c.startswith("ipsae") else "sc-DockQ: "+c.replace("sc_dockq_","") for c in A.index]
ax.errorbar(np.arange(len(A)),A.auc,yerr=[A.auc-A.lo,A.hi-A.auc],fmt="s",color=NAVY,ms=4,ecolor="black",elinewidth=0.9,capsize=0); ax.axhline(0.5,color=GREY,lw=0.8,ls=(0,(4,3)))
ax.set_xticks(np.arange(len(A))); ax.set_xticklabels(A.lbl,rotation=60,ha="right",fontsize=6.5); ax.set_ylabel("ROC-AUC (target-cluster\nbootstrap 95% CI)")
save(fig,"S9_label_classifier_uncertainty")
