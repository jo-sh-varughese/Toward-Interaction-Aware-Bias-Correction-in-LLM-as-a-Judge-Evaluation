import json, re, warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from sklearn.metrics import cohen_kappa_score
from pathlib import Path
from itertools import combinations

warnings.filterwarnings("ignore")
FIGS = Path("/home/claude/v4/figures")
DATA = Path("/home/claude/v4/data")
FIGS.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family":"DejaVu Serif","font.size":11,
    "axes.titlesize":12,"axes.labelsize":11,"xtick.labelsize":10,
    "ytick.labelsize":10,"legend.fontsize":10,"figure.dpi":150,
    "savefig.dpi":300,"savefig.bbox":"tight",
    "axes.spines.top":False,"axes.spines.right":False})
W=dict(blue="#0072B2",orange="#E69F00",green="#009E73",red="#D55E00",
       purple="#CC79A7",sky="#56B4E9",black="#000000",grey="#999999")
sigmoid=lambda x:1/(1+np.exp(-x))
logit_=lambda p:np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9)))
FULL_F="Y_bin ~ P + V + S + PxV + PxS + VxS + PxVxS"
def drop_term(f,t):
    f=re.sub(r'\s*\+\s*\b'+re.escape(t)+r'\b','',f)
    return re.sub(r'\b'+re.escape(t)+r'\b\s*\+\s*','',f).strip()
def safe_bic(bij,bi,bj,eps=0.10):
    if abs(bi)<eps or abs(bj)<eps: return np.nan
    return bij/np.sqrt(abs(bi)*abs(bj))

# DATA
JUDGES={"GPT-4o":dict(aP=0.55,aV=0.48,aS=0.31,u0=0.08,alpha=0.10),
        "Gemini 1.5":dict(aP=0.48,aV=0.55,aS=0.24,u0=-0.06,alpha=0.08),
        "Claude 3.5":dict(aP=0.42,aV=0.38,aS=0.38,u0=0.04,alpha=-0.02),
        "Llama-3-70B":dict(aP=0.62,aV=0.65,aS=0.19,u0=-0.10,alpha=0.12),
        "Mistral-L2":dict(aP=0.51,aV=0.42,aS=0.27,u0=0.03,alpha=0.05)}
DGP=dict(bPV=0.55,bVS=-0.30,bPS=0.05,bPVS=0.08)
TASKS=["Summarization","Code Review","Math Reasoning","Open-ended QA"]
rng=np.random.default_rng(2024)
item_re=rng.normal(0,0.22,200); task_re=rng.normal(0,0.10,4)
rows=[]
for k in range(200):
    ti=k%4
    for ji,(jn,jc) in enumerate(JUDGES.items()):
        for P in [0,1]:
            for V in [0,1]:
                for S in [0,1]:
                    eta=(jc["alpha"]+jc["aP"]*P+jc["aV"]*V+jc["aS"]*S
                         +DGP["bPV"]*P*V+DGP["bVS"]*V*S+DGP["bPS"]*P*S
                         +DGP["bPVS"]*P*V*S+jc["u0"]+item_re[k]+task_re[ti])
                    prob=sigmoid(eta); r=rng.uniform()
                    v="A" if r<prob-0.04 else("B" if r>prob+0.04 else "tie")
                    rows.append(dict(pair_id=k,task=TASKS[ti],judge=jn,judge_id=ji,
                                    P=P,V=V,S=S,PxV=P*V,PxS=P*S,VxS=V*S,PxVxS=P*V*S,
                                    verdict=v,Y_star=(1. if v=="A" else .5 if v=="tie" else 0.),
                                    Y_bin=(1 if v=="A" else 0)))
df=pd.DataFrame(rows); df_nb=df[df.verdict!="tie"].copy()
jnames=list(JUDGES.keys())

def fit(d,f=FULL_F):
    return smf.logit(f,data=d).fit(cov_type="cluster",cov_kwds={"groups":d["judge_id"]},disp=False,maxiter=500)
def get_bic(m):
    p=m.params; bP,bV,bS=p.get("P",0),p.get("V",0),p.get("S",0)
    bPV,bPS,bVS=p.get("PxV",0),p.get("PxS",0),p.get("VxS",0)
    pv=safe_bic(bPV,bP,bV); ps=safe_bic(bPS,bP,bS); vs=safe_bic(bVS,bV,bS)
    vals=[x for x in [pv,ps,vs] if not np.isnan(x)]
    return dict(bP=bP,bV=bV,bS=bS,bPV=bPV,bPS=bPS,bVS=bVS,BIC_PV=pv,BIC_PS=ps,BIC_VS=vs,
                BIC_global=float(np.sqrt(sum(x**2 for x in vals))) if vals else np.nan)
def lrt(d,drop):
    nf=drop_term(FULL_F,drop)
    mf=smf.logit(FULL_F,data=d).fit(disp=False,maxiter=500)
    mn=smf.logit(nf,data=d).fit(disp=False,maxiter=500)
    c=max(0.,2*(mf.llf-mn.llf)); dfd=max(1,int(mn.df_resid-mf.df_resid))
    return c,1.-stats.chi2.cdf(c,dfd)
def dwr(d,fac,B=2000,seed=0):
    rng2=np.random.default_rng(seed)
    hi=d[d[fac]==1]["Y_star"].values; lo=d[d[fac]==0]["Y_star"].values
    obs=hi.mean()-lo.mean()
    boot=[rng2.choice(hi,len(hi),replace=True).mean()-rng2.choice(lo,len(lo),replace=True).mean() for _ in range(B)]
    ci=np.percentile(boot,[2.5,97.5])
    return dict(d=obs,lo=ci[0],hi=ci[1],sig=not(ci[0]<=0<=ci[1]))

print("Fitting models...")
m_pool=fit(df_nb); bpool=get_bic(m_pool)
lrt_pv=lrt(df_nb,"PxV"); lrt_ps=lrt(df_nb,"PxS"); lrt_vs=lrt(df_nb,"VxS")
jbic={jn:get_bic(smf.logit(FULL_F,data=df_nb[df_nb.judge==jn]).fit(disp=False,maxiter=500))
      for jn in jnames}
tbic={tk:get_bic(smf.logit(FULL_F,data=df_nb[df_nb.task==tk]).fit(disp=False,maxiter=500))
      for tk in TASKS}
pdwr={f:dwr(df,f) for f in ["P","V","S"]}
jdwr={jn:{f:dwr(df[df.judge==jn],f) for f in ["P","V","S"]} for jn in jnames}

# BIC bootstrap CIs
print("Bootstrap CIs...")
items=df_nb["pair_id"].unique()
bpv_b,bvs_b,bps_b=[],[],[]
for _ in range(300):
    si=np.random.default_rng().choice(items,len(items),replace=True)
    bd=pd.concat([df_nb[df_nb.pair_id==it] for it in si],ignore_index=True)
    try:
        m=smf.logit(FULL_F,data=bd).fit(disp=False,maxiter=300)
        p=m.params
        bP=p.get("P",0); bV=p.get("V",0); bS=p.get("S",0)
        bpv_b.append(safe_bic(p.get("PxV",0),bP,bV))
        bvs_b.append(safe_bic(p.get("VxS",0),bV,bS))
        bps_b.append(safe_bic(p.get("PxS",0),bP,bS))
    except: pass
bpv_b=[x for x in bpv_b if not np.isnan(x)]
bvs_b=[x for x in bvs_b if not np.isnan(x)]
bps_b=[x for x in bps_b if not np.isnan(x)]
bic_cis={"BIC_PV":dict(mean=round(float(np.mean(bpv_b)),3),ci=[round(float(np.percentile(bpv_b,2.5)),3),round(float(np.percentile(bpv_b,97.5)),3)]),
         "BIC_VS":dict(mean=round(float(np.mean(bvs_b)),3),ci=[round(float(np.percentile(bvs_b,2.5)),3),round(float(np.percentile(bvs_b,97.5)),3)]),
         "BIC_PS":dict(mean=round(float(np.mean(bps_b)),3),ci=[round(float(np.percentile(bps_b,2.5)),3),round(float(np.percentile(bps_b,97.5)),3)])}
print(f"  BIC_PV={bic_cis['BIC_PV']['mean']:.3f} {bic_cis['BIC_PV']['ci']}")
print(f"  BIC_VS={bic_cis['BIC_VS']['mean']:.3f} {bic_cis['BIC_VS']['ci']}")
print(f"  BIC_PS={bic_cis['BIC_PS']['mean']:.3f} {bic_cis['BIC_PS']['ci']}")

# Sensitivity analysis
print("Sensitivity analysis...")
sens_results={"base":{"mean":round(bic_cis["BIC_PV"]["mean"],3),"lo":bic_cis["BIC_PV"]["ci"][0],"hi":bic_cis["BIC_PV"]["ci"][1]}}
for label,bpv_mult in [("+30%",1.30),("-30%",0.70),("alt_bPS",1.0),("alt_bVS",1.0)]:
    rng2=np.random.default_rng(77)
    bpv_sim=DGP["bPV"]*bpv_mult; bvs_sim=DGP["bVS"]*(-1.67 if label=="alt_bVS" else 1.0)
    bps_sim=0.20 if label=="alt_bPS" else DGP["bPS"]
    svals=[]
    for rep in range(80):
        ir=rng2.normal(0,0.22,200); tr=rng2.normal(0,0.10,4)
        sr=[]
        for k in range(200):
            ti=k%4
            for ji,(jn,jc) in enumerate(JUDGES.items()):
                for P in [0,1]:
                    for V in [0,1]:
                        for S in [0,1]:
                            eta=(jc["alpha"]+jc["aP"]*P+jc["aV"]*V+jc["aS"]*S
                                 +bpv_sim*P*V+bvs_sim*V*S+bps_sim*P*S
                                 +jc["u0"]+ir[k]+tr[ti])
                            sr.append(dict(judge_id=ji,P=P,V=V,S=S,
                                          PxV=P*V,PxS=P*S,VxS=V*S,PxVxS=P*V*S,
                                          Y_bin=int(rng2.binomial(1,sigmoid(eta)))))
        ds=pd.DataFrame(sr)
        try:
            m=smf.logit(FULL_F,data=ds).fit(cov_type="cluster",cov_kwds={"groups":ds["judge_id"]},disp=False,maxiter=300)
            p=m.params; bP=p.get("P",0); bV=p.get("V",0)
            svals.append(safe_bic(p.get("PxV",0),bP,bV))
        except: pass
    svals=[x for x in svals if not np.isnan(x)]
    if svals:
        sens_results[label]=dict(mean=round(float(np.mean(svals)),3),
                                  lo=round(float(np.percentile(svals,2.5)),3),
                                  hi=round(float(np.percentile(svals,97.5)),3))
        print(f"  {label}: {sens_results[label]['mean']:.3f} [{sens_results[label]['lo']:.3f},{sens_results[label]['hi']:.3f}]")

# Correction comparison
print("Correction comparison...")
def run_corr(bic_lvls=[0.0,0.5,1.0,1.5,2.0],n_reps=200,n_cal=400,n_eval=1600,seed=42,nc_list=[100,500,2000]):
    rng3=np.random.default_rng(seed)
    res={b:{nc:{"seq":[],"jbc":[],"no":[]} for nc in nc_list} for b in bic_lvls}
    for bic in bic_lvls:
        bPV=bic*np.sqrt(0.50*0.55)
        for _ in range(n_reps):
            rows2=[]
            for _ in range(max(nc_list)+n_eval//4):
                for P in [0,1]:
                    for V in [0,1]:
                        eta=0.50*P+0.55*V+bPV*P*V
                        rows2.append({"P":P,"V":V,"PxV":P*V,"Y":int(rng3.binomial(1,sigmoid(eta)))})
            dall=pd.DataFrame(rows2); deval=dall.tail(n_eval)
            r11=deval[(deval.P==1)&(deval.V==1)]["Y"].mean()
            rl=logit_(r11)
            raw_b=abs(r11-0.5)
            for nc in nc_list:
                dcal=dall.head(nc*4)
                res[bic][nc]["no"].append(raw_b)
                try:
                    mP=smf.logit("Y~P",data=dcal).fit(disp=False,maxiter=200)
                    mV=smf.logit("Y~V",data=dcal).fit(disp=False,maxiter=200)
                    res[bic][nc]["seq"].append(abs(sigmoid(rl-mP.params["P"]-mV.params["V"])-0.5))
                except: pass
                try:
                    mJ=smf.logit("Y~P+V+PxV",data=dcal).fit(disp=False,maxiter=200)
                    res[bic][nc]["jbc"].append(abs(sigmoid(rl-mJ.params["P"]-mJ.params["V"]-mJ.params["PxV"])-0.5))
                except: pass
    summ={}
    for bic in bic_lvls:
        summ[bic]={}
        for nc in nc_list:
            seq=np.array(res[bic][nc]["seq"]); jbc=np.array(res[bic][nc]["jbc"])
            no=np.array(res[bic][nc]["no"])
            paired=[(s-j) for s,j in zip(res[bic][nc]["seq"],res[bic][nc]["jbc"]) if not np.isnan(j)]
            t,tp=stats.ttest_1samp(paired,0) if len(paired)>1 else (np.nan,np.nan)
            gain=(seq.mean()-jbc.mean())/seq.mean()*100 if seq.mean()>0 else 0
            summ[bic][nc]=dict(no_mean=round(float(no.mean()),4),seq_mean=round(float(seq.mean()),4),
                               seq_se=round(float(seq.std()/np.sqrt(len(seq))),4),
                               jbc_mean=round(float(jbc.mean()),4),
                               jbc_se=round(float(jbc.std()/np.sqrt(len(jbc))),4),
                               gain_pct=round(gain,1),p_val=round(float(tp),4) if not np.isnan(tp) else None)
    return summ
corr=run_corr()

# Theorem table
def seq_res_fn(bPV,n=10000,seed=0):
    rng4=np.random.default_rng(seed); rows3=[]
    for _ in range(n//4):
        for P in [0,1]:
            for V in [0,1]:
                eta=0.50*P+0.55*V+bPV*P*V
                rows3.append({"P":P,"V":V,"PxV":P*V,"Y":int(rng4.binomial(1,sigmoid(eta)))})
    d=pd.DataFrame(rows3)
    mP=smf.logit("Y~P",data=d).fit(disp=False,maxiter=300)
    mV=smf.logit("Y~V",data=d).fit(disp=False,maxiter=300)
    r11=d[(d.P==1)&(d.V==1)]["Y"].mean()
    return abs(sigmoid(logit_(r11)-mP.params["P"]-mV.params["V"])-0.5)
bpv_g=np.linspace(0,1.1,30)
bic_g=[b/np.sqrt(0.50*0.55) for b in bpv_g]
res_g=[seq_res_fn(b) for b in bpv_g]
null_res=res_g[0]
thresh_idx=next((i for i,r in enumerate(res_g) if r>0.020),len(res_g)-1)
thresh_bic=round(bic_g[thresh_idx],2)
theorem_rows=[dict(bPV=round(b,3),BIC=round(bc,2),residual=round(r,4),ratio=round(r/max(null_res,1e-6),1))
              for b,bc,r in zip(bpv_g[::4],bic_g[::4],res_g[::4])]

reg_p=m_pool.params; reg_se=m_pool.bse; reg_z=m_pool.tvalues; reg_pv=m_pool.pvalues
reg={t:dict(b=float(reg_p[t]),se=float(reg_se[t]),z=float(reg_z[t]),p=float(reg_pv[t]))
     for t in ["Intercept","P","V","S","PxV","PxS","VxS","PxVxS"]}

# Save numbers
def ns(v):
    if isinstance(v,(float,np.floating)) and np.isnan(v): return None
    if isinstance(v,(np.floating,np.integer)): return float(v)
    return v
N=dict(bic_pool={k:ns(v) for k,v in bpool.items()},
       lrt=dict(PxV=dict(chi2=round(lrt_pv[0],3),p=round(lrt_pv[1],8)),
                VxS=dict(chi2=round(lrt_vs[0],3),p=round(lrt_vs[1],8)),
                PxS=dict(chi2=round(lrt_ps[0],3),p=round(lrt_ps[1],5))),
       bic_cis=bic_cis, regression=reg,
       reg_meta=dict(N=int(len(df_nb)),LL=float(m_pool.llf),AIC=float(m_pool.aic)),
       dwr={f:{k:(round(float(v),5) if isinstance(v,float) else v) for k,v in d.items()} for f,d in pdwr.items()},
       judge_bic={j:{k:ns(v) for k,v in b.items()} for j,b in jbic.items()},
       task_bic={t:{k:ns(v) for k,v in b.items()} for t,b in tbic.items()},
       theorem=theorem_rows,null_res=round(null_res,4),thresh_bic=thresh_bic,
       sensitivity=sens_results,correction_sim={str(k):{str(nc):{kk:ns(vv) for kk,vv in sv.items()} for nc,sv in v.items()} for k,v in corr.items()})
(DATA/"numbers.json").write_text(json.dumps(N,indent=2,default=str))
print(f"\nKEY NUMBERS:")
print(f"  BIC_PV={N['bic_pool']['BIC_PV']:.4f} chi2={N['lrt']['PxV']['chi2']} p={N['lrt']['PxV']['p']:.2e}")
print(f"  BIC_VS={N['bic_pool']['BIC_VS']:.4f} chi2={N['lrt']['VxS']['chi2']} p={N['lrt']['VxS']['p']:.2e}")
print(f"  BIC_PS={N['bic_pool']['BIC_PS']:.4f} p={N['lrt']['PxS']['p']:.4f}")
print(f"  beta_PV={reg['PxV']['b']:.4f} z={reg['PxV']['z']:.3f}")
print(f"  beta_VS={reg['VxS']['b']:.4f} z={reg['VxS']['z']:.3f}")
print(f"  BIC_PV CI: {bic_cis['BIC_PV']['ci']}")
print(f"  Thresh={thresh_bic}  null_res={null_res:.4f}")
print(f"  JBC N=2000 BIC=2.0 gain={corr[2.0][2000]['gain_pct']:.1f}%")

# ── FIGURES 1-7 ────────────────────────────────────────────────────────
print("\nGenerating figures...")

# FIG 1: BIC Heatmap
fig,ax=plt.subplots(figsize=(7.2,4.6))
cols=["BIC_PV","BIC_PS","BIC_VS"]
clbl=["$\\mathcal{B}_P{\\times}\\mathcal{B}_V$","$\\mathcal{B}_P{\\times}\\mathcal{B}_S$","$\\mathcal{B}_V{\\times}\\mathcal{B}_S$"]
mat=np.array([[jbic[j].get(c,np.nan) for c in cols] for j in jnames],dtype=float)
pr=np.array([bpool.get(c,np.nan) for c in cols])
mfull=np.vstack([mat,pr]); rlbls=jnames+["Pooled"]
vmax=max(np.nanmax(np.abs(mfull)),0.5)
norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax)
im=ax.imshow(mfull,aspect="auto",cmap="RdBu_r",norm=norm,extent=[-0.5,2.5,len(rlbls)-0.5,-0.5])
for i in range(len(rlbls)):
    for j in range(3):
        v=mfull[i,j]
        if np.isnan(v): txt,fw="n/a","normal"
        else:
            txt=f"{'+'if v>=0 else'−'}{abs(v):.2f}"; fw="bold" if abs(v)>0.7 else "normal"
        clr="white" if abs(v if not np.isnan(v) else 0)>0.5*vmax else "black"
        ax.text(j,i,txt,ha="center",va="center",fontsize=10,color=clr,fontweight=fw)
ax.axhline(len(jnames)-0.5,color="black",lw=2.0)
ax.set_xticks([0,1,2]); ax.set_xticklabels(clbl,fontsize=11)
ax.set_yticks(range(len(rlbls))); ax.set_yticklabels(rlbls,fontsize=10)
ax.set_xlabel("Bias Pair",fontsize=11)
ax.set_title("Bias Interaction Coefficients (BIC)\nRed: amplifying (+)  Blue: suppressive (−)  n/a: undefined",fontsize=11,pad=7)
cbar=fig.colorbar(im,ax=ax,fraction=0.027,pad=0.01)
cbar.set_label("BIC",fontsize=10); cbar.ax.axhline(0,color="black",lw=0.9,alpha=0.5)
plt.tight_layout(); fig.savefig(FIGS/"fig1_bic_heatmap.png"); plt.close(); print("  ✓ fig1")

# FIG 2: Forest plot
fig,axes=plt.subplots(1,3,figsize=(13,5.2),sharey=True)
fcmap={"P":W["blue"],"V":W["orange"],"S":W["green"]}
titls={"P":"Position Bias  $\\mathcal{B}_P$","V":"Verbosity Bias  $\\mathcal{B}_V$","S":"Self-Preference  $\\mathcal{B}_S$"}
for ax2,fac in zip(axes,["P","V","S"]):
    for i,jn in enumerate(jnames):
        d=jdwr[jn][fac]; sig=d["sig"]
        ax2.errorbar(d["d"],i,xerr=[[d["d"]-d["lo"]],[d["hi"]-d["d"]]],fmt="o",color=fcmap[fac],
                    markerfacecolor=fcmap[fac] if sig else "white",markeredgecolor=fcmap[fac],
                    markeredgewidth=1.8,capsize=5,capthick=1.8,elinewidth=1.8,markersize=10,zorder=3)
    pd_=pdwr[fac]; yp=len(jnames)+0.8
    ax2.errorbar(pd_["d"],yp,xerr=[[pd_["d"]-pd_["lo"]],[pd_["hi"]-pd_["d"]]],
                fmt="D",color="black",markerfacecolor="black",capsize=5,capthick=1.8,elinewidth=1.8,markersize=11,zorder=4)
    ax2.axvline(0,color="black",lw=0.9,ls="--",alpha=0.5)
    ax2.axhline(len(jnames)+0.3,color=W["grey"],lw=0.7,ls=":")
    ax2.set_yticks(list(range(len(jnames)))+[yp]); ax2.set_yticklabels(jnames+["Pooled ◆"],fontsize=9.5)
    ax2.set_xlabel("$\\Delta$WR  (95% CI)",fontsize=10)
    ax2.set_title(titls[fac],fontsize=11,color=fcmap[fac],fontweight="bold")
    ax2.set_ylim(-0.7,len(jnames)+1.5); ax2.spines["left"].set_visible(False)
sp=mpatches.Patch(facecolor=W["blue"],edgecolor="black",lw=0.6,label="Significant")
ns2=mpatches.Patch(facecolor="white",edgecolor="black",lw=1.2,label="n.s.")
axes[-1].legend(handles=[sp,ns2],loc="lower right",fontsize=9,framealpha=0.9)
fig.suptitle("Marginal Bias Magnitudes ($\\Delta$WR) by Simulated Agent",fontsize=12,y=1.01)
plt.tight_layout(); fig.savefig(FIGS/"fig2_forest_dwr.png"); plt.close(); print("  ✓ fig2")

# FIG 3: Interaction profiles
fig,axes=plt.subplots(1,3,figsize=(13,5.0))
pairs=[("P","V"),("V","S"),("P","S")]
plbl={("P","V"):("Position","Verbosity"),("V","S"):("Verbosity","Self-Pref."),("P","S"):("Position","Self-Pref.")}
lc={0:W["blue"],1:W["orange"]}
for ax3,(f1,f2) in zip(axes,pairs):
    cell=df.groupby([f1,f2])["Y_star"].agg(["mean","sem"]).reset_index()
    bk=f"BIC_{f1}{f2}"; bval=bpool.get(bk,bpool.get(f"BIC_{f2}{f1}",np.nan))
    for v2 in [0,1]:
        sub=cell[cell[f2]==v2].sort_values(f1)
        x=sub[f1].values.astype(float); y=sub["mean"].values; se=sub["sem"].values
        ax3.plot(x,y,marker="os"[v2],color=lc[v2],lw=2.2,ls="-" if v2==0 else "--",ms=10,
                label=f'{plbl[(f1,f2)][1]} = {"High" if v2==1 else "Low"}',zorder=3)
        ax3.fill_between(x,y-se,y+se,color=lc[v2],alpha=0.15)
    ax3.axhline(0.5,color=W["grey"],lw=0.9,ls=":",alpha=0.7)
    ax3.set_xticks([0,1]); ax3.set_xticklabels([f'{plbl[(f1,f2)][0]}={"Low" if v==0 else "High"}' for v in [0,1]],fontsize=9.5)
    ax3.set_ylabel("P(position-A preferred)",fontsize=10); ax3.set_ylim(0.35,0.74)
    ax3.legend(fontsize=9,loc="upper left",framealpha=0.9)
    bstr=f"BIC={bval:+.3f}" if not(isinstance(bval,float) and np.isnan(bval)) else "BIC=n/a"
    ax3.text(0.97,0.05,bstr,transform=ax3.transAxes,ha="right",va="bottom",fontsize=10,style="italic",
             bbox=dict(facecolor="white",edgecolor=W["grey"],alpha=0.85,boxstyle="round,pad=0.3"))
    y00=cell[(cell[f1]==0)&(cell[f2]==0)]["mean"].values; y10=cell[(cell[f1]==1)&(cell[f2]==0)]["mean"].values
    y01=cell[(cell[f1]==0)&(cell[f2]==1)]["mean"].values; y11=cell[(cell[f1]==1)&(cell[f2]==1)]["mean"].values
    if all(len(a)==1 for a in [y00,y10,y01,y11]):
        isc=(y11[0]-y10[0])-(y01[0]-y00[0])
        ax3.set_title(f"$\\mathcal{{B}}_{{{f1}}}\\times\\mathcal{{B}}_{{{f2}}}$ — {'amplifying' if isc>0 else 'suppressive'}",fontsize=11,pad=6)
fig.suptitle("Two-Way Interaction Cell-Mean Profiles\nDiverging = amplification  Converging = suppression  Parallel = independence",fontsize=11,y=1.02)
plt.tight_layout(); fig.savefig(FIGS/"fig3_interaction_profiles.png"); plt.close(); print("  ✓ fig3")

# FIG 4: Theorem + correction
fig,axes=plt.subplots(1,2,figsize=(13,5.2))
ax4=axes[0]
bpvc=np.array(bic_g); resc=np.array(res_g)
ax4.plot(bpvc,resc,lw=2.5,color=W["orange"],label="SC residual (simulation)",zorder=3)
ax4.plot(bpvc,np.array(bpv_g)/(2*4),lw=2.0,color=W["blue"],ls="--",label="Analytic bound $\\frac{1}{2}\\beta_{PV}\\cdot p(1-p)$")
ax4.axhline(null_res,color=W["grey"],lw=1.0,ls=":",label=f"Null baseline ({null_res:.4f})")
ax4.axvline(thresh_bic,color=W["red"],lw=1.4,ls="-.",alpha=0.85,label=f"Action threshold (BIC={thresh_bic})")
ax4.axvline(bpool["BIC_PV"],color="black",lw=1.6,ls="--",label=f"Observed BIC$_{{PV}}$={bpool['BIC_PV']:.2f}")
ax4.fill_between(bpvc,null_res,resc,where=(resc>null_res),color=W["orange"],alpha=0.15)
ax4.set_xlabel("$\\mathrm{BIC}_{PV}$",fontsize=11); ax4.set_ylabel("Residual bias  $(P{=}1,V{=}1)$",fontsize=10)
ax4.set_title("Theorem Verification: SC Residual\nGrows With $\\mathrm{BIC}_{PV}$",fontsize=11)
ax4.legend(fontsize=8.5,loc="upper left",framealpha=0.92); ax4.set_ylim(-0.003,max(resc)*1.28)
ax5=axes[1]
bic_vals=[0.0,0.5,1.0,1.5,2.0]; x5=np.arange(len(bic_vals)); w5=0.22
col_nc={100:W["orange"],500:W["blue"],2000:W["green"]}
seq_m=[corr[b][100]["seq_mean"] for b in bic_vals]
ax5.bar(x5-1.5*w5,seq_m,3*w5,color=W["grey"],alpha=0.35,edgecolor="black",lw=0.6,label="Sequential (SC)",zorder=1)
for i,(nc,col) in enumerate(col_nc.items()):
    jbc_m=[corr[b][nc]["jbc_mean"] for b in bic_vals]
    stars_=["**" if corr[b][nc]["p_val"] and corr[b][nc]["p_val"]<0.01
            else "*" if corr[b][nc]["p_val"] and corr[b][nc]["p_val"]<0.05
            else "" for b in bic_vals]
    off=(i-1)*w5
    ax5.bar(x5+off,jbc_m,w5,color=col,alpha=0.85,edgecolor="black",lw=0.6,label=f"JBC  $N_{{cal}}={nc}$",zorder=2)
    for xi,bv,s in zip(x5,jbc_m,stars_):
        if s: ax5.text(xi+off,bv+0.001,s,ha="center",va="bottom",fontsize=10,color=W["red"],fontweight="bold")
ax5.axhline(0.020,color=W["red"],lw=1.1,ls="--",alpha=0.8,label="Threshold (0.020)")
ax5.set_xticks(x5); ax5.set_xticklabels([f"BIC={b:.1f}" for b in bic_vals],fontsize=9.5)
ax5.set_ylabel("Mean |residual bias|",fontsize=10)
ax5.set_title("JBC vs. SC by Calibration Size\n(200 reps per BIC level)",fontsize=11)
ax5.legend(fontsize=8.5,loc="upper left",framealpha=0.92,ncol=2); ax5.set_ylim(0,max(seq_m)*1.4)
fig.suptitle("Correction Analysis: Theorem Verification and JBC Calibration Curve",fontsize=12,y=1.02)
plt.tight_layout(); fig.savefig(FIGS/"fig4_theorem_correction.png"); plt.close(); print("  ✓ fig4")

# FIG 5: Power curves
fig,axes=plt.subplots(1,2,figsize=(12,5.0))
def anl_power(N,bic,bP=0.50,bV=0.55,a=0.017):
    bpv=bic*np.sqrt(abs(bP)*abs(bV)); zc=stats.norm.ppf(1-a/2)
    z=bpv/(1./np.sqrt(N*0.25*0.1875))
    return 1-stats.norm.cdf(zc-z)+stats.norm.cdf(-zc-z)
bic_r=np.linspace(0,2.0,120)
for N,npairs,ls_,lc_ in [(8000,200,"-",W["blue"]),(14000,350,"--",W["orange"]),(20000,500,"-.",W["green"])]:
    axes[0].plot(bic_r,[anl_power(N,b) for b in bic_r],ls=ls_,lw=2.2,color=lc_,label=f"$N$={npairs} pairs ({N:,} judg.)")
axes[0].axhline(0.80,color=W["red"],lw=1.2,ls=":",alpha=0.8,label="Target=0.80")
axes[0].axvspan(0,thresh_bic,alpha=0.06,color=W["grey"]); axes[0].axvspan(thresh_bic,2.0,alpha=0.07,color=W["green"])
axes[0].axvline(thresh_bic,color=W["red"],lw=1.3,ls="-.",alpha=0.8,label=f"Action threshold ({thresh_bic})")
axes[0].axvline(bpool["BIC_PV"],color="black",lw=1.5,ls="--",label=f"Observed BIC={bpool['BIC_PV']:.2f}")
axes[0].set_xlabel("$|\\mathrm{BIC}_{PV}|$",fontsize=11); axes[0].set_ylabel("Power",fontsize=11)
axes[0].set_title("Analytical Power Curves\n($\\alpha=0.017$ Bonferroni)",fontsize=11)
axes[0].legend(fontsize=9,loc="lower right",framealpha=0.92); axes[0].set_ylim(0,1.04); axes[0].set_xlim(0,2.0)
bic_pts=[0.3,0.47,0.5,0.6,1.0,1.5]
x2b=np.arange(len(bic_pts)); w2b=0.38
axes[1].bar(x2b-w2b/2,[anl_power(8000,b) for b in bic_pts],w2b,label="Analytical",color=W["blue"],edgecolor="black",lw=0.7,alpha=0.85)
axes[1].bar(x2b+w2b/2,[anl_power(8000/(3.3**2),b) for b in bic_pts],w2b,label="ICC-adjusted",color=W["sky"],edgecolor="black",lw=0.7,alpha=0.85)
axes[1].axhline(0.80,color=W["red"],lw=1.4,ls="--",label="Target=0.80")
axes[1].set_xticks(x2b); axes[1].set_xticklabels([f"BIC={b}" for b in bic_pts],fontsize=9.0)
axes[1].set_ylabel("Power",fontsize=11); axes[1].set_title("Power by BIC ($N=200$ pairs)",fontsize=11)
axes[1].legend(fontsize=9.5,framealpha=0.92); axes[1].set_ylim(0,1.10)
plt.tight_layout(); fig.savefig(FIGS/"fig5_power_curves.png"); plt.close(); print("  ✓ fig5")

# FIG 6: Sensitivity + BIC CIs
fig=plt.figure(figsize=(13,5.0)); gs6=GridSpec(1,2,figure=fig,wspace=0.35)
ax6a=fig.add_subplot(gs6[0])
var_lbl=["Base\n(β$_{PV}$=0.55)","+30%\n(β$_{PV}$=0.72)","-30%\n(β$_{PV}$=0.39)","Alt β$_{PS}$=0.20","Alt β$_{VS}$=−0.50"]
smeans=[sens_results.get(k,{}).get("mean",0) for k in ["base","+30%","-30%","alt_bPS","alt_bVS"]]
slo=[sens_results.get(k,{}).get("mean",0)-sens_results.get(k,{}).get("lo",0) for k in ["base","+30%","-30%","alt_bPS","alt_bVS"]]
shi=[sens_results.get(k,{}).get("hi",0)-sens_results.get(k,{}).get("mean",0) for k in ["base","+30%","-30%","alt_bPS","alt_bVS"]]
cols6=[W["blue"],W["orange"],W["green"],W["purple"],W["sky"]]
y6=np.arange(len(var_lbl))
for i,(m,l,h,c) in enumerate(zip(smeans,slo,shi,cols6)):
    ax6a.errorbar(m,i,xerr=[[l],[h]],fmt="o",color=c,markerfacecolor=c,markeredgecolor="black",
                 markeredgewidth=0.8,capsize=5,capthick=1.8,elinewidth=1.8,markersize=10,zorder=3)
ax6a.axvline(thresh_bic,color=W["red"],lw=1.4,ls="--",alpha=0.85,label=f"Action threshold ({thresh_bic})")
ax6a.axvline(0,color="black",lw=0.8,ls=":",alpha=0.5)
ax6a.fill_betweenx([-0.6,len(var_lbl)-0.4],0,thresh_bic,alpha=0.07,color=W["grey"],label="SC adequate")
ax6a.fill_betweenx([-0.6,len(var_lbl)-0.4],thresh_bic,3.5,alpha=0.07,color=W["green"],label="JBC zone")
ax6a.set_yticks(y6); ax6a.set_yticklabels(var_lbl,fontsize=9)
ax6a.set_xlabel("Estimated BIC$_{PV}$ (95% CI, 80 reps)",fontsize=10)
ax6a.set_title("Sensitivity: BIC$_{PV}$ Across\nDGP Parameter Variations",fontsize=11)
ax6a.legend(fontsize=8.5,loc="lower right",framealpha=0.9); ax6a.set_xlim(-0.3,3.5); ax6a.set_ylim(-0.6,len(var_lbl)-0.4)

ax6b=fig.add_subplot(gs6[1])
bic_n=["BIC$_{PV}$\n(P×V)","BIC$_{VS}$\n(V×S)","BIC$_{PS}$\n(P×S)"]
bic_m=[bic_cis["BIC_PV"]["mean"],bic_cis["BIC_VS"]["mean"],bic_cis["BIC_PS"]["mean"]]
bic_lo=[bic_cis["BIC_PV"]["mean"]-bic_cis["BIC_PV"]["ci"][0],
        bic_cis["BIC_VS"]["mean"]-bic_cis["BIC_VS"]["ci"][0],
        bic_cis["BIC_PS"]["mean"]-bic_cis["BIC_PS"]["ci"][0]]
bic_hi=[bic_cis["BIC_PV"]["ci"][1]-bic_cis["BIC_PV"]["mean"],
        bic_cis["BIC_VS"]["ci"][1]-bic_cis["BIC_VS"]["mean"],
        bic_cis["BIC_PS"]["ci"][1]-bic_cis["BIC_PS"]["mean"]]
bic_c=[W["red"],W["blue"],W["grey"]]
x6b=np.arange(3)
ax6b.bar(x6b,bic_m,color=bic_c,alpha=0.82,edgecolor="black",lw=0.7,width=0.5)
ax6b.errorbar(x6b,bic_m,yerr=[bic_lo,bic_hi],fmt="none",color="black",capsize=6,capthick=2,elinewidth=2,zorder=4)
ax6b.scatter(x6b,[1.48,-0.83,0.13],marker="^",s=80,color="black",zorder=5,label="DGP true value")
ax6b.axhline(0,color="black",lw=0.8,ls="--",alpha=0.5)
ax6b.axhline(thresh_bic,color=W["red"],lw=1.2,ls=":",alpha=0.8,label=f"Action threshold ({thresh_bic})")
ax6b.axhline(-thresh_bic,color=W["blue"],lw=1.2,ls=":",alpha=0.8)
ax6b.set_xticks(x6b); ax6b.set_xticklabels(bic_n,fontsize=10)
ax6b.set_ylabel("BIC estimate (95% bootstrap CI)",fontsize=10)
ax6b.set_title("BIC Point Estimates with\n95% Bootstrap CIs",fontsize=11)
ax6b.legend(fontsize=9,framealpha=0.9); ax6b.set_ylim(-2.2,2.8)
for i,(m,) in enumerate(zip(bic_m,)):
    s="**" if i<2 else "n.s."
    clr=W["red"] if i<2 else W["grey"]
    ax6b.text(i,max(abs(m),0.5)*np.sign(m)+0.15*np.sign(m),s,ha="center",
              va="bottom" if m>0 else "top",fontsize=11,color=clr,fontweight="bold")
fig.suptitle("Robustness and Uncertainty of BIC Estimates",fontsize=12,y=1.02)
plt.tight_layout(); fig.savefig(FIGS/"fig6_sensitivity_ci.png"); plt.close(); print("  ✓ fig6")

# FIG 7: Practitioner workflow
fig7,ax7=plt.subplots(figsize=(9,7))
ax7.set_xlim(0,10); ax7.set_ylim(0,10); ax7.axis("off")
def box7(x,y,w,h,txt,fc,ec="black",fs=9.5,bold=False):
    ax7.add_patch(plt.Rectangle((x-w/2,y-h/2),w,h,facecolor=fc,edgecolor=ec,linewidth=1.5,zorder=3))
    ax7.text(x,y,txt,ha="center",va="center",fontsize=fs,fontweight="bold" if bold else "normal",zorder=4,multialignment="center")
def arr7(x1,y1,x2,y2,lbl="",color="black"):
    ax7.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="->",color=color,lw=1.6),zorder=2)
    if lbl: ax7.text((x1+x2)/2+0.1,(y1+y2)/2,lbl,fontsize=9,color=color,ha="left",va="center",style="italic")
box7(5,9.3,6.5,0.9,"STEP 1: Collect calibration set\n(100–500 pairs × 4 (P,V) conditions = 400–2,000 API calls)","#E8F4FD","#0072B2",bold=True)
box7(5,7.7,6.0,0.8,"STEP 2: Estimate BIC$_{PV}$ from calibration data  (Eq. 2)","#EEF7EE","#009E73")
box7(2.5,5.8,3.8,1.0,"BIC$_{PV}$ < 0.47\n(SC residual likely\n< 0.020 win-rate pts)","#F0F0F0","#999999")
box7(7.5,5.8,3.8,1.0,"BIC$_{PV}$ ≥ 0.47\n(SC residual may\nexceed 0.020)","#FFF3E0","#E69F00")
box7(2.5,3.8,3.8,1.0,"Apply Sequential Correction (SC)\nPosition-swap + length-control","#E8F8F5","#009E73")
box7(7.5,3.8,3.8,1.1,"N_cal ≥ 500?\nYes → apply JBC\nNo  → collect more or report\n    BIC as caveat","#FFF8E1","#E69F00",fs=9)
box7(7.5,2.0,3.8,0.9,"Apply Joint Bias Correction (JBC)\n(Eq. 4–5)","#FDE8E8","#D55E00")
box7(5,0.5,6.5,0.8,"STEP 3: Report BIC alongside results as calibration-quality indicator","#F5F5F5","#333333")
arr7(5,8.85,5,8.1); arr7(5,7.3,2.5,6.3,"BIC < 0.47",W["green"]); arr7(5,7.3,7.5,6.3,"BIC ≥ 0.47",W["orange"])
arr7(2.5,5.3,2.5,4.3); arr7(7.5,5.3,7.5,4.35); arr7(7.5,3.35,7.5,2.45,"N_cal ≥ 500",W["red"])
arr7(2.5,3.3,5,0.9); arr7(7.5,1.55,5,0.9)
ax7.text(2.5,1.6,"Also check BIC$_{PS}$, BIC$_{VS}$:\nnear-zero → independent corrections\nremain valid",
         fontsize=8.5,ha="center",va="center",style="italic",color=W["blue"],
         bbox=dict(facecolor="white",alpha=0.8,edgecolor=W["blue"],boxstyle="round,pad=0.3"))
ax7.set_title("Practitioner Decision Workflow for Bias-Interaction-Aware Correction",fontsize=11,pad=10)
plt.tight_layout(); fig7.savefig(FIGS/"fig7_workflow.png"); plt.close(); print("  ✓ fig7")

print("\nAll figures saved to:", FIGS)
print("All numbers saved to:", DATA/"numbers.json")
