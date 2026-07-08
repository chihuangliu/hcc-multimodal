import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from hcc_multimodal.eval.ensemble import load_ensemble_aligned
from hcc_multimodal.survival.grid_scores import route_grid_scores_ensemble
from hcc_multimodal.survival.cutoffs import CUTOFF_METHODS
from hcc_multimodal.survival.analysis import analyze_groups
from hcc_multimodal.survival.restricted import restricted_stats

mids=["a6f970d6","dc7e1d10","982a6fa2"]
train,blocks=load_ensemble_aligned(mids,"resection")
soramic,_=load_ensemble_aligned(mids,"soramic")
lus,_=load_ensemble_aligned(mids,"lusanne")
fs,model,k="Mutual Info","Elastic Net",85
oof,sc_so,bp=route_grid_scores_ensemble(fs,model,train,soramic.X,blocks,k)
_,sc_lu,_=route_grid_scores_ensemble(fs,model,train,lus.X,blocks,k)
_,sc_res,_=route_grid_scores_ensemble(fs,model,train,train.X,blocks,k)
print("insample resection score quantiles:",np.round(np.quantile(sc_res.values,[0,.25,.5,.75,1]),3))
print("soramic score quantiles:          ",np.round(np.quantile(sc_so.values,[0,.25,.5,.75,1]),3))
print("lusanne score quantiles:          ",np.round(np.quantile(sc_lu.values,[0,.25,.5,.75,1]),3))
print()
def r(x,n=3):
    return round(x,n) if x is not None and not (isinstance(x,float) and np.isnan(x)) else None
freeze=sc_res
rows=[]
for cut in ["median_frozen","kmeans_frozen","kmeans_log_frozen","youden_frozen","median_within"]:
    fn=CUTOFF_METHODS[cut]
    g_so,meta=fn(freeze if cut!="median_within" else None,train.rfs_2year,sc_so)
    fz=freeze if cut!="median_within" else None
    g_lu,_=fn(fz if cut!="median_within" else None,train.rfs_2year,sc_lu)
    g_re,_=fn(fz if cut!="median_within" else None,train.rfs_2year,sc_res)
    nso=(g_so=="high").sum(),(g_so=="low").sum()
    nlu=(g_lu=="high").sum(),(g_lu=="low").sum()
    nre=(g_re=="high").sum(),(g_re=="low").sum()
    # soramic tau24 logrank + full logrank
    st24=restricted_stats(g_so,sc_so,soramic.time,soramic.event,24.0)
    stfull=restricted_stats(g_so,sc_so,soramic.time,soramic.event,float("inf"))
    stlu=restricted_stats(g_lu,sc_lu,lus.time,lus.event,float("inf"))
    rows.append(dict(cutoff=cut,thr=round(meta["threshold"],4),
        soramic_split=f"{nso[0]}/{nso[1]}",lus_split=f"{nlu[0]}/{nlu[1]}",res_split=f"{nre[0]}/{nre[1]}",
        so_tau24_lr=r(st24["logrank_p"]),so_tau24_ptp=r((st24.get("rmst") or {}).get("point_p")),
        so_full_lr=r(stfull["logrank_p"]),so_full_hr=r(stfull["hr_high_vs_low"],2),
        lu_full_lr=r(stlu["logrank_p"]),lu_full_hr=r(stlu["hr_high_vs_low"],2)))
print(pd.DataFrame(rows).to_string(index=False))
