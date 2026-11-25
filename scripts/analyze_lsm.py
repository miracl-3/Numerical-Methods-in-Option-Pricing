"""Read saved LSM CSVs and render statistical figures. Never runs pricing code."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from experiment_files import ROOT, prepare_results, plot_locations
from plot_lsm_paper import save
from lsm_statistics import (audit_existing_results, existing_convergence_runs, missing_convergence_keys,
    summarize_prices, standardized_difference, relative_efficiency, paired_difference_summary,
    sha256, DEFAULT_ANALYSIS, T_975_DF4)

COLORS = {"plain": "#bd6026", "antithetic": "#00857d"}
LABELS = {"plain": "Plain LSM", "antithetic": "AV-LSM"}
ATM_CONTRACT = "S = K = 40; r = 6%; σ = 20%; T = 1 year; 50 exercise dates/year plus t = 0."
KEYS = ["spot", "volatility", "maturity"]
TABLE_NAMES = ("pricing_errors", "sampling_runs", "sampling_summary", "variance_reduction", "efficiency",
               "table2_pairs", "table2_summary", "missing_configurations")


def style():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.titlesize":11,
        "axes.labelsize":10, "figure.titlesize":14, "figure.titleweight":"bold",
        "axes.spines.top":False, "axes.spines.right":False, "legend.frameon":False,
        "figure.facecolor":"white", "savefig.facecolor":"white", "svg.hashsalt":"lsm-statistical-figures"})


def build_tables(frames, runs, reference, missing):
    published = frames["published_table1"][KEYS + ["paper_lsm_american", "paper_standard_error"]].rename(
        columns={"paper_lsm_american":"published_price", "paper_standard_error":"published_se"})
    trees = frames["paper_table"][KEYS + ["american_grid_reference_crr"]].rename(columns={"american_grid_reference_crr":"matched_crr"})
    errors = frames["control_table"].merge(published,on=KEYS,validate="many_to_one").merge(trees,on=KEYS,validate="many_to_one")
    if len(errors)!=40 or not np.allclose(errors.comparison_reference_crr, errors.matched_crr,atol=1e-12,rtol=0):
        raise ValueError("Saved CRR reference mapping is inconsistent.")
    errors["error_published"] = errors.price-errors.published_price
    errors["error_crr"] = errors.price-errors.matched_crr
    errors["z_published"] = [standardized_difference(p,r,s,rs) for p,r,s,rs in
                              zip(errors.price,errors.published_price,errors.standard_error,errors.published_se)]
    errors["z_crr"] = [standardized_difference(p,r,s) for p,r,s in zip(errors.price,errors.matched_crr,errors.standard_error)]
    errors = errors[KEYS+["sampling_method","paths","seed","price","standard_error","published_price","published_se",
                         "matched_crr","error_published","error_crr","z_published","z_crr"]]
    summaries=[]
    for (method, paths),group in runs.groupby(["sampling_method","paths"],sort=True):
        row=summarize_prices(group.price.tolist(),group.standard_error.tolist(),reference)
        row.update(sampling_method=method,paths=int(paths),mean_pathwise_se=group.pathwise_se.mean())
        summaries.append(row)
    sampling_summary=pd.DataFrame(summaries)
    reduction=[]
    common_paths=set(sampling_summary.query("sampling_method=='plain'").paths)&set(sampling_summary.query("sampling_method=='antithetic'").paths)
    for paths in sorted(common_paths):
        data=sampling_summary[sampling_summary.paths==paths].set_index("sampling_method")
        plain_var=data.loc["plain","empirical_sd"]**2
        av_var=data.loc["antithetic","empirical_sd"]**2
        reduction.append({"paths":int(paths),"runs_per_method":int(data.runs.min()),
                          "variance_factor_plain_over_av":plain_var/av_var,
                          "variance_reduction_percent":100*(1-av_var/plain_var)})
    repeats=frames["control_runs"]
    plain=repeats[repeats.sampling_method=="plain"]
    base_variance=plain.price.var(ddof=1)
    base_time=plain.runtime_ms.median()/1000
    efficiency=[]
    for method,group in repeats.groupby("sampling_method"):
        variance=group.price.var(ddof=1)
        duration=group.runtime_ms.median()/1000
        efficiency.append({"sampling_method":method,"paths":100000,"runs":len(group),"empirical_variance":variance,
            "median_runtime_seconds":duration,"runtime_q1_seconds":group.runtime_ms.quantile(.25)/1000,
            "runtime_q3_seconds":group.runtime_ms.quantile(.75)/1000,"variance_time_product":variance*duration,
            "relative_efficiency":relative_efficiency(base_variance,base_time,variance,duration)})
    data=frames["table2"]
    columns=KEYS+["seed","price","standard_error"]
    pairs=data.loc[data.valuation_mode=="in-sample",columns].merge(
        data.loc[data.valuation_mode=="out-of-sample",columns],on=KEYS+["seed"],suffixes=("_in","_out"),validate="one_to_one").merge(trees,on=KEYS)
    pairs["difference_in_minus_out"]=pairs.price_in-pairs.price_out
    pairs["error_in_crr"]=pairs.price_in-pairs.matched_crr
    pairs["error_out_crr"]=pairs.price_out-pairs.matched_crr
    table2=[]
    for key,group in pairs.groupby(KEYS,sort=True):
        if len(group)!=5: raise ValueError("Table 2 confidence intervals require the five saved paired runs.")
        row=dict(zip(KEYS,key))
        row.update(paired_difference_summary(group.price_in.tolist(),group.price_out.tolist(),T_975_DF4))
        row.update(runs=5,ci_level=.95,ci_method="paired Student t",degrees_of_freedom=4,critical_value=T_975_DF4,
            mean_error_in=group.error_in_crr.mean(),mean_error_out=group.error_out_crr.mean(),
            sd_in=group.price_in.std(ddof=1),sd_out=group.price_out.std(ddof=1))
        table2.append(row)
    missing_rows=[]
    for method,paths in sorted(set((m,n) for m,n,_ in missing)):
        seeds=[s for m,n,s in missing if m==method and n==paths]
        missing_rows.append({"sampling_method":method,"paths":paths,"missing_runs":len(seeds),
                             "first_seed":min(seeds),"last_seed":max(seeds),"spot":40,"strike":40,
                             "volatility":.2,"maturity":1,"exercise_dates":50})
    return {"pricing_errors":errors,"sampling_runs":repeats,"sampling_summary":sampling_summary,
        "variance_reduction":pd.DataFrame(reduction),"efficiency":pd.DataFrame(efficiency),
        "table2_pairs":pairs,"table2_summary":pd.DataFrame(table2),"missing_configurations":pd.DataFrame(missing_rows)}


def error_plots(errors,output):
    cases=errors[KEYS].drop_duplicates().sort_values(KEYS)
    labels=[f"{int(r.spot)}\n{int(r.volatility*100)}%\n{int(r.maturity)}y" for r in cases.itertuples()]
    for standardized in (False,True):
        fig,axes=plt.subplots(2,1,figsize=(12,6.8),sharex=True,layout="constrained")
        fig.suptitle("Standardized pricing discrepancies" if standardized else "Signed pricing errors")
        for i,target in enumerate(("published","crr")):
            axis=axes[i]
            if standardized:
                axis.axhspan(-1.96,1.96,color="#edf1f5")
                axis.axhline(-1.96,ls="--",color="#a54f57",lw=1)
                axis.axhline(1.96,ls="--",color="#a54f57",lw=1,label="±1.96 thresholds")
            axis.axhline(0,color="#526273",lw=1)
            for method,offset,marker in (("plain",-.10,"s"),("antithetic",.10,"o")):
                frame=cases.merge(errors[errors.sampling_method==method],on=KEYS)
                axis.scatter(np.arange(len(frame))+offset,frame[("z_" if standardized else "error_")+target],
                             color=COLORS[method],marker=marker,s=28,label=LABELS[method])
            axis.set(title="Published AV-LSM" if target=="published" else "Grid-matched American CRR",
                     ylabel="Standardized difference" if standardized else "Estimate − reference")
            axis.grid(axis="y",alpha=.2); axis.margins(y=.25); axis.legend(fontsize=8,ncol=3,loc="upper right")
        axes[1].set_xticks(range(len(labels)),labels,fontsize=8)
        axes[1].set_xlabel("Initial stock price / volatility / maturity")
        caption=("American puts; K = 40; r = 6%; 100,000 paths per estimate; 50 exercise dates/year plus t = 0.\n"
                 "Published: combined SE; CRR: our reported SE. Thresholds are descriptive; in-sample SEs are approximate."
                 if standardized else "American puts; K = 40; r = 6%; seed = 42; 100,000 paths per estimate.\n"
                 "CRR: 100,000 time steps, with the same 50 exercise dates/year plus t = 0.")
        fig.supxlabel(caption,fontsize=8.5)
        save(fig,output,"standardized_differences" if standardized else "signed_pricing_errors")


def sampling_plot(repeats,reference,output):
    fig,axis=plt.subplots(figsize=(7.5,5),layout="constrained")
    fig.suptitle("Sampling variability at 100,000 paths")
    for i,method in enumerate(LABELS):
        data=repeats[repeats.sampling_method==method].sort_values("seed")
        axis.scatter(i+np.linspace(-.14,.14,len(data)),data.price,color=COLORS[method],alpha=.55,s=24)
        axis.errorbar(i,data.price.mean(),yerr=data.price.std(ddof=1),fmt="D",ms=7,capsize=7,color="#263b57",
                      label="Mean ± empirical SD" if i==0 else None)
    axis.axhline(reference,color="#526273",ls="--",label="Grid-matched CRR")
    axis.set(xticks=[0,1],xticklabels=list(LABELS.values()),ylabel="Estimated price",xlim=(-.5,1.5))
    axis.grid(axis="y",alpha=.2); axis.legend(fontsize=9,loc="upper right")
    fig.supxlabel(ATM_CONTRACT+"\nDots: 30 saved runs per method. Whiskers show empirical SD.",fontsize=8.5)
    save(fig,output,"sampling_variability")


def calibration_plot(summary,output):
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout="constrained")
    fig.suptitle("Standard-error calibration")
    for method in LABELS:
        frame=summary[summary.sampling_method==method].sort_values("paths")
        axes[0].plot(frame.paths,frame.reported_se_over_empirical_sd,"o-" if len(frame)>1 else "s",
                     color=COLORS[method],label=LABELS[method])
    axes[0].axhline(1,color="#263b57",ls="--")
    axes[0].set(xscale="log",xlabel="Total simulated paths",ylabel="Mean reported SE / empirical SD",title="Agreement corresponds to one")
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{x:,.0f}"))
    axes[0].set_xticks([5000,10000,25000,50000,100000]); axes[0].tick_params(axis="x",labelrotation=35)
    for i,method in enumerate(LABELS):
        row=summary[(summary.sampling_method==method)&(summary.paths==100000)].iloc[0]
        for j,(field,color,label) in enumerate((("empirical_sd","#263b57","Empirical SD"),
                ("mean_reported_se","#00857d","Mean reported SE"),("mean_pathwise_se","#bd6026","Mean individual-path SE"))):
            axes[1].bar(i+(j-1)*.23,row[field],width=.21,color=color,label=label if i==0 else None)
    axes[1].set(xticks=[0,1],xticklabels=list(LABELS.values()),ylabel="Price units",title="100,000 paths")
    axes[1].set_ylim(0,summary[summary.paths==100000][["empirical_sd","mean_pathwise_se"]].to_numpy().max()*1.4)
    for axis in axes: axis.grid(axis="y",alpha=.2); axis.legend(fontsize=8)
    fig.supxlabel(ATM_CONTRACT+"\n30 runs per point. AV-LSM reported SE uses pairs; individual-path SE ignores covariance. Plain data below 100,000 are absent.",fontsize=8.5)
    save(fig,output,"standard_error_calibration")


def efficiency_plot(reduction,efficiency,output):
    row=reduction.loc[reduction.paths==100000].iloc[0]
    data=efficiency.set_index("sampling_method").loc[list(LABELS)]
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8),layout="constrained")
    fig.suptitle("Variance reduction and time efficiency")
    variances=[1.0,1/row.variance_factor_plain_over_av]
    axes[0].bar(range(2),variances,color=list(COLORS.values()))
    axes[0].axhline(1,color="#526273",ls="--")
    axes[0].text(1,variances[1]+.06,f"{row.variance_reduction_percent:.1f}% reduction",ha="center",fontsize=10)
    axes[0].set(title="Empirical variance",ylabel="Relative variance (plain = 1)",ylim=(0,1.25))
    axes[1].bar(range(2),data.relative_efficiency,color=list(COLORS.values()))
    axes[1].axhline(1,color="#526273",ls="--")
    for i,(_,estimate) in enumerate(data.iterrows()):
        axes[1].text(i,estimate.relative_efficiency+.06,
            f"{estimate.relative_efficiency:.2f}×\n{1000*estimate.median_runtime_seconds:.0f} ms median",ha="center",fontsize=9)
    axes[1].set(title="Computational efficiency",ylabel="Relative efficiency (plain = 1)",ylim=(0,data.relative_efficiency.max()*1.35))
    for axis in axes:
        axis.set_xticks([0,1],list(LABELS.values())); axis.grid(axis="y",alpha=.2)
    fig.supxlabel(ATM_CONTRACT+"\n100,000 paths; 30 runs per method. Efficiency = 1 / (variance × median runtime). Timings exclude CRR and depend on machine load.",fontsize=8.5)
    save(fig,output,"variance_and_efficiency")


def table2_plot(pairs,summary,output):
    fig,axes=plt.subplots(1,2,figsize=(13,6.3),sharey=True,layout="constrained")
    fig.suptitle("Table 2: in-sample versus out-of-sample")
    labels=[]
    for i,(key,group) in enumerate(pairs.groupby(KEYS,sort=True)):
        labels.append(f"S={int(key[0])}, σ={key[1]:.0%}, T={int(key[2])}y")
        row=summary[(summary.spot==key[0])&(summary.volatility==key[1])&(summary.maturity==key[2])].iloc[0]
        halfwidth=T_975_DF4*row.se_of_mean_difference
        axes[0].scatter(group.difference_in_minus_out,i+np.linspace(-.10,.10,len(group)),color="#00857d",s=23,alpha=.5,
                        label="Individual paired differences" if i==0 else None)
        axes[0].errorbar(row.mean_difference,i,xerr=halfwidth,fmt="D",capsize=4,color="#263b57",
                         label="Paired mean and 95% CI" if i==0 else None)
        for offset,mode,color in ((-.12,"in","#bd6026"),(.12,"out","#00857d")):
            halfwidth=T_975_DF4*row["sd_"+mode]/np.sqrt(row.runs)
            axes[1].errorbar(row["mean_error_"+mode],i+offset,xerr=halfwidth,fmt="o",capsize=4,color=color,
                            label=("In-sample" if mode=="in" else "Out-of-sample") if i==0 else None)
    axes[0].set(title="Paired mean difference (95% CI)",xlabel="In-sample − out-of-sample")
    axes[1].set(title="Mean error against grid-matched CRR (95% CI)",xlabel="Estimate − reference")
    for axis in axes:
        axis.axvline(0,color="#526273",ls="--"); axis.grid(axis="x",alpha=.2)
        axis.legend(fontsize=8,loc="upper center",bbox_to_anchor=(.5,-.14),ncol=2)
        axis.set_yticks(range(8),labels,fontsize=9); axis.set_ylim(7.6,-.6)
    fig.supxlabel("American puts; K = 40; r = 6%; 50 exercise dates/year plus t = 0.\n"
                  "AV-LSM: 100,000 training paths and 100,000 independent evaluation paths. Pointwise 95% Student-t intervals across five seeds (df = 4).\n"
                  "Intervals assume approximately normal run estimates/differences; no adjustment for comparing eight contracts.",fontsize=8.5)
    save(fig,output,"in_sample_out_of_sample")


def render(tables,output):
    style()
    errors=tables["pricing_errors"]
    reference=float(tables["sampling_runs"].comparison_reference_crr.iloc[0])
    error_plots(errors,output)
    sampling_plot(tables["sampling_runs"],reference,output)
    calibration_plot(tables["sampling_summary"],output)
    efficiency_plot(tables["variance_reduction"],tables["efficiency"],output)
    table2_plot(tables["table2_pairs"],tables["table2_summary"],output)


def analyze(output_dir=None,audit_only=False,figures_only=False):
    if figures_only:
        output=Path(output_dir or DEFAULT_ANALYSIS)
        manifest=json.loads((output/"manifest.json").read_text())
        for name,digest in manifest["inputs_sha256"].items():
            if sha256(ROOT/name)!=digest: raise ValueError("A source CSV changed; recompute statistical summaries in a fresh directory.")
        tables={name:pd.read_csv(output/f"{name}.csv",float_precision="round_trip") for name in TABLE_NAMES}
    else:
        frames,audit=audit_existing_results()
        runs=existing_convergence_runs(frames,audit["matched_crr_atm"])
        missing=missing_convergence_keys(runs)
        audit.update(missing_convergence_runs=len(missing),simulations_run=0,crr_calculations_run=0)
        if audit_only:
            print(json.dumps(audit,indent=2)); return
        output=prepare_results("lsm_analysis",output_dir,False)
        tables=build_tables(frames,runs,audit["matched_crr_atm"],missing)
        for name,frame in tables.items(): frame.to_csv(output/f"{name}.csv",index=False,float_format="%.17g")
        (output/"audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
        summary={"simulations_run":0,"crr_calculations_run":0,"reused_convergence_runs":len(runs),"missing_convergence_runs":len(missing),
            "standardized_threshold_counts":{method:{"published_outside_1_96":int((group.z_published.abs()>1.96).sum()),
                "crr_outside_1_96":int((group.z_crr.abs()>1.96).sum())} for method,group in tables["pricing_errors"].groupby("sampling_method")},
            "variance_reduction":tables["variance_reduction"].to_dict("records"),"efficiency":tables["efficiency"].to_dict("records"),
            "table2_mean_difference":tables["table2_pairs"].difference_in_minus_out.mean(),
            "table2_intervals_excluding_zero":int(((tables["table2_summary"].difference_ci_lower>0)|(tables["table2_summary"].difference_ci_upper<0)).sum())}
        (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
        manifest={"profile":"full","analysis_only":True,"generated_at_utc":datetime.now(timezone.utc).isoformat(),
            "inputs_sha256":{**audit["inputs_sha256"],**audit["manifest_sha256"]},
            "analysis_source_sha256":{name:sha256(ROOT/name) for name in ("scripts/analyze_lsm.py","scripts/lsm_statistics.py")},
            "simulations_run":0,"crr_calculations_run":0,"missing_convergence_runs":len(missing),
            "limitations":["In-sample SEs are approximate; published SE convention is unreported.",
                "The ±1.96 reference thresholds are descriptive, not simultaneous confidence tests.",
                "Table 2 intervals are pointwise paired Student-t intervals (df=4), assuming normal run differences; five seeds give limited precision.",
                "Shared seeds correlate contract comparisons; no multiple-comparison adjustment is applied.",
                "CRR retains numerical discretization error.","Runtime efficiency uses only same-build saved 100000-path timings and is machine/load dependent.",
                "Plain-LSM replicates at 5000,10000,25000,50000 paths are missing and are not generated."]}
        (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    _,figure_dir=plot_locations("lsm_analysis",output)
    render(tables,figure_dir)
    print(f"Saved six figure pairs using existing CSVs only: {figure_dir}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--audit-only",action="store_true")
    parser.add_argument("--figures-only",action="store_true",help="Redraw the saved statistical summaries without recalculating them.")
    args=parser.parse_args()
    try:
        analyze(args.output_dir,args.audit_only,args.figures_only)
    except (ValueError,RuntimeError,OSError) as error:
        parser.exit(1,str(error)+"\n")
