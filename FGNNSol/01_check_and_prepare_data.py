from __future__ import annotations
import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from utils.fasta_utils import normalize_sequence, read_fasta_directory

DEFAULT_ROOT=Path(__file__).resolve().parent
DEFAULT_DATASET_DIR=DEFAULT_ROOT.parent/"data"/"fgnnsol"
STANDARD=set("ACDEFGHIKLMNPQRSTVWY"); AMBIGUOUS=set("BJOUXZ")
SEQUENCE_NAMES=("sequence","seq","protein_sequence"); LABEL_NAMES=("solubility","solubility 37","solubility_score","label")

def read_csv_detect(path):
    for enc in ("utf-8-sig","utf-8","gb18030","latin1"):
        try: return pd.read_csv(path,encoding=enc),enc
        except UnicodeDecodeError: continue
    raise UnicodeError(f"Unable to decode {path}")

def choose(columns,candidates,kind):
    lookup={str(x).strip().lower():x for x in columns}
    for c in candidates:
        if c in lookup: return lookup[c]
    raise ValueError(f"Cannot identify {kind} column. Existing columns: {list(columns)}")

def length_stats(values):
    x=np.asarray(values,float)
    return {"count":int(len(x)),"min":int(x.min()),"max":int(x.max()),"mean":float(x.mean()),
            "median":float(np.median(x)),"q90":float(np.quantile(x,.90)),"q95":float(np.quantile(x,.95)),"q99":float(np.quantile(x,.99))}

def inspect(path, split):
    df,enc=read_csv_detect(path); seqcol=choose(df.columns,SEQUENCE_NAMES,"sequence")
    labelcol=choose(df.columns,LABEL_NAMES,"continuous solubility")
    if labelcol.lower()=="label" and "solubility" not in labelcol.lower():
        raise ValueError(f"Refusing classification-only label as regression target in {path}; columns={list(df.columns)}")
    out=df.copy(); out["sequence"] = out[seqcol].map(normalize_sequence)
    out["solubility"] = pd.to_numeric(out[labelcol],errors="coerce")
    idcols=[c for c in df.columns if str(c).strip().lower() in {"gene","uniprot","uniprot_name","pdb_filename","afname","id","sample_id"}]
    original=out[idcols[0]].astype(str) if idcols else pd.Series([str(i) for i in range(len(out))])
    out["original_id"]=original; out["sequence_length"]=out["sequence"].str.len()
    chars=Counter(ch for s in out["sequence"] for ch in set(s)-STANDARD)
    report={"path":str(path.resolve()),"encoding":enc,"columns":list(df.columns),"rows":len(df),
      "sequence_column":str(seqcol),"solubility_column":str(labelcol),"identifier_columns":[str(x) for x in idcols],
      "missing":df.isna().sum().astype(int).to_dict(),"duplicate_sequences":int(out["sequence"].duplicated(keep=False).sum()),
      "duplicate_primary_ids":int(out["original_id"].duplicated(keep=False).sum()),"label_min":float(out["solubility"].min()),
      "label_max":float(out["solubility"].max()),"labels_outside_0_1":int((~out["solubility"].between(0,1)).sum()),
      "nonstandard_characters":dict(chars),"length_stats":length_stats(out["sequence_length"])}
    if out["sequence"].eq("").any() or out["solubility"].isna().any(): raise ValueError(f"Missing/invalid sequence or solubility in {path}")
    if not out["solubility"].between(0,1).all():
        if split != "external":
            raise ValueError(f"Solubility outside [0,1] in {path}: {report['label_min']}..{report['label_max']}")
        print(f"WARNING: external labels outside [0,1] are preserved (not clipped/deleted): {path}, range={report['label_min']}..{report['label_max']}")
    out.attrs.update(source_csv=path.name,id_columns=idcols); return out,report

def tokens(value): return {x.lower() for x in re.findall(r"[A-Za-z0-9]+",str(value)) if x}

def map_fasta(csv, records, split, idcols):
    byseq=defaultdict(list)
    for i,s in enumerate(csv["sequence"]): byseq[s].append(i)
    used=set(); mapped=[]; unresolved=[]
    for rec in records:
        candidates=[i for i in byseq.get(rec["sequence"],[]) if i not in used]; method="sequence_exact"
        if len(candidates)!=1:
            ft=tokens(rec["fasta_header"]+" "+rec["fasta_filename"])
            idmatches=[]
            search=candidates if candidates else [i for i in csv.index if i not in used]
            for i in search:
                vals=" ".join(str(csv.at[i,c]) for c in idcols if c in csv)
                if ft & tokens(vals): idmatches.append(i)
            candidates=idmatches; method="sequence_plus_id" if byseq.get(rec["sequence"]) else "identifier"
        if len(candidates)==1:
            i=candidates[0]; used.add(i); mapped.append((i,rec,method))
        else: unresolved.append({**rec,"split":split,"candidate_count":len(candidates),"candidate_indices":";".join(map(str,candidates))})
    return mapped,unresolved

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project_root",type=Path,default=DEFAULT_ROOT); ap.add_argument("--dataset_dir",type=Path)
    ap.add_argument("--prepared_dir",type=Path); ap.add_argument("--reports_dir",type=Path); a=ap.parse_args()
    root=a.project_root.resolve(); ds=(a.dataset_dir or DEFAULT_DATASET_DIR).resolve(); prep=(a.prepared_dir or root/"prepared").resolve(); reports=(a.reports_dir or root/"reports").resolve()
    prep.mkdir(parents=True,exist_ok=True); reports.mkdir(parents=True,exist_ok=True)
    paths={"train":ds/"eSol_train.csv","combined":ds/"eSol_test.csv","external":ds/"S.cerevisiae_test.csv"}
    data={}; report={}
    for name,path in paths.items(): data[name],report[name]=inspect(path,name); print(f"{name}: rows={len(data[name])}, columns={report[name]['columns']}")
    # Persist CSV-only diagnostics even when official FASTA retrieval is blocked.
    (reports/"data_check_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    csv_stats=[{"split_or_source":name,**item["length_stats"]} for name,item in report.items()]
    pd.DataFrame(csv_stats).to_csv(reports/"source_sequence_length_stats.csv",index=False)
    eval_records=read_fasta_directory(ds/"eval_data"/"fastaEval"); test_records=read_fasta_directory(ds/"test_data"/"fastaTest")
    combined=data["combined"]; idcols=combined.attrs["id_columns"]
    em,eu=map_fasta(combined,eval_records,"validation",idcols); tm,tu=map_fasta(combined,test_records,"internal_test",idcols)
    unresolved=eu+tu
    assigned_eval={x[0] for x in em}; assigned_test={x[0] for x in tm}; overlap=assigned_eval&assigned_test
    if unresolved or overlap:
        pd.DataFrame(unresolved).to_csv(prep/"unresolved_mapping.csv",index=False)
        raise RuntimeError(f"Official split mapping unresolved={len(unresolved)}, overlap={len(overlap)}; see {prep/'unresolved_mapping.csv'}")
    if assigned_eval|assigned_test != set(combined.index): raise RuntimeError("Official FASTA records do not cover every eSol_test.csv row")
    lookup={i:(rec,method,"validation") for i,rec,method in em}; lookup.update({i:(rec,method,"internal_test") for i,rec,method in tm})
    def canonical(frame,split,mapping=None):
        rows=[]
        for pos,(i,row) in enumerate(frame.iterrows()):
            rec,method=(mapping[i][0],mapping[i][1]) if mapping else ({"fasta_filename":""},"source_csv")
            oid=str(row["original_id"]); rows.append({"sample_id":f"{split}_{pos:04d}_{oid}","original_id":oid,"split":split,
              "sequence":row["sequence"],"sequence_length":int(row["sequence_length"]),"solubility":float(row["solubility"]),
              "source_csv":frame.attrs.get("source_csv","eSol_test.csv"),"fasta_filename":rec["fasta_filename"],"mapping_method":method})
        return pd.DataFrame(rows)
    train=canonical(data["train"],"train"); ext=canonical(data["external"],"external_test")
    ev=canonical(combined.loc[sorted(assigned_eval)],"validation",lookup); it=canonical(combined.loc[sorted(assigned_test)],"internal_test",lookup)
    expected={"train":(train,2019),"validation":(ev,268),"internal_test":(it,392),"external_test":(ext,108)}
    for split,(frame,n) in expected.items():
        if len(frame)!=n: raise AssertionError(f"{split}: expected {n}, found {len(frame)}")
    train.to_csv(prep/"eSol_train.csv",index=False); ev.to_csv(prep/"eSol_eval.csv",index=False); it.to_csv(prep/"eSol_internal_test.csv",index=False); ext.to_csv(prep/"S_cerevisiae_external_test.csv",index=False)
    manifest=pd.concat([x[0] for x in expected.values()],ignore_index=True); manifest.to_csv(prep/"split_manifest.csv",index=False)
    pd.DataFrame([{"split":s,"samples":len(f),"mapping_methods":json.dumps(f["mapping_method"].value_counts().to_dict())} for s,(f,_) in expected.items()]).to_csv(prep/"split_mapping_report.csv",index=False)
    cross={"train_vs_validation":len(set(train.sequence)&set(ev.sequence)),"train_vs_internal_test":len(set(train.sequence)&set(it.sequence)),"train_vs_external_test":len(set(train.sequence)&set(ext.sequence))}
    report["official_mapping"]={"fasta_eval_files":len(eval_records),"fasta_test_files":len(test_records),"unresolved":0,"cross_split_duplicate_sequences":cross}
    stats=[]
    for split,(frame,_) in expected.items(): stats.append({"split":split,**length_stats(frame.sequence_length)})
    pd.DataFrame(stats).to_csv(reports/"sequence_length_stats.csv",index=False)
    (reports/"sequence_length_stats.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    (reports/"data_check_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(pd.DataFrame(stats).to_string(index=False)); print("Cross-split exact sequence overlaps:",cross); print(f"Prepared manifest: {prep/'split_manifest.csv'}")

if __name__=="__main__": main()
