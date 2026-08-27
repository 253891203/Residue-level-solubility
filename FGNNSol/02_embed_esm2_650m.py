from __future__ import annotations
import argparse, contextlib, hashlib, json, os, time, traceback
from pathlib import Path
import pandas as pd
import torch
from utils.long_sequence import make_windows
from utils.training_utils import str2bool

DEFAULT_ROOT=Path(__file__).resolve().parent
MODEL="esm2_t33_650M_UR50D"

def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def fair_name(name): return Path(name).name.replace("facebook/", "")

def fair_checkpoint_candidates(root: Path, name: str):
    filename=f"{fair_name(name)}.pt"; home=Path.home()
    values=[root/"models"/filename,root/"pretrained_models"/filename,root/filename,
            home/".cache"/"torch"/"hub"/"checkpoints"/filename]
    if os.environ.get("TORCH_HOME"): values.insert(0,Path(os.environ["TORCH_HOME"])/"hub"/"checkpoints"/filename)
    return values

@contextlib.contextmanager
def legacy_torch_load():
    original=torch.load
    def patched(*args,**kwargs): kwargs.setdefault("weights_only",False); return original(*args,**kwargs)
    torch.load=patched
    try: yield
    finally: torch.load=original

def load_backend(name,backend,root,device,offline):
    requested=Path(name).expanduser(); hf_dirs=[requested,root/name,root/"models"/name,root/"pretrained_models"/name,root/"models"/fair_name(name),root/"pretrained_models"/fair_name(name)]
    hf_dir=next((p.resolve() for p in hf_dirs if p.is_dir() and (p/"config.json").exists()),None)
    fair_file=requested.resolve() if requested.is_file() and requested.suffix==".pt" else next((p.resolve() for p in fair_checkpoint_candidates(root,name) if p.is_file()),None)
    selected=("fair_esm" if fair_file else "transformers") if backend=="auto" else backend
    if offline: os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
    if selected=="fair_esm":
        try: import esm
        except ImportError as exc: raise ImportError("fair-esm backend requires: pip install fair-esm") from exc
        if fair_file:
            with legacy_torch_load(): model,alphabet=esm.pretrained.load_model_and_alphabet_local(str(fair_file))
            source=str(fair_file)
        elif offline:
            raise FileNotFoundError("Local fair-esm checkpoint not found. Searched: "+", ".join(map(str,fair_checkpoint_candidates(root,name))))
        else: model,alphabet=getattr(esm.pretrained,fair_name(name))(); source=fair_name(name)
        model.eval().requires_grad_(False).to(device)
        return {"backend":"fair_esm","model":model,"batch_converter":alphabet.get_batch_converter(),"source":source}
    try: from transformers import AutoTokenizer,AutoModel
    except ImportError as exc: raise ImportError("transformers backend requires transformers") from exc
    if offline and hf_dir is None: raise FileNotFoundError(f"Local Transformers ESM2 directory not found for {name}")
    source=str(hf_dir or name); tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=offline); model=AutoModel.from_pretrained(source,local_files_only=offline)
    if model.config.hidden_size!=1280: raise RuntimeError(f"Required hidden size 1280, loaded {model.config.hidden_size}")
    model.eval().requires_grad_(False).to(device)
    return {"backend":"transformers","model":model,"tokenizer":tokenizer,"source":source}

def embed_window(seq, bundle, device, dtype):
    amp=device.type=="cuda" and dtype in {"float16","bfloat16"}; amp_dtype=torch.float16 if dtype=="float16" else torch.bfloat16
    if bundle["backend"]=="fair_esm":
        _,_,tokens=bundle["batch_converter"]([("protein",seq)]); tokens=tokens.to(device)
        with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=amp): residue=bundle["model"](tokens,repr_layers=[33],return_contacts=False)["representations"][33][0,1:len(seq)+1]
    else:
        tokenizer=bundle["tokenizer"]; encoded={k:v.to(device) for k,v in tokenizer(seq,return_tensors="pt",add_special_tokens=True).items()}
        with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=amp): hidden=bundle["model"](**encoded).last_hidden_state[0]
        special=tokenizer.get_special_tokens_mask(encoded["input_ids"][0].tolist(),already_has_special_tokens=True); residue=hidden[torch.tensor([not x for x in special],device=device,dtype=torch.bool)]
    if residue.shape[0]!=len(seq): raise RuntimeError(f"Tokenizer residue mismatch: expected {len(seq)}, got {residue.shape[0]}")
    return residue.float().cpu()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project_root",type=Path,default=DEFAULT_ROOT); ap.add_argument("--prepared_dir",type=Path); ap.add_argument("--dataset_dir",type=Path)
    ap.add_argument("--embedding_dir",type=Path); ap.add_argument("--output_dir",type=Path); ap.add_argument("--model_name_or_path",default=MODEL); ap.add_argument("--backend",choices=["auto","fair_esm","transformers"],default="auto"); ap.add_argument("--offline",type=str2bool,nargs="?",const=True,default=True)
    ap.add_argument("--long_sequence_mode",choices=["error","chunk","truncate"],default="chunk"); ap.add_argument("--chunk_size",type=int,default=1022); ap.add_argument("--chunk_overlap",type=int,default=128)
    ap.add_argument("--max_tokens_per_batch",type=int,default=1024); ap.add_argument("--dtype",choices=["float32","float16","bfloat16"],default="float16"); ap.add_argument("--resume",type=str2bool,nargs="?",const=True,default=True); ap.add_argument("--limit_per_split",type=int,default=0); a=ap.parse_args()
    root=a.project_root.resolve(); prep=(a.prepared_dir or root/"prepared").resolve(); embdir=(a.embedding_dir or root/"cache"/"esm2_650m_embeddings").resolve(); reports=(a.output_dir or root/"reports").resolve()
    embdir.mkdir(parents=True,exist_ok=True); reports.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(prep/"split_manifest.csv");
    if a.limit_per_split>0: manifest=manifest.groupby("split",sort=False).head(a.limit_per_split)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); bundle=load_backend(a.model_name_or_path,a.backend,root,device,a.offline)
    source={"requested":a.model_name_or_path,"resolved":bundle["source"],"backend":bundle["backend"],"offline":a.offline}; print("Model source:",json.dumps(source,ensure_ascii=False)); (reports/"esm2_model_source.json").write_text(json.dumps(source,indent=2),encoding="utf-8"); print(f"Loaded ESM2-650M on {device}")
    rows=[]; failures=[]; started=time.time(); total_samples=len(manifest)
    print(f"Starting residue embedding: samples={total_samples}, output={embdir}",flush=True)
    for sample_no,(_,r) in enumerate(manifest.iterrows(),1):
        sid=str(r.sample_id); seq=str(r.sequence); out=embdir/f"{sid}.pt"; digest=sha(seq)
        try:
            if a.resume and out.exists():
                obj=torch.load(out,map_location="cpu",weights_only=False); x=obj["embedding"]
                if tuple(x.shape)==(len(seq),1280) and obj.get("sequence_sha256")==digest:
                    rows.append({**r.to_dict(),"embedding_path":str(out),"embedding_shape":f"{len(seq)}x1280","dtype":str(x.dtype),"model_name":a.model_name_or_path,"sequence_sha256":digest,"status":"ok","used_chunking":len(seq)>a.chunk_size,"chunk_count":obj.get("chunk_count",1),"chunk_size":a.chunk_size,"chunk_overlap":a.chunk_overlap})
                    if sample_no==1 or sample_no%10==0 or sample_no==total_samples:
                        elapsed=time.time()-started; print(f"[{sample_no}/{total_samples}] resume-skip sample={sid} split={r.split} length={len(seq)} elapsed={elapsed:.1f}s",flush=True)
                    continue
            if len(seq)>a.chunk_size and a.long_sequence_mode=="error": raise RuntimeError(f"Long sequence forbidden: sample={sid}, length={len(seq)}")
            if len(seq)>a.chunk_size and a.long_sequence_mode=="truncate": windows=[(0,a.chunk_size)]
            else: windows=make_windows(len(seq),a.chunk_size,a.chunk_overlap) if a.long_sequence_mode=="chunk" else [(0,len(seq))]
            total=torch.zeros(len(seq),1280); counts=torch.zeros(len(seq),1)
            for start,end in windows: total[start:end]+=embed_window(seq[start:end],bundle,device,a.dtype); counts[start:end]+=1
            if (counts==0).any(): raise RuntimeError("Truncate mode cannot satisfy full Lx1280 output; use chunk mode")
            x=total/counts
            if tuple(x.shape)!=(len(seq),1280): raise RuntimeError(f"Final embedding shape mismatch: {tuple(x.shape)}")
            save=x.to(getattr(torch,a.dtype)); torch.save({"embedding":save,"sequence_sha256":digest,"model_name":a.model_name_or_path,"chunk_count":len(windows)},out)
            rows.append({**r.to_dict(),"embedding_path":str(out),"embedding_shape":f"{len(seq)}x1280","dtype":str(save.dtype),"model_name":a.model_name_or_path,"sequence_sha256":digest,"status":"ok","used_chunking":len(windows)>1,"chunk_count":len(windows),"chunk_size":a.chunk_size,"chunk_overlap":a.chunk_overlap})
        except Exception as exc:
            failure={"script":Path(__file__).name,"sample_id":sid,"split":r.split,"file_path":str(out),"sequence_length":len(seq),"exception_type":type(exc).__name__,"exception_message":str(exc),"traceback":traceback.format_exc()}; failures.append(failure); rows.append({**r.to_dict(),"embedding_path":str(out),"model_name":a.model_name_or_path,"sequence_sha256":digest,"status":"failed"}); print("ERROR",failure)
        if sample_no==1 or sample_no%10==0 or sample_no==total_samples:
            elapsed=time.time()-started; rate=sample_no/elapsed if elapsed else 0; eta=(total_samples-sample_no)/rate if rate else float("inf")
            print(f"[{sample_no}/{total_samples}] sample={sid} split={r.split} length={len(seq)} chunks={len(windows) if 'windows' in locals() else '?'} elapsed={elapsed:.1f}s eta={eta/60:.1f}min",flush=True)
            pd.DataFrame(rows).to_csv(embdir/"embedding_manifest.partial.csv",index=False)
    pd.DataFrame(rows).to_csv(embdir/"embedding_manifest.csv",index=False); pd.DataFrame(failures).to_csv(reports/"embedding_failures.csv",index=False)
    if failures: raise RuntimeError(f"Embedding failed for {len(failures)} samples; see {reports/'embedding_failures.csv'}")

if __name__=="__main__": main()
