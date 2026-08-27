from __future__ import annotations
import argparse, shutil, subprocess
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent
SPARSE_PATHS = (Path("dataset/eval_data/fastaEval"), Path("dataset/test_data/fastaTest"))
DEFAULT_DATASET_DIR = DEFAULT_ROOT.parent / "data" / "fgnnsol"

def count_files(path): return sum(p.is_file() for p in path.rglob("*")) if path.exists() else 0

def run(args, cwd=None):
    print("+", subprocess.list2cmdline([str(x) for x in args]))
    subprocess.run(args, cwd=cwd, check=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project_root",type=Path,default=DEFAULT_ROOT)
    ap.add_argument("--dataset_dir",type=Path); ap.add_argument("--repo_url",default="https://github.com/SCrownJ/FGNNSol.git")
    ap.add_argument("--remove_temp",action="store_true"); a=ap.parse_args()
    root=a.project_root.resolve(); dataset=(a.dataset_dir or DEFAULT_DATASET_DIR).resolve(); tmp=dataset/"_official_repo_tmp"
    targets=[dataset/"eval_data"/"fastaEval",dataset/"test_data"/"fastaTest"]
    if all(count_files(x)>0 for x in targets):
        print("Official FASTA already present; download skipped.")
    else:
        dataset.mkdir(parents=True,exist_ok=True)
        if not (tmp/".git").exists():
            if tmp.exists() and any(tmp.iterdir()): raise RuntimeError(f"Temporary path is non-empty but not a git repository: {tmp}")
            run(["git","clone","--depth","1","--filter=blob:none","--sparse",a.repo_url,str(tmp)])
        run(["git","sparse-checkout","set",*[x.as_posix() for x in SPARSE_PATHS]],cwd=tmp)
        for rel,target in zip(SPARSE_PATHS,targets):
            source=tmp/rel
            if not source.is_dir(): raise FileNotFoundError(f"Sparse checkout did not produce {source}")
            target.mkdir(parents=True,exist_ok=True)
            for src in source.rglob("*"):
                if src.is_file():
                    dst=target/src.relative_to(source); dst.parent.mkdir(parents=True,exist_ok=True)
                    if not dst.exists(): shutil.copy2(src,dst)
    for target in targets: print(f"{target}: {count_files(target)} files")
    if a.remove_temp and tmp.exists(): shutil.rmtree(tmp); print(f"Removed temporary repository: {tmp}")
    elif tmp.exists(): print(f"Temporary sparse repository retained at: {tmp}")

if __name__=="__main__": main()
