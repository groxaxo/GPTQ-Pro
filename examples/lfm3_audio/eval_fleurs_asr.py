#!/usr/bin/env python3
"""FLEURS held-out human-speech ASR test, BF16 vs GPTQ-Pro overlay.

Independent FLEURS audio is NOT in the quantization calibration set.
Source: google/fleurs (CC-BY-4.0), test split, en_us and es_419.
Signal is simulated 8 kHz mu-law telephone bandwidth and decoded to 16 kHz;
not a test with a real telephone network / live callers.
"""
from __future__ import annotations
import argparse, json, re, time, unicodedata
from pathlib import Path
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

def norm(s):
    s=unicodedata.normalize("NFKC",s).casefold()
    s=re.sub(r"[^\w\s]"," ",s,flags=re.UNICODE)
    return " ".join(s.split())
def edit_distance(a,b):
    m=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]+[0]*len(b)
        for j,y in enumerate(b,1):
            cur[j]=min(cur[j-1]+1,m[j]+1,m[j-1]+int(x!=y))
        m=cur
    return m[-1]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="model")
    ap.add_argument("--manifest",default="evaluation/fleurs/manifest.json")
    ap.add_argument("--outdir",default="results/fleurs-bf16")
    ap.add_argument("--overlay",default=None)
    ap.add_argument("--limit",type=int,default=0)
    a=ap.parse_args()
    base=Path(a.model).resolve()
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    samples=json.loads(Path(a.manifest).read_text())["samples"]
    if a.limit:samples=samples[:a.limit]
    model=AutoModelForMultimodalLM.from_pretrained(str(base),dtype=torch.bfloat16,device_map="cuda:0",trust_remote_code=True).eval()
    processor=AutoProcessor.from_pretrained(str(base),trust_remote_code=True)
    if a.overlay:
        from overlay_runtime import apply_overlay
        info=apply_overlay(model,a.overlay,device="cuda:0")
    else:info=None
    result={"model":str(base),"quantization":"GPTQ-Pro native INT4 g64" if info else "BF16",
            "data_source":"google/fleurs test split (CC-BY-4.0); independently recorded human read-speech, synthetic 8k mu-law simulation",
            "n":len(samples),"per_sample":[],"metrics":{},
            "steady_gpu_GiB":round(torch.cuda.memory_allocated()/2**30,3)}
    aggregate={}
    for i,row in enumerate(samples):
        lang=row["language"]
        conversation=[{"role":"system","content":f"Perform ASR in {'English' if lang=='en' else 'Spanish'}."},
                      {"role":"user","content":[{"type":"audio","audio":row["file"]}]}]
        t_pre=time.perf_counter()
        inp=processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt").to("cuda:0")
        torch.cuda.synchronize()
        prep_s=time.perf_counter()-t_pre
        t=time.perf_counter()
        with torch.inference_mode():
            output=model.generate(**inp,max_new_tokens=256,suppress_mixed=True,suppress_audio_out=True,return_dict_in_generate=True)
        torch.cuda.synchronize()
        gen_s=time.perf_counter()-t
        prediction=processor.decode(output.sequences[0,inp["input_ids"].shape[1]:],skip_special_tokens=True)
        reference=row["reference"]
        target_words=norm(reference).split()
        predicted_words=norm(prediction).split()
        errors=edit_distance(target_words,predicted_words)
        item={"sample":str(row["file"]),"locale":row["locale"],"duration_s":round(row["seconds"],3),
              "reference":reference,"hypothesis":prediction,"ref_words":len(target_words),
              "errors":int(errors),"wer":round(errors/max(1,len(target_words)),3),
              "preprocess_s":round(prep_s,3),"generation_s":round(gen_s,3)}
        result["per_sample"].append(item)
        stats=aggregate.setdefault(row["locale"],{"n":0,"ref_words":0,"errors":0,"audio_s":0,"compute_s":0})
        stats["n"]+=1;stats["ref_words"]+=len(target_words);stats["errors"]+=errors
        stats["audio_s"]+=row["seconds"];stats["compute_s"]+=gen_s
        print(f"{i+1:02d}/{len(samples)} {row['locale']} errors={errors}/{len(target_words)} gen_s={gen_s:.3f} {prediction[:95]!r}",flush=True)
    for lang,stats in aggregate.items():
        result["metrics"][lang]={**{k:round(v,3) if isinstance(v,float) else v for k,v in stats.items()},
            "wer":round(stats["errors"]/max(1,stats["ref_words"]),4),
            "compute_rtf":round(stats["compute_s"]/max(.01,stats["audio_s"]),4)}
    result["peak_gpu_GiB"]=round(torch.cuda.max_memory_allocated()/2**30,3)
    (out/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print("METRICS",json.dumps(result["metrics"],ensure_ascii=False),flush=True)
if __name__=="__main__":main()
