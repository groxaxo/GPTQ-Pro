#!/usr/bin/env python3
"""Prepare a larger holdout evaluation of *human* FLEURS read speech.

Retain only locally; Hugging Face model card publishes metrics + sample refs,
not third-party copyrighted recordings. Preserves 8 kHz telephone simulation.
"""
import argparse, io, json, shutil, pathlib, time
import numpy as np, soundfile as sf, librosa
from datasets import load_dataset, Audio
from bench_lfm3_call import phone_simulation
def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--per-language",type=int,default=50)
    cli.add_argument("--outdir",default="evaluation/fleurs_50")
    a=cli.parse_args()
    if a.per_language<1 or a.per_language>250: raise ValueError("Count out of allowed range")
    dest=pathlib.Path(a.outdir).resolve()
    dest.mkdir(parents=True,exist_ok=True)
    earlier=pathlib.Path("evaluation/fleurs").resolve()
    results=[]
    for locale in ["en_us","es_419"]:
        print("STREAM_DATASET",locale,flush=True)
        ds=load_dataset("google/fleurs",locale,split="test",streaming=True).cast_column("audio",Audio(decode=False))
        for idx,record in enumerate(ds):
            if idx>=a.per_language: break
            target=dest/f"{locale}_{idx:03d}.wav"
            prior=earlier/f"{locale}_{idx:02d}.wav"
            if not target.exists():
                if idx<12 and prior.is_file():
                    shutil.copyfile(prior,target)
                else:
                    source=record["audio"]
                    sound,sr=sf.read(io.BytesIO(source["bytes"]),dtype="float32")
                    if sound.ndim>1: sound=sound.mean(axis=1)
                    if sr!=16000:sound=librosa.resample(sound,orig_sr=sr,target_sr=16000)
                    phone_simulation(sound,16000,target)
            with sf.SoundFile(target) as f:
                secs=f.frames/f.samplerate
                if f.samplerate!=16000: raise ValueError(f"Incorrect audio rate: {target}")
            results.append({"language":"en" if locale=="en_us" else "es","locale":locale,
                           "id":int(record["id"]),"file":str(target),"reference":record["transcription"],
                           "seconds":round(secs,4),"index":idx})
            if idx%10==0 or idx==a.per_language-1:
                print("SAMPLE",locale,idx,round(secs,2),"sec",flush=True)
            # Continuous progress on incomplete interrupted runs.
            (dest/"manifest.json").write_text(json.dumps({"dataset":"google/fleurs",
                "split":"test","license":"CC-BY-4.0","note":"Human read speech; simulated 8k mu-law telephone band",
                "samples":results},indent=2,ensure_ascii=False),encoding="utf-8")
    print("READY_SAMPLES",len(results), "en",sum(x["locale"]=="en_us" for x in results),
          "es",sum(x["locale"]=="es_419" for x in results),flush=True)
if __name__=="__main__":main()
