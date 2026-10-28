#!/usr/bin/env python3
"""English LiveASR chunk test for LFM3-Audio-2B, using the official streaming API.
Audio is simulated at 100ms microphone increments, NOT actual real-time PSTN input.
This yields per-chunk compute latency rather than a live human-call response benchmark.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch, soundfile as sf
from transformers import AutoModelForMultimodalLM, AutoProcessor

def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="model")
    ap.add_argument("--audio",default="results/bf16/en_phone16k.wav")
    ap.add_argument("--overlay",default=None)
    ap.add_argument("--outdir",default="results/streaming-bf16")
    a=ap.parse_args()
    root=Path(a.outdir); root.mkdir(parents=True,exist_ok=True)
    model=AutoModelForMultimodalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map="cuda:0",trust_remote_code=True).eval()
    processor=AutoProcessor.from_pretrained(a.model,trust_remote_code=True)
    if a.overlay:
        from overlay_runtime import apply_overlay
        info=apply_overlay(model,a.overlay)
    else:
        info=None
    wav,sr=sf.read(a.audio,dtype="float32")
    if sr!=16000: raise RuntimeError(f"Expected 16k sampled phone simulation, found {sr}")
    if wav.ndim!=1: wav=wav.mean(1)
    tokenizer=processor.tokenizer
    chunk_frames=8*model.config.interleaved_n_audio
    text_prompt=tokenizer("<|startoftext|><|im_start|>system\nPerform live ASR.<|im_end|>\n<|im_start|>assistant\n",add_special_tokens=False,return_tensors="pt").input_ids.to("cuda:0")
    suppress=tokenizer.convert_tokens_to_ids(["<|mixed_end|>","<|endoftext|>"])
    stop=tokenizer.convert_tokens_to_ids(["<|endoftext|>","<|im_start|>"])
    state=None
    feature_cache=None
    mel=torch.empty((1,processor.feature_extractor.feature_size,0),device="cuda:0")
    decoded=[]
    latency=[]
    def transcribe(frames,is_final=False):
        nonlocal state
        sync(); t0=time.perf_counter()
        with torch.inference_mode():
            result=model.transcribe_live(frames,state,input_ids=text_prompt if state is None else None,is_final=is_final,
                                         max_new_tokens=96 if is_final else 48,
                                         suppress_token_ids=suppress,stop_token_ids=stop)
        sync()
        time_s=time.perf_counter()-t0
        state=result.state
        phrase=tokenizer.decode(result.sequences[0],skip_special_tokens=True)
        decoded.append(phrase)
        latency.append({"final":is_final,"seconds":round(time_s,4),"chars":len(phrase),"frames":int(frames.shape[-1])})
        print(f"CHUNK ({'final' if is_final else 'ongoing'}) compute={time_s*1000:.1f}ms text={phrase!r}",flush=True)
    t_total=time.perf_counter()
    for begin in range(0,len(wav),1600):
        feats,feature_cache=processor.feature_extractor.stream(wav[begin:begin+1600],feature_cache)
        mel=torch.cat((mel,feats.input_features.to("cuda:0")),dim=-1)
        while mel.shape[-1]>=chunk_frames:
            transcribe(mel[...,:chunk_frames])
            mel=mel[...,chunk_frames:]
    feats,feature_cache=processor.feature_extractor.stream([],feature_cache,is_final=True)
    mel=torch.cat((mel,feats.input_features.to("cuda:0")),dim=-1)
    while mel.shape[-1]>chunk_frames:
        transcribe(mel[...,:chunk_frames])
        mel=mel[...,chunk_frames:]
    transcribe(mel,is_final=True)
    report={"model":a.model,"precision":"GPTQ-Pro INT4 g64" if info else "BF16",
            "audio":a.audio,"audio_duration_s":len(wav)/sr,
            "chunk_target_sec":round(chunk_frames*0.01,2),
            "audio_frames_ms":100,"stream_compute_s":sum(k["seconds"] for k in latency),
            "simulation_wall_s":round(time.perf_counter()-t_total,3),
            "final_transcript":"".join(decoded),
            "latencies":latency,"number_of_chunks":len(latency)}
    (root/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("REPORT",json.dumps(report,indent=2,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
