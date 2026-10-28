#!/usr/bin/env python3
"""LFM3-Audio-2B: portable native GPTQ-Pro W4 G64 demonstration.

Requires Linux + NVIDIA CUDA sm_80+ and the *exact* GPTQ-Pro backend packaged
as vendor/GPTQ-Pro. The original LiquidAI BF16 model is fetched separately.
This is a custom selective overlay, NOT a native transformers checkpoint.
"""
import argparse, json, os, pathlib, time
import torch
import soundfile as sf
from huggingface_hub import snapshot_download
from transformers import AutoModelForMultimodalLM, AutoProcessor
from overlay_runtime import apply_overlay, REV

BASE_ID="LiquidAI/LFM3-Audio-2B"
RELEASE_ID="groxaxo/LFM3-Audio-2B-GPTQ-Pro-INT4"
HERE=pathlib.Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base",default=None,help="Local original BF16 base-model directory")
    p.add_argument("--artifact",default=None,help="Local model-repository directory with overlays/g64/manifest.json")
    p.add_argument("--mode",choices=["asr","voice","tts"],default="asr")
    p.add_argument("--audio",default=None,help="Input WAV (prefer 16k mono) for ASR or voice")
    p.add_argument("--language",choices=["en","es"],default="en")
    p.add_argument("--text",default=None,help="For TTS only")
    p.add_argument("--out",default="lfm3_reply.wav",help="Audio output WAV (24kHz)")
    p.add_argument("--bf16",action="store_true",help="Run source BF16 control without quantization")
    p.add_argument("--max-new-tokens",type=int,default=512)
    a=p.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0]<8:
        raise RuntimeError("Native GPTQ-Pro inference requires Linux NVIDIA GPU compute capability >= 8.0")
    if a.mode!="tts" and not a.audio:p.error("--audio is required for ASR/voice")
    if a.mode=="tts" and not a.text:p.error("--text is required for TTS")
    base=pathlib.Path(a.base).resolve() if a.base else snapshot_download(repo_id=BASE_ID,revision=REV)
    if a.artifact:
        artifact=pathlib.Path(a.artifact).resolve()
    elif (HERE/"overlays/g64/manifest.json").exists():
        artifact=HERE
    else:
        artifact=pathlib.Path(snapshot_download(repo_id=RELEASE_ID,allow_patterns=["overlays/g64/*","overlay_runtime.py"]))
    print("Model",BASE_ID,"base revision",REV,"checkpoint",base,flush=True)
    model=AutoModelForMultimodalLM.from_pretrained(str(base),dtype=torch.bfloat16,device_map="cuda:0",trust_remote_code=True).eval()
    processor=AutoProcessor.from_pretrained(str(base),trust_remote_code=True)
    codec=processor.audio_tokenizer.to("cuda:0").eval()
    if not a.bf16:
        apply_overlay(model,artifact/"overlays/g64/manifest.json",device="cuda:0")
    language={"en":"English","es":"Spanish"}[a.language]
    if a.mode=="tts":
        msgs=[{"role":"system","content":f"Perform TTS in {language} using a female voice."},
              {"role":"user","content":a.text},
              {"role":"assistant","content":"<|audio_start|>"}]
        kwargs={"continue_final_message":True}
    elif a.mode=="asr":
        msgs=[{"role":"system","content":f"Perform ASR in {language}."},
              {"role":"user","content":[{"type":"audio","audio":str(pathlib.Path(a.audio).resolve())}]}]
        kwargs={"add_generation_prompt":True}
    else:
        msgs=[{"role":"system","content":"Respond with interleaved text and audio. You are a friendly receptionist. Do not claim to have booked anything without a real booking backend. Reply in the caller's language."},
              {"role":"user","content":[{"type":"audio","audio":str(pathlib.Path(a.audio).resolve())}]},
              {"role":"assistant","content":"<|mixed_start|>"}]
        kwargs={"continue_final_message":True}
    inputs=processor.apply_chat_template(msgs,tokenize=True,return_dict=True,return_tensors="pt",**kwargs).to("cuda:0")
    torch.cuda.synchronize(); before=time.perf_counter()
    with torch.inference_mode():
        gen_kwargs={"max_new_tokens":a.max_new_tokens,"return_dict_in_generate":True}
        if a.mode=="asr":
            gen_kwargs.update({"suppress_mixed":True,"suppress_audio_out":True})
        else:
            gen_kwargs.update({"audio_temperature":1.0,"audio_top_k":128,"audio_repetition_penalty":1.1,"audio_repetition_window":4096})
        output=model.generate(**inputs,**gen_kwargs)
    torch.cuda.synchronize(); generation_s=time.perf_counter()-before
    text=processor.decode(output.sequences[0,inputs["input_ids"].shape[1]:],skip_special_tokens=True)
    result={"mode":a.mode,"precision":"BF16" if a.bf16 else "GPTQ-Pro-INT4",
            "text":text,"generation_s":round(generation_s,3),
            "gpu_memory_allocated_GiB":round(torch.cuda.memory_allocated()/2**30,3)}
    if a.mode!="asr":
        with torch.inference_mode():
            wav=codec.decode(output.audio_codes[:1,:,:output.audio_lengths[0]]).audio_values[0].float().cpu().numpy()
        sf.write(a.out,wav,24000)
        result["wav"]=str(pathlib.Path(a.out).resolve())
        result["duration_s"]=round(len(wav)/24000,3)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
