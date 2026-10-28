#!/usr/bin/env python3
"""Reproducible LFM3-Audio-2B *baseline* call-workload smoke tests.
Do not mislabel this script as a GPTQ benchmark: it loads the BF16 checkpoint.
Official upstream model revision is pinned in README/manifest. No live calls.
"""
from __future__ import annotations
import argparse, json, time, traceback
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import librosa
from transformers import AutoModelForMultimodalLM, AutoProcessor

CASES = {
    "en": ("American English", "Hello, I would like to book a dental appointment for Friday morning."),
    "es": ("Spanish", "Hola, quisiera reservar una cita para el viernes por la mañana."),
}

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def elapsed_generate(model, inputs, **kwargs):
    sync()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, return_dict_in_generate=True, **kwargs)
    sync()
    return out, time.perf_counter() - t0

def phone_simulation(wav, sr, outpath):
    """A 24k WAV -> 8k mono, 8-bit mu-law companding, -> 16k encoder WAV.
    This is a synthetic phone-band codec path, NOT a real PSTN roundtrip.
    """
    x = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=8000)
    x = np.clip(x, -1.0, 1.0)
    mu = 255.0
    companded = np.sign(x) * np.log1p(mu*np.abs(x)) / np.log1p(mu)
    companded = np.round((companded + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    signed = companded.astype(np.float32) / 127.5 - 1.0
    restored = np.sign(signed) * np.expm1(np.abs(signed) * np.log1p(mu)) / mu
    upsampled = librosa.resample(restored, orig_sr=8000, target_sr=16000)
    sf.write(outpath, upsampled, 16000)
    return len(upsampled)/16000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model")
    parser.add_argument("--overlay", default=None, help="Manifest path for optional validated GPTQ-Pro native INT4 overlay")
    parser.add_argument("--allow-partial-overlay", action="store_true", help="Allow pilot partial layer overlay for debugging only")
    parser.add_argument("--outdir", default="results/bf16")
    parser.add_argument("--tasks", default="tts,asr,voice,tool")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    a = parser.parse_args()
    torch.manual_seed(42)
    outdir=Path(a.outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_dir=Path(a.model).resolve()
    tasks={x.strip() for x in a.tasks.split(",")}
    print("Loading model:",str(model_dir),"GPU:",device,flush=True)
    t0=time.perf_counter()
    model=AutoModelForMultimodalLM.from_pretrained(str(model_dir),dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,device_map=device,trust_remote_code=True).eval()
    processor=AutoProcessor.from_pretrained(str(model_dir),trust_remote_code=True)
    codec=processor.audio_tokenizer.to(device=device).eval()
    overlay_info=None
    if a.overlay:
        from overlay_runtime import apply_overlay
        overlay_info=apply_overlay(model,a.overlay,device=device,require_complete=not a.allow_partial_overlay)
    sync()
    report={"model":str(model_dir),"precision":"GPTQ-Pro-INT4-g64" if overlay_info else "BF16","device":device,"load_s":round(time.perf_counter()-t0,3),"tasks":{},"environment":{"torch":torch.__version__}}
    if overlay_info:
        report["overlay_manifest"]=str(Path(a.overlay).resolve())
        report["overlay_layers"]=len(overlay_info["parts"])
        report["quantized_projections"]=sum(len(p["modules"]) for p in overlay_info["parts"])
    if torch.cuda.is_available():
        report["steady_model_gpu_GiB"]=round(torch.cuda.memory_allocated()/2**30,3)
    print("Model loaded in",report["load_s"],"s",flush=True)
    for language,(locale,phrase) in CASES.items():
        source=outdir/f"{language}_tts.wav"
        phone=outdir/f"{language}_phone16k.wav"
        if "tts" in tasks:
            print("TTS",language,flush=True)
            convo=[{"role":"system","content":f"Perform TTS in {locale} using a female voice."},
                   {"role":"user","content":phrase},
                   {"role":"assistant","content":"<|audio_start|>"}]
            inp=processor.apply_chat_template(convo,continue_final_message=True,tokenize=True,return_dict=True,return_tensors="pt").to(device)
            try:
                out, gen_s = elapsed_generate(model, inp, max_new_tokens=a.max_new_tokens, audio_temperature=1.0, audio_top_k=128,audio_repetition_penalty=1.1,audio_repetition_window=4096)
                sync()
                tdec=time.perf_counter()
                with torch.inference_mode():
                    wav=codec.decode(out.audio_codes[:1,:,:out.audio_lengths[0]]).audio_values[0].float().cpu().numpy()
                sync()
                dec_s=time.perf_counter()-tdec
                sf.write(source,wav,24000)
                dur=phone_simulation(wav,24000,phone)
                report["tasks"][f"tts_{language}"]={"synthesis_generation_s":round(gen_s,3),"codec_decode_s":round(dec_s,3),"duration_s":round(dur,3),"generation_rtf":round(gen_s/max(dur,1e-5),3),"reference_text":phrase,"file":str(source),"phone_file":str(phone)}
            except Exception as exc:
                report["tasks"][f"tts_{language}"]={"error":repr(exc),"trace":traceback.format_exc()[-1200:]}
                print("TTS ERR",language,repr(exc),flush=True)
        if "asr" in tasks and phone.exists():
            print("ASR",language,flush=True)
            convo=[{"role":"system","content":f"Perform ASR in {'English' if language=='en' else 'Spanish'}."},
                   {"role":"user","content":[{"type":"audio","audio":str(phone)}]}]
            try:
                inp=processor.apply_chat_template(convo,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt").to(device)
                out,sec=elapsed_generate(model,inp,max_new_tokens=128,suppress_mixed=True,suppress_audio_out=True)
                txt=processor.decode(out.sequences[0,inp["input_ids"].shape[1]:],skip_special_tokens=True)
                report["tasks"][f"asr_{language}"]={"generate_s":round(sec,3),"transcript":txt,"reference_text":phrase,"phone_file":str(phone)}
                print("ASR:",txt,flush=True)
            except Exception as exc:
                report["tasks"][f"asr_{language}"]={"error":repr(exc),"trace":traceback.format_exc()[-1200:]}
                print("ASR ERR",language,repr(exc),flush=True)
        if "voice" in tasks and phone.exists():
            print("VOICE",language,flush=True)
            convo=[{"role":"system","content":"Respond with interleaved text and audio. You are a friendly NZ clinic receptionist. Speak in the caller's language, reply with one short sentence, and don't confirm availability without checking."},
                   {"role":"user","content":[{"type":"audio","audio":str(phone)}]},
                   {"role":"assistant","content":"<|mixed_start|>"}]
            try:
                inp=processor.apply_chat_template(convo,continue_final_message=True,tokenize=True,return_dict=True,return_tensors="pt").to(device)
                out,sec=elapsed_generate(model,inp,max_new_tokens=a.max_new_tokens,audio_temperature=1.0,audio_top_k=128,audio_repetition_penalty=1.1,audio_repetition_window=4096)
                txt=processor.decode(out.sequences[0,inp["input_ids"].shape[1]:],skip_special_tokens=True)
                with torch.inference_mode():
                    wav=codec.decode(out.audio_codes[:1,:,:out.audio_lengths[0]]).audio_values[0].float().cpu().numpy()
                sf.write(outdir/f"{language}_reply.wav",wav,24000)
                report["tasks"][f"voice_{language}"]={"generate_s":round(sec,3),"text":txt,"duration_s":round(len(wav)/24000,3),"reply_file":str(outdir/f"{language}_reply.wav")}
                print("VOICE TEXT",txt,flush=True)
            except Exception as exc:
                report["tasks"][f"voice_{language}"]={"error":repr(exc),"trace":traceback.format_exc()[-1200:]}
                print("VOICE ERR",language,repr(exc),flush=True)
    if "tool" in tasks:
        print("TOOL CALL",flush=True)
        tools=[{"type":"function","function":{"name":"check_appointment_availability","description":"Check whether a dental appointment is available at a specific date and time. Never invent availability.","parameters":{"type":"object","properties":{"day":{"type":"string","description":"Requested day"},"time":{"type":"string","description":"Requested time"}},"required":["day","time"]}}}]
        audio=outdir/"en_phone16k.wav"
        if audio.exists():
            convo=[{"role":"system","content":"Respond with function calls, and interleaved text and audio.\n\nList of tools:"},
                   {"role":"user","content":[{"type":"audio","audio":str(audio)}]}]
            try:
                inp=processor.apply_chat_template(convo,tools=tools,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt").to(device)
                out,sec=elapsed_generate(model,inp,max_new_tokens=256,audio_temperature=1.0,audio_top_k=128,audio_repetition_penalty=1.1,audio_repetition_window=4096)
                txt=processor.decode(out.sequences[0,inp["input_ids"].shape[1]:],skip_special_tokens=False)
                report["tasks"]["tool_en"]={"generate_s":round(sec,3),"raw_text":txt,"has_function_call":"<function=check_appointment_availability>" in txt}
                print("TOOL:",txt[:1200],flush=True)
            except Exception as exc:
                report["tasks"]["tool_en"]={"error":repr(exc),"trace":traceback.format_exc()[-1200:]}
                print("TOOL ERR",repr(exc),flush=True)
    if torch.cuda.is_available(): report["peak_gpu_GiB"]=round(torch.cuda.max_memory_allocated()/2**30,3)
    (outdir/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("REPORT",json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
