#!/usr/bin/env python3
"""Native GPTQ-Pro INT4 g64 overlay for the LFM3-Audio-2B *language backbone*.

Preserves: all acoustic components, codec, embedding, lm_head, normalizations,
attention gates and convolution kernels. For development and reproducible local
comparison. No out-of-the-box GPTQModel multimodal model registration is implied.
The original BF16 base checkpoint is required to restore the overlay.
"""
from __future__ import annotations
import argparse, copy, json, os, time, traceback
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForMultimodalLM, AutoProcessor

REV="9d49701c372d6011d900f6abb40c6b296fb706dd"
QUANT_PREFIX="model.language_model.layers."
PROFILE_VERSION=1
PROJECT_ROOT=Path(__file__).resolve().parent

def named_target_modules(layer):
    """Explicit allow-list for reproducible layers, never the audio towers."""
    names=[]
    if hasattr(layer,"conv"):
        names.extend(["conv.in_proj","conv.out_proj"])
    if hasattr(layer,"self_attn"):
        names.extend(["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj"])
    names.extend(["feed_forward.w1","feed_forward.w3","feed_forward.w2"])
    result=[]
    for name in names:
        obj=layer.get_submodule(name)
        if not isinstance(obj,nn.Linear):
            raise TypeError(f"{name}: expected nn.Linear, got {type(obj).__name__}")
        if obj.bias is not None:
            raise RuntimeError(f"{name}: biased linear not yet supported by overlay")
        if obj.in_features%64 or obj.out_features%32:
            raise RuntimeError(f"{name}: incompatible GPTQ-Pro native shape {obj.in_features}x{obj.out_features}")
        result.append(name)
    return result

def set_relative(root,relative,new_module):
    parent,field=relative.rsplit(".",1)
    setattr(root.get_submodule(parent),field,new_module)

def write_json_atomic(path,data):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    temp.replace(path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",default=str(PROJECT_ROOT/"model"))
    ap.add_argument("--calibration",default=str(PROJECT_ROOT/"calibration"/"manifest.json"))
    ap.add_argument("--out",default=str(PROJECT_ROOT/"overlays"/"g64"))
    ap.add_argument("--max-layers",type=int,default=24)
    ap.add_argument("--max-rows",type=int,default=2400)
    ap.add_argument("--min-rows",type=int,default=300)
    ap.add_argument("--limit-calibration",type=int,default=16)
    a=ap.parse_args()
    if a.max_layers < 1 or a.max_layers>24: ap.error("max-layers must be in [1,24]")
    torch.manual_seed(20261029)
    base=Path(a.base).resolve()
    dest=Path(a.out).resolve()
    dest.mkdir(parents=True,exist_ok=True)
    samples=json.loads(Path(a.calibration).read_text())
    samples=samples[:a.limit_calibration]
    if len(samples)<4: raise RuntimeError("Need >=4 distinct calibrations")
    source_sha_file=base/".cache"/"huggingface"/"download"/"config.json.metadata"
    manifest_file=dest/"manifest.json"
    if manifest_file.exists():
        manifest=json.loads(manifest_file.read_text())
        if (manifest["revision"]!=REV or manifest["group_size"]!=64 or manifest["calibration_samples"]!=len(samples)):
            raise RuntimeError("Overlay manifest mismatch: use an empty outdir to prevent mixing configurations")
    else:
        manifest={"version":PROFILE_VERSION,"revision":REV,"source_repository":"LiquidAI/LFM3-Audio-2B",
                  "base_model":str(base),"format":"gptq-pro-packed-int4-overlay",
                  "quantized_subtree":"model.language_model.layers",
                  "keep_bf16":["model.audio_encoder","model.audio_projector","model.depth_decoder","model.codebook_embedding","model.language_model.embed_tokens","lm_head","model.language_model.embedding_norm","audio_tokenizer"],
                  "exclude_names":["*.self_attn.gate_proj","*.conv.conv","*.norm*"],
                  "bits":4,"group_size":64,"sym":True,"desc_act":False,"calibration_samples":len(samples),
                  "calibration_type":"synthetic TTS samples at 8k mu-law equivalent (not live PSTN calls)",
                  "calibration_files":[s["wav"] for s in samples],
                  "parts":[],"created":"2026-10-29","full_model_certified":False}
        write_json_atomic(manifest_file,manifest)
    print("LOAD BF16 BASE",flush=True)
    model=AutoModelForMultimodalLM.from_pretrained(str(base),dtype=torch.bfloat16,device_map="cuda:0",trust_remote_code=True).eval()
    processor=AutoProcessor.from_pretrained(str(base),trust_remote_code=True)
    from gptqmodel import QuantizeConfig
    from gptqmodel.quantization.gptq import GPTQ
    from gptqmodel.nn_modules.qlinear.gptq_pro import GptqProQuantLinear
    from overlay_runtime import apply_overlay

    # Resumption restores ALL earlier packed layers so activation statistics are
    # collected under the same evolving quantized backbone.
    if manifest["parts"]:
        apply_overlay(model,manifest_file,device="cuda:0",require_complete=False)
    layers=model.model.language_model.layers
    preexisting={int(p["index"]) for p in manifest["parts"]}
    for i in range(a.max_layers):
        if i in preexisting:
            print(f"SKIP EXISTING layer {i}",flush=True)
            continue
        # Prevent accidentally capturing layer >= i while previous layer is
        # incomplete; prefix must be contiguous to keep resume deterministic.
        assert preexisting==set(range(i)),f"Non-contiguous overlay parts before {i}: {preexisting}"
        layer=layers[i]
        names=named_target_modules(layer)
        captures={n:[] for n in names}
        handles=[]
        for name in names:
            def hook(_module,args,_name=name):
                arr=args[0].detach().reshape(-1,args[0].shape[-1]).to(device="cpu",dtype=torch.bfloat16)
                arr=arr.contiguous()
                if sum(z.shape[0] for z in captures[_name])<a.max_rows:
                    captures[_name].append(arr)
            handles.append(layer.get_submodule(name).register_forward_pre_hook(hook))
        tcap=time.perf_counter()
        try:
            # Mix speech-to-text, mixed spoken response, and TTS prompt prefixes;
            # this captures actual acoustic-prefix states, not just raw tokens.
            with torch.inference_mode():
                for k,row in enumerate(samples):
                    lang=row["language"]
                    mode="asr" if (k%2==0) else "voice"
                    if mode=="asr":
                        conversation=[{"role":"system","content":f"Perform ASR in {'English' if lang=='en' else 'Spanish'}."},
                                      {"role":"user","content":[{"type":"audio","audio":row["wav"]}]}]
                        kw={"add_generation_prompt":True}
                    else:
                        conversation=[{"role":"system","content":"Respond with interleaved text and audio. You are a friendly NZ business receptionist. Speak in the caller's language and check availability before confirming bookings."},
                                     {"role":"user","content":[{"type":"audio","audio":row["wav"]}]},
                                     {"role":"assistant","content":"<|mixed_start|>"}]
                        kw={"continue_final_message":True}
                    inp=processor.apply_chat_template(conversation,tokenize=True,return_dict=True,return_tensors="pt",**kw).to("cuda:0")
                    _=model(**inp,use_cache=False)
                # Add TTS-related prefixes to stabilize generated-speech token paths.
                for row in samples[:min(8,len(samples))]:
                    language="Spanish" if row["language"]=="es" else "American English"
                    conversation=[{"role":"system","content":f"Perform TTS in {language} using a female voice."},
                                  {"role":"user","content":row["text"]},
                                  {"role":"assistant","content":"<|audio_start|>"}]
                    inp=processor.apply_chat_template(conversation,continue_final_message=True,tokenize=True,return_dict=True,return_tensors="pt").to("cuda:0")
                    _=model(**inp,use_cache=False)
        finally:
            for h in handles: h.remove()
        cap_s=time.perf_counter()-tcap
        rows=min([sum(x.shape[0] for x in captures[name]) for name in names])
        print(f"LAYER {i}/24 captured {rows} activation rows over {len(samples)} audio calls; capture={cap_s:.2f}s",flush=True)
        if rows<a.min_rows:
            raise RuntimeError(f"Layer {i} only {rows} rows < {a.min_rows} minimum. Cannot publish an uncalibrated artifact.")

        payload={}
        entries=[]
        layer_start=time.perf_counter()
        for name in names:
            target=layer.get_submodule(name)
            num_in=target.in_features
            x=torch.cat(captures[name],dim=0)[:a.max_rows].to(device="cuda:0",dtype=torch.bfloat16,non_blocking=True)
            del captures[name][:]
            qcfg=QuantizeConfig.quality_4bit()
            g=GPTQ(target,qcfg)
            g.quantizer.configure(perchannel=True)
            with torch.inference_mode():
                # y is not used by GPTQ Hessian accumulation, but send the real projection output
                # so this follows the exact same GPTQ add_batch signature as production.
                g.add_batch(x,target(x))
                qweight,scales,zeros,gidx,qt,loss,damp,nsamp=g.quantize()
            xval=x[-min(32,len(x)):].contiguous()
            with torch.inference_mode():
                y_ref=target(xval)
                y_gptq=F.linear(xval,qweight.to("cuda:0",dtype=xval.dtype))
            rel=float((y_ref.float()-y_gptq.float()).square().mean()/y_ref.float().square().mean().clamp_min(1e-9))
            if not torch.isfinite(torch.tensor(rel)) or rel>0.05:
                raise RuntimeError(f"Poor GPTQ accuracy in layer {i}.{name}: relative mse {rel}")
            corrected=nn.Linear(target.in_features,target.out_features,bias=False,dtype=torch.bfloat16)
            corrected.weight.data.copy_(qweight.detach().to(device="cpu",dtype=torch.bfloat16))
            native=GptqProQuantLinear(bits=4,group_size=64,sym=True,desc_act=False,
                          in_features=target.in_features,out_features=target.out_features,bias=False,dtype=torch.bfloat16)
            native.pack(corrected,scales.detach().cpu(),zeros.detach().cpu(),gidx.detach().cpu(),workers=1)
            state=native.state_dict()
            prefix=f"{QUANT_PREFIX}{i}.{name}."
            for key,value in state.items():
                if not isinstance(value,torch.Tensor):
                    raise RuntimeError(f"Unsupported GPTQ-Pro state key {key}")
                payload[prefix+key]=value.detach().cpu().contiguous()
            native=native.to("cuda:0")
            native.post_init()
            with torch.inference_mode():
                y_native=native(xval)
            native_rel=float((y_native.float()-y_gptq.float()).square().mean()/y_gptq.float().square().mean().clamp_min(1e-9))
            if not torch.isfinite(torch.tensor(native_rel)) or native_rel>0.001:
                raise RuntimeError(f"Native GPTQ-Pro parity failure layer {i}.{name}: rel mse {native_rel}")
            set_relative(layer,name,native)
            entries.append({"name":f"{QUANT_PREFIX}{i}.{name}","in_features":int(num_in),
                            "out_features":int(target.out_features),"captured_rows":int(x.shape[0]),
                            "gptq_relative_mse":rel,"kernel_relative_mse":native_rel,
                            "solver_s":round(float(qt),3),"loss":str(loss),"damp":float(damp)})
            del x,xval,qweight,scales,zeros,gidx,corrected,g,native
            print(f"  {name:<24} rel_mse={rel:.3e} kernel={native_rel:.3e}  solver={qt:.2f}s",flush=True)
        fname=f"layer_{i:02d}.safetensors"
        fpath=dest/fname
        tmppath=dest/(fname+".tmp")
        save_file(payload,str(tmppath))
        tmppath.replace(fpath)
        rows_part={"index":i,"file":fname,"capture_s":round(cap_s,3),"quant_s":round(time.perf_counter()-layer_start,3),
                   "modules":entries,"bytes":fpath.stat().st_size}
        manifest["parts"].append(rows_part)
        manifest["full_model_certified"]=False
        write_json_atomic(manifest_file,manifest)
        preexisting.add(i)
        print(f"SAVED LAYER {i}; {len(entries)} projections, {round(fpath.stat().st_size/2**20,1)} MiB",flush=True)
        del payload,entries,captures
        torch.cuda.empty_cache()
    manifest["completed_layers"]=len(manifest["parts"])
    manifest["completed_quantized_projections"]=sum(len(p["modules"]) for p in manifest["parts"])
    manifest["complete_backbone"]=len(manifest["parts"])==len(layers)
    manifest["full_model_certified"]=False # User acceptance evaluation still required
    write_json_atomic(manifest_file,manifest)
    print("DONE",json.dumps({"layers":manifest["completed_layers"],"projections":manifest["completed_quantized_projections"],"complete_backbone":manifest["complete_backbone"]}),flush=True)

if __name__=="__main__": main()
