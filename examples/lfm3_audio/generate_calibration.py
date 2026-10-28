#!/usr/bin/env python3
"""Synthetic bilingual phone-band calibration audio generated locally with LFM3-Audio-2B.
Calibration speech is synthetic, and must not be represented as a real-call evaluation set.
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch, soundfile as sf
from transformers import AutoModelForMultimodalLM, AutoProcessor
from bench_lfm3_call import phone_simulation

PHRASES=[
 ("en","Hi, can I book an appointment for Monday at nine thirty in the morning?"),
 ("en","Do you have anything available next Tuesday afternoon after three?"),
 ("en","My name is Samuel Thomson. I am calling to reschedule my appointment."),
 ("en","Please tell me your opening hours and where your business is located."),
 ("en","I have already booked a visit. Could you check if it is confirmed?"),
 ("en","Can I speak with a person, please? This is quite urgent."),
 ("en","I would like to cancel tomorrow's appointment and request a callback."),
 ("en","Could you repeat the price, including GST, and spell your email address?"),
 ("es","Hola, necesito pedir un turno para el lunes a las nueve y media."),
 ("es","¿Tienen disponibilidad el martes después de las tres de la tarde?"),
 ("es","Me llamo Valentina Fernández. Quisiera cambiar la fecha de mi cita."),
 ("es","¿Me pueden decir el horario de atención y la dirección del local?"),
 ("es","Tengo una reserva para mañana y necesito que me la confirmen."),
 ("es","Quiero hablar con una persona por favor, es bastante urgente."),
 ("es","Necesito cancelar el turno de mañana y pedir una devolución de llamada."),
 ("es","¿Podrías repetir el precio total con impuestos y tu correo electrónico?"),
]
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--model",default="model")
 ap.add_argument("--outdir",default="calibration")
 ap.add_argument("--count",type=int,default=16)
 a=ap.parse_args()
 torch.manual_seed(20261029)
 path=Path(a.outdir); path.mkdir(parents=True,exist_ok=True)
 model=AutoModelForMultimodalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map="cuda:0",trust_remote_code=True).eval()
 processor=AutoProcessor.from_pretrained(a.model,trust_remote_code=True)
 codec=processor.audio_tokenizer.to("cuda:0").eval()
 rows=[]
 for idx,(lang,phrase) in enumerate(PHRASES[:a.count]):
  fname=path/f"{idx:02}_{lang}_phone16k.wav"
  if fname.exists() and fname.stat().st_size>16000:
   rows.append({"language":lang,"text":phrase,"wav":str(fname),"reused":True})
   continue
  convo=[{"role":"system","content":"Perform TTS in "+("American English" if lang=="en" else "Spanish")+" using a female voice."},
         {"role":"user","content":phrase},
         {"role":"assistant","content":"<|audio_start|>"}]
  inp=processor.apply_chat_template(convo,continue_final_message=True,tokenize=True,return_dict=True,return_tensors="pt").to("cuda:0")
  ts=time.perf_counter()
  with torch.inference_mode():
   out=model.generate(**inp,max_new_tokens=480,return_dict_in_generate=True,audio_temperature=1.0,audio_top_k=128,audio_repetition_penalty=1.1,audio_repetition_window=4096)
   wav=codec.decode(out.audio_codes[:1,:,:out.audio_lengths[0]]).audio_values[0].float().cpu().numpy()
  duration=phone_simulation(wav,24000,fname)
  rows.append({"language":lang,"text":phrase,"wav":str(fname),"duration_s":round(duration,2),"generation_s":round(time.perf_counter()-ts,2),"synthetic":True})
  print(f"{idx+1:02d}/{a.count} {lang} {duration:.2f}s audio generated",flush=True)
 (path/"manifest.json").write_text(json.dumps(rows,indent=2,ensure_ascii=False))
 print("CALIBRATION_SET_READY",len(rows),flush=True)
if __name__=="__main__":main()
