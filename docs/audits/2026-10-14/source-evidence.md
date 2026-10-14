# Numbered source evidence

Private paths are replaced with placeholders. SHA-256 values identify the original inspected files, not the sanitized excerpts below. These sources include campaign-local files absent from main.

Generated 2026-10-14T20:07:32.428627+13:00

## job/run_quant.py
Path: `${CAMPAIGN_ROOT}/run_quant.py`
SHA-256: `ab6e6ab180de0c13b50e241537f0268d7b75340ae9732e295b33d408f06c42e9`

```text
31: def source_check():
32:     cfg=json.loads((SOURCE/"config.json").read_text())
33:     tx=cfg["text_config"]
34:     assert cfg["model_type"]=="qwen3_5_moe"
35:     assert tx["num_hidden_layers"]==40 and tx["num_experts"]==256 and tx["num_experts_per_tok"]==8
36:     assert tx["mtp_num_hidden_layers"]==1
37:     ix=json.loads((SOURCE/"model.safetensors.index.json").read_text())
38:     missing=[name for name in set(ix["weight_map"].values()) if not (SOURCE/name).is_file()]
39:     assert not missing,missing
40:     assert (SOURCE/"model-auxiliary.safetensors").stat().st_size>1_000_000_000
41:     counts={"vision":sum("visual" in k for k in ix["weight_map"]),
42:             "mtp":sum(k.startswith("mtp.") for k in ix["weight_map"]),
43:             "shards":len(set(ix["weight_map"].values())),
44:             "tensors":len(ix["weight_map"])}
45:     assert counts["mtp"]==19 and counts["vision"]>200,counts
46:     return counts
```

```text
62: def watchdog(out, stop):
63:     import psutil
64:     while not stop.wait(10):
65:         disk=shutil.disk_usage(ROOT).free
66:         ram=psutil.virtual_memory().available
67:         atomic(out/"resources.json",{"disk_free_gib":round(disk/GiB,2),
68:                                     "ram_available_gib":round(ram/GiB,2),
69:                                     "phase":(out/"phase.txt").read_text() if (out/"phase.txt").exists() else "unknown"})
70:         if disk < 20*GiB or ram < 16*GiB:
71:             atomic(out/"fatal-resource-pressure.json",{"disk_free_gib":disk/GiB,"ram_available_gib":ram/GiB})
72:             emit("resource_abort",free_disk_gib=round(disk/GiB,2),ram_avail_gib=round(ram/GiB,2))
73:             os._exit(75)
```

```text
92:     phase=args.phase
93:     if phase=="full":
94:         pilot=ROOT/"pilot"/"phase.txt"
95:         if not pilot.is_file() or pilot.read_text().strip()!="quantized":
96:             raise RuntimeError("Full quantization gated: successful GPU pilot required")
97:         # The pilot's source-weight and calibration spills are temporary.
98:         # It has already exited before gpuq admits this GPU-exclusive job.
99:         for scratch_name in ("weight-offload","native-spill","layer-spill"):
100:             scratch=ROOT/"pilot"/scratch_name
101:             if scratch.is_dir() and not scratch.is_symlink():
102:                 emit("cleanup_finished_pilot_scratch",directory=str(scratch))
103:                 shutil.rmtree(scratch)
104:     run_dir=ROOT/("pilot" if phase=="pilot" else "full")
105:     run_dir.mkdir(exist_ok=True)
106:     os.environ.setdefault("GPTQ_HQ_NATIVE_SPILL",str(run_dir/"native-spill"))
107:     os.environ.setdefault("GPTQ_HQ_LAYER_SPILL",str(run_dir/"layer-spill"))
108:     os.environ.setdefault("GPTQMODEL_EXT_BUILD",str(ROOT/"extensions"))
109:     os.environ.setdefault("MAX_JOBS","2")
110:     if phase=="full" and (run_dir/"started").exists():
111:         raise RuntimeError(f"Full run directory already used: {run_dir}. Refusing overwrite.")
112:     (run_dir/"started").write_text(str(time.time()))
113:     (run_dir/"phase.txt").write_text("preflight")
114:     initial_check()
115:     source_stats=source_check()
116:     policy=verify()
117:     rules=dynamic_rules([0] if phase=="pilot" else None)
118:     q=QuantizeConfig.quality_4bit(group_size=64)
119:     q.device=torch.device("cuda:0")
120:     q.calibration_data_device="cpu"
121:     q.offload_to_disk=True
122:     q.offload_to_disk_path=str(run_dir/"weight-offload")
123:     q.auto_forward_data_parallel=False
124:     q.wait_for_submodule_finalizers=True
125:     q.fallback=None
126:     q.dynamic=rules
127:
128:     # Prove that GPTQ-Pro's own PCRE matcher sees the intended overrides.
129:     ex=lambda layer, expert, proj: f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}"
130:     assert dynamic_get(q.dynamic,ex(0,0,"gate_proj"),"group_size",64)==64
131:     assert dynamic_get(q.dynamic,ex(0,0,"down_proj"),"group_size",64)==32
132:     if phase=="full":
133:         assert dynamic_get(q.dynamic,ex(10,0,"down_proj")) is False
134:         assert dynamic_get(q.dynamic,ex(10,0,"up_proj"),"group_size",64)==32
135:     else:
136:         assert dynamic_get(q.dynamic,ex(10,0,"up_proj")) is False
137:     manifest={"phase":phase,"source":str(SOURCE),"source_count":source_stats,
138:               "calibration":str(CALIB),
139:               "calibration_sha256":hashlib.sha256(CALIB.read_bytes()).hexdigest(),
140:               "precision":"native-W4A16+unquantized-sensitive-components",
141:               "group_size":q.group_size,"bits":q.bits,"dynamic":q.dynamic,
142:               "policy":policy,
143:               "torch":torch.__version__,"cuda":torch.version.cuda,
144:               "branch":"ud-inspired-no-unverified-W8-support"}
145:     atomic(run_dir/"manifest.json",manifest)
```

```text
175:     stop=threading.Event()
176:     watcher=threading.Thread(target=watchdog,args=(run_dir,stop),daemon=True)
177:     watcher.start()
178:     try:
179:         (run_dir/"phase.txt").write_text("quantizing")
180:         started=time.time()
181:         result=model.quantize(calibration,batch_size=1,calibration_sort=None,calibration_concat_size=None)
182:         emit("quantize_returned",seconds=int(time.time()-started),result_type=str(type(result)))
183:         atomic(run_dir/"quantize-result.json",result)
184:         (run_dir/"phase.txt").write_text("quantized")
185:         if phase=="pilot":
186:             emit("pilot_completed",saved=False,note="Pilot intentionally does not serialize 70 GB partial checkpoint.")
187:             return
188:         (run_dir/"phase.txt").write_text("saving")
189:         output=ROOT/"Qwen3.6-35B-A3B-Heretic-UD-inspired-GPTQPro-W4"
190:         if output.exists() and any(output.iterdir()):
191:             raise RuntimeError("Output exists and is nonempty; refusing overwrite")
```

```text
208:         actual_files=list(output.glob("*.safetensors"))
209:         if not actual_files:raise RuntimeError("No safetensors outputs")
210:         file_size=sum(f.stat().st_size for f in actual_files)
211:         from safetensors import safe_open
212:         tensor_names=[]
213:         for shard in actual_files:
214:             with safe_open(str(shard),framework="pt",device="cpu") as safe:
215:                 tensor_names.extend(safe.keys())
216:         check={"mtp":sum(k.startswith("mtp.") for k in tensor_names),
217:                "vision":sum("visual" in k for k in tensor_names),
218:                "qweights":sum(k.endswith("qweight") for k in tensor_names),
219:                "tensor_count":len(tensor_names),
220:                "total_gib":file_size/GiB}
221:         atomic(run_dir/"serialized-tensor-audit.json",check)
222:         emit("serialized_audit",**check)
223:         if check["mtp"]!=19 or check["vision"]<200 or check["qweights"]<1:
224:             raise RuntimeError("Serialization did not preserve 19 MTP tensors, vision, and quantized expert weights")
225:         emit("saved",files=len(actual_files),total_gib=round(file_size/GiB,2))
226:         atomic(run_dir/"artifact.json",{"checkpoint":str(output),"safetensors_files":len(actual_files),"size_gib":file_size/GiB,"status":"needs_load_and_quality_verification"})
227:         (run_dir/"phase.txt").write_text("exported_unverified")
228:     finally:
229:         stop.set()
230:         watcher.join(timeout=2)
231:     emit("run_finished",phase=phase)
232:
233: def main():
234:     parser=argparse.ArgumentParser()
235:     parser.add_argument("--phase",choices=["pilot","full"],required=True)
236:     args=parser.parse_args()
237:     try:run(args)
238:     except BaseException as exc:
239:         out=ROOT/args.phase
240:         out.mkdir(exist_ok=True)
241:         atomic(out/"failure.json",{"exception":str(exc),"traceback":traceback.format_exc()})
242:         (out/"phase.txt").write_text("failed")
243:         emit("failure",error=str(exc),phase=args.phase)
244:         raise
```

## job/verify_after.py
Path: `${CAMPAIGN_ROOT}/verify_after.py`
SHA-256: `7789ce7fbb858c95c57c6bf6e602a9b116323298c6fbd551dfcdb7cacdcf54ab`

```text
24: def main():
25:  full=ROOT/"full"
26:  phase=(full/"phase.txt").read_text().strip() if (full/"phase.txt").is_file() else "missing"
27:  if phase!="exported_unverified":
28:      save("status.json",{"status":"not_ready","full_phase":phase,"note":"No completed checkpoint; runtime was NOT attempted."})
29:      print("Model not exported/verified:",phase,flush=True)
30:      return 4
31:  if not (full/"artifact.json").is_file():
32:      raise RuntimeError("No artifact report in full job")
33:  if not (OUTPUT/"config.json").is_file():
34:      raise RuntimeError("Missing quantized model config.json")
35:  if not (OUTPUT/"quantize_config.json").is_file():
36:      raise RuntimeError("Missing GPTQ quantize_config.json")
37:  files=list(OUTPUT.glob("*.safetensors"))
38:  if not files:raise RuntimeError("Missing quantized safetensors")
39:  index_path=OUTPUT/"model.safetensors.index.json"
40:  if index_path.exists():
41:      ix=json.loads(index_path.read_text())
42:      referenced=set(ix["weight_map"].values())
43:      if not referenced.issubset(set(x.name for x in files)):
44:          raise RuntimeError("Index references missing safetensors shard")
45:  qcfg=json.loads((OUTPUT/"quantize_config.json").read_text())
46:  all_names=[]
47:  for f in files:
48:      with safe_open(str(f),framework="pt",device="cpu") as sf:
49:          all_names.extend(sf.keys())
50:  counters={"mtp":sum(x.startswith("mtp.") for x in all_names),
51:            "vision":sum("visual" in x for x in all_names),
52:            "qweights":sum(x.endswith("qweight") for x in all_names),
53:            "scales":sum(x.endswith("scales") for x in all_names),
54:            "tensors":len(all_names)}
55:  mixed_groups=any(isinstance(v,dict) and v.get("group_size")==32 for v in (qcfg.get("dynamic") or {}).values())
56:  summary={"status":"structure_checked","checkpoint":str(OUTPUT),"shards":len(files),
57:           "bytes":sum(p.stat().st_size for p in files),"quant_base_bits":qcfg.get("bits"),
58:           "quant_base_group":qcfg.get("group_size"),"mixed_group_size":mixed_groups,
59:           "tensors":counters,"index":index_path.exists(),
60:           "runtime_note":"vLLM fused MoE may not support g64 gate/up with g32 down; smoke execution is required."}
61:  save("structural.json",summary)
62:  if counters["mtp"]!=19 or counters["vision"]<200 or counters["qweights"]<1:
63:      raise RuntimeError("Missing expected MTP/vision/quantized matrices")
64:  if not mixed_groups:
65:      raise RuntimeError("Missing required g32 dynamic expert projection rules")
```

```text
81:  vllm_python="${VLLM_ENV}/bin/python"
82:  cmd=[vllm_python,"-u",str(ROOT/"vllm_probe.py"),str(OUTPUT)]
83:  print("vLLM real inference smoke:",cmd,flush=True)
84:  save("status.json",{"status":"running_inference","checkpoint":str(OUTPUT)})
85:  env=os.environ.copy()
86:  env.update({"VLLM_WORKER_MULTIPROC_METHOD":"spawn","TOKENIZERS_PARALLELISM":"false","OMP_NUM_THREADS":"4","HF_HUB_OFFLINE":"1"})
87:  try:
88:      result=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=900,check=False)
89:      (RESULT/"vllm.log").write_text(result.stdout[-2_000_000:])
90:      status="passed" if result.returncode==0 else "failed_runtime"
91:      save("status.json",{"status":status,"exit_code":result.returncode,
92:           "checkpoint":str(OUTPUT),"log":str(RESULT/"vllm.log"),
93:           "warning":"A failed vLLM probe does not imply quantization corrupted; fused MoE mixed g32/g64 may require a compatible runtime."})
94:      print("vLLM smoke:",status,flush=True)
95:      return result.returncode
96:  except subprocess.TimeoutExpired as exc:
97:      save("status.json",{"status":"runtime_timeout","error":str(exc)})
98:      return 124
99:
100: if __name__=="__main__":
101:  try:sys.exit(main())
102:  except BaseException as exc:
103:      save("status.json",{"status":"validation_failed","error":str(exc),"traceback":traceback.format_exc()})
104:      raise
```

## job/vllm_probe.py
Path: `${CAMPAIGN_ROOT}/vllm_probe.py`
SHA-256: `a046636318f23e737e930f055c16b97f78ebb0c8edf627815a47655af3ef5a18`

```text
9: path=Path(sys.argv[1])
10: try:
11:  from vllm import LLM,SamplingParams
12:  tell("runtime_start",model=str(path))
13:  llm=LLM(model=str(path),tokenizer=str(path),
14:          tensor_parallel_size=2,quantization="gptq_marlin",
15:          dtype="float16",max_model_len=2048,max_num_seqs=1,
16:          language_model_only=True,trust_remote_code=False,
17:          gpu_memory_utilization=0.83,enforce_eager=True,
18:          disable_log_stats=True,seed=42)
19:  requests=[
20:      "Answer in exactly one word: ready",
21:      "What is 2 + 2? Reply using a single digit."
22:  ]
23:  outputs=llm.generate(requests,SamplingParams(temperature=0.0,max_tokens=96))
24:  answers=[{"prompt":o.prompt,"output":o.outputs[0].text,"tokens":len(o.outputs[0].token_ids)} for o in outputs]
25:  tell("runtime_outputs",answers=answers)
26:  if len(answers)!=2 or not any("4" in a["output"] for a in answers):
27:      tell("quality_smoke_warning",note="No correct arithmetic marker in basic prompt response")
28:      sys.exit(3)
29:  tell("runtime_smoke_pass")
30: except BaseException as exc:
31:  tell("runtime_failure",error=str(exc),trace=traceback.format_exc()[-8000:])
32:  sys.exit(2)
```

## job/policy.py
Path: `${CAMPAIGN_ROOT}/policy.py`
SHA-256: `fc4e773e5671911cddb88d0be3c67c6675c0020c7ca2c9a8e3fae2db467b649b`

```text
14: def dynamic_rules(pilot_layers=None):
15:     d = {
16:         # Preserve always-used attention and linear-attention modules.
17:         rf"-:^{LAYER_ROOT}\.\d+\.(?:self_attn|linear_attn)\.": {},
18:         # Preserve shared expert on every layer, including its gating path.
19:         rf"-:^{LAYER_ROOT}\.\d+\.mlp\.shared_expert\.": {},
20:         # Higher precision for the eleventh decoder block's expert down.
21:         rf"-:^{LAYER_ROOT}\.10\.mlp\.experts\.\d+\.down_proj$": {},
22:         # Additional quantizer resolution on the sensitive block's gate/up.
23:         rf"+:^{LAYER_ROOT}\.10\.mlp\.experts\.\d+\.(?:gate_proj|up_proj)$": {"group_size": 32},
24:         # Extra scale granularity for every routed expert down projection.
25:         rf"+:^{LAYER_ROOT}\.\d+\.mlp\.experts\.\d+\.down_proj$": {"group_size": 32},
26:     }
27:     if pilot_layers is not None:
28:         permitted = {int(x) for x in pilot_layers}
29:         assert permitted <= set(range(LAYER_COUNT)) and permitted
30:         # Insert before all other rules so selected pilot does not quantize remaining layers.
31:         skipped = "|".join(str(i) for i in range(LAYER_COUNT) if i not in permitted)
32:         if skipped:
33:             d = {rf"-:^{LAYER_ROOT}\.(?:{skipped})\.": {}, **d}
34:     return d
```

```text
48: def verify():
49:     counts = {}
50:     examples=[]
51:     for layer in range(LAYER_COUNT):
52:         for expert in range(EXPERT_COUNT):
53:             for proj in ("gate_proj","up_proj","down_proj"):
54:                 name=f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}"
55:                 p=classify(name)
56:                 counts[p]=counts.get(p,0)+1
57:                 if layer in (0,SELECTIVE_LAYER,39) and expert == 0:
58:                     examples.append((name,p))
59:     assert counts=={"INT4-g64":19968,"INT4-g32":10496,"BF16-preserve":256},counts
60:     assert classify("model.language_model.layers.7.self_attn.q_proj")=="BF16-preserve"
61:     assert classify("model.language_model.layers.2.linear_attn.in_proj_qkv")=="BF16-preserve"
62:     assert classify("model.language_model.layers.12.mlp.shared_expert.up_proj")=="BF16-preserve"
63:     assert classify("model.language_model.layers.10.mlp.experts.55.down_proj")=="BF16-preserve"
64:     assert classify("model.language_model.layers.0.mlp.experts.55.down_proj")=="INT4-g32"
65:     assert classify("model.language_model.layers.11.mlp.experts.55.up_proj")=="INT4-g64"
66:     return {"counts":counts, "samples": examples}
```

## engine/gptqmodel/utils/activation_spill.py
Path: `${CAMPAIGN_ROOT}/engine/gptqmodel/utils/activation_spill.py`
SHA-256: `68c56194ce256f250d1d7d11d967b1ae4a0c76689e90db588147a2f6e0081be1`

```text
13: class DiskTensorQueue:
14:     lock = threading.RLock()
15:     stats = dict(written_bytes=0, read_bytes=0, live_bytes=0, peak_live_bytes=0,
16:                  appended=0, consumed=0, verified=0, queues=0)
17:
18:     def __init__(self, root, name):
19:         self.path = Path(root) / hashlib.sha256(name.encode()).hexdigest()
20:         self.path.mkdir(parents=True, exist_ok=False)
21:         self.entries = deque()
22:         self.appended = 0
23:         self.consumed = 0
24:         with self.lock:
25:             self.stats['queues'] += 1
26:
27:     @staticmethod
28:     def checksum(tensor):
29:         return hashlib.sha256(memoryview(tensor.contiguous().view(torch.uint8).numpy())).hexdigest()
30:
31:     def append(self, tensor, batch_index):
32:         if batch_index != self.appended:
33:             raise RuntimeError(f'Native input order mismatch: {batch_index} != {self.appended}')
34:         tensor = tensor.detach().to('cpu').contiguous()
35:         if not torch.isfinite(tensor).all():
36:             raise ValueError('Nonfinite native input')
37:         path = self.path / f'{batch_index:06d}.safetensors'
38:         checksum = self.checksum(tensor)
39:         save_file({'input': tensor}, str(path))
40:         size = path.stat().st_size
41:         self.entries.append((path, checksum, size, tensor.shape, tensor.dtype))
```

```text
114:     def clear(self):
115:         for path,_,size in self.entries:
116:             path.unlink(missing_ok=True)
117:             with self.lock:self.stats['live_bytes']-=size
118:         self.entries.clear()
119:         self.path.rmdir()
120:
121:     def __del__(self):
122:         if getattr(self,'entries',None):self.clear()
123:
124:     @classmethod
125:     def snapshot(cls):
126:         with cls.lock:return dict(cls.stats)
127:
128:
129: def layer_cache_list():
130:     root=os.getenv('GPTQ_HQ_LAYER_SPILL')
131:     return DiskLayerList(root) if root else []
```

## engine/gptqmodel/looper/native_processor.py
Path: `${CAMPAIGN_ROOT}/engine/gptqmodel/looper/native_processor.py`
SHA-256: `88a14760f2cc123bfb5f368300ba206c726d7df77f86906225c77a31fb0e9a46`

```text
60:     def set_calibration_dataset(self, calibration_dataset):
61:         """Rejects dataset replacement because capture setup is fixed at construction."""
62:
63:         raise NotImplementedError("NativeProcessor's calibration_dataset cannot be modified")
64:
65:     def preprocess(self, module: NamedModule):
66:         """Allocates the per-module cache used by the forward hook."""
67:
68:         spill_root = os.getenv('GPTQ_HQ_NATIVE_SPILL')
69:         self.native_inp_caches[module.name] = (
70:             DiskTensorQueue(spill_root, module.full_name) if spill_root else []
71:         )
72:
73:     def is_skipped(self, module: NamedModule) -> bool:
74:         """Reports that native input capture currently runs for every eligible module."""
75:
76:         # TODO: Add skipping certain modules
77:         return False
78:
79:     def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
80:         """Builds the forward hook that captures detached native inputs for a module."""
81:
82:         def tmp(module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
83:             """Copies the module input to the configured GPTAQ staging device."""
84:
85:             # gptq is mutable.
86:             inp = inp[0].detach()
87:
88:             if self.qcfg.gptaq is not None:
89:                 gptaq_device = self.qcfg.gptaq.device
90:             elif self.qcfg.foem is not None:
91:                 gptaq_device = self.qcfg.foem.device
92:             else:
93:                 gptaq_device = "auto"
94:             if gptaq_device == "auto":
95:                 target_device = DEVICE_1
96:             elif gptaq_device == "cpu":
97:                 # slower but >= 4x vram memory reduction
98:                 target_device = CPU
99:             elif isinstance(gptaq_device, str):
100:                 target_device = torch.device(gptaq_device)
101:             elif isinstance(gptaq_device, torch.device):
102:                 target_device = gptaq_device
103:             else:
104:                 target_device = DEVICE_1
105:
106:             cache = self.native_inp_caches[name]
107:             if isinstance(cache, DiskTensorQueue):
108:                 if target_device != CPU:
109:                     raise ValueError('Native spilling requires GPTAQ CPU staging')
110:                 cache.append(inp, self.current_batch_index())
111:             else:
112:                 cache += [inp.to(device=target_device)]
113:             del inp, out
114:
115:         return tmp
116:
117:     def process(
118:         self,
119:         module: NamedModule,
120:         device: torch.device = None,
121:         subset: Optional[Dict[str, NamedModule]] = None,
```

## engine/gptqmodel/quantization/gptq.py
Path: `${CAMPAIGN_ROOT}/engine/gptqmodel/quantization/gptq.py`
SHA-256: `bef545ae4b0383f16ff8e0a2396e7011bd335ac40db5ce891703e6208aac84ce`

```text
524:         try:
525:             xtx = self.compute_hessian_xtx(reshaped_inp).to(dtype=torch.float32)
526:         except RuntimeError as exc:
527:             if (
528:                 torch.device(inp_device).type == "cuda"
529:                 and "out of memory" in str(exc).lower()
530:             ):
531:                 log.warn(
532:                     "GPTQ module '%s' fell back to CPU Hessian accumulation due to GPU OOM during batch processing.",
533:                     getattr(self, "name", "<unknown>"),
534:                 )
535:                 reshaped_inp_cpu = reshaped_inp.to(device=torch.device("cpu"))
536:                 del reshaped_inp
537:                 if torch.cuda.is_available():
538:                     torch.cuda.empty_cache()
539:                 canonical_device = torch.device("cpu")
540:                 xtx = self.compute_hessian_xtx(reshaped_inp_cpu).to(dtype=torch.float32)
541:                 xtx = xtx.detach()
542:                 del reshaped_inp_cpu
543:             else:
544:                 del reshaped_inp
545:                 raise
```

```text
815:     @torch.inference_mode()
816:     def hessian_inverse(self, H: torch.Tensor):
817:         # Capture a writable view of the Hessian diagonal so we can restore it between attempts.
818:         diag_view = H.diagonal()
819:         orig_diag = diag_view.clone()
820:
821:         # When a block is numerically singular, pure damping can stall at 1.0.
822:         # Prepare a tiny diagonal floor (relative to the largest entry) that we
823:         # only inject if the normal damping loop fails. Keeping the scale near 1e-6
824:         # of the dominant entry keeps the bias negligible for healthy layers while
825:         # still rescuing pathological Hessian blocks.
826:         base_abs_max = torch.max(orig_diag.abs()).item()
827:         if not math.isfinite(base_abs_max) or base_abs_max == 0.0:
828:             base_abs_max = 1.0
829:         floor_base = base_abs_max * 1e-6
830:         max_floor_attempts = 6
831:         used_damp = self.qcfg.damp_percent
832:         last_error = None
833:
834:         attempt = 0
835:         while attempt <= max_floor_attempts:
836:             if attempt == 0:
837:                 current_diag = orig_diag
838:             else:
839:                 floor_increment = floor_base * math.pow(10.0, attempt - 1)
840:                 current_diag = torch.clamp(orig_diag + floor_increment, min=floor_increment)
841:                 if attempt == 1:
842:                     log.warn(
843:                         f"Quantization: Module `{self.name}` -> Applying Hessian diagonal floor (+{floor_increment:.2e}) to recover positive definiteness.")
844:                 else:
845:                     log.warn(
846:                         f"Quantization: Module `{self.name}` -> Increasing Hessian diagonal floor to +{floor_increment:.2e}.")
```

## vllm/auto_gptq.py
Path: `${VLLM_ENV}/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/auto_gptq.py`
SHA-256: `e0b8266ea877fa209ad3b093197d3c2a0a17fc12c243fedb1e80bb71b39b2666`

```text
73:     layer: RoutedExperts,
74:     prefix: str,
75:     moe_method_cls: type,
76: ):
77:     cloned_config = deepcopy(config)
78:
79:     assert isinstance(layer, RoutedExperts)
80:     # False = skip module, None = no override, else = Positive match
81:     if (
82:         get_dynamic_override(  # noqa: E712
83:             cloned_config,  # noqa: E712
84:             layer_name=prefix,
85:         )
86:         == False
87:     ):  # noqa: E712
88:         return UnquantizedFusedMoEMethod(layer.moe_config)
89:
90:     if prefix:
91:         # Dynamic per module/layer rules may override base config
92:         override_config(cloned_config, prefix=prefix)
93:
94:     return moe_method_cls(cloned_config, layer.moe_config)
95:
96:
```

```text
238:         return None
239:
240:     def get_quant_method(
241:         self, layer: torch.nn.Module, prefix: str
242:     ) -> "QuantizeMethodBase | None":
243:         if isinstance(layer, RoutedExperts):
244:             from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config
245:
246:             if not check_moe_marlin_supports_layer(layer, self.group_size):
247:                 logger.warning_once(
248:                     f"Layer '{prefix}' is not supported by GPTQMoeMarlin. "
249:                     "Falling back to Moe WNA16 kernels."
250:                 )
251:                 return MoeWNA16Config.from_config(self.full_config).get_quant_method(
252:                     layer, prefix
253:                 )
254:             moe_quant_method = get_moe_quant_method(
255:                 self, layer, prefix, AutoGPTQMoEMethod
256:             )
257:             if moe_quant_method is None:
258:                 return None
259:             moe_quant_method.input_dtype = get_marlin_input_dtype(prefix)
260:             return moe_quant_method
261:
262:         quant_method = get_linear_quant_method(
263:             self, layer, prefix, AutoGPTQLinearMethod
264:         )
265:         if quant_method is None:
266:             return None
```

```text
515:         if self.quant_config.group_size != -1:
516:             scales_size13 = hidden_size // self.quant_config.group_size
517:             w2_scales_size = (
518:                 intermediate_size_full
519:                 if self.quant_config.desc_act
520:                 else intermediate_size_per_partition
521:             )
522:             scales_size2 = w2_scales_size // self.quant_config.group_size
523:             strategy = FusedMoeWeightScaleSupported.GROUP.value
524:         else:
525:             scales_size13 = 1
526:             scales_size2 = 1
```
