#!/usr/bin/env python3
"""Inspect 4B storage on CPU, or explicitly test prebuilt V3 on one admitted GPU.

No downloads, production changes, JIT compilation or full checkpoint exports.
RTN is used only for the diagnostic CPU storage audit, never as a quality recipe.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def tensor_sha256(tensor):
    import torch
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def cpu_audit(manifest, output_parent):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file, load_file
    from gptqmodel.utils.gptq_pro_contract import numerical_ladder
    from gptqmodel.utils.gptq_pro_manifest import sha256_file
    selected = [next(m for m in manifest['modules'] if m['name'].endswith(suffix))
                for suffix in ('linear_attn.in_proj_z.weight','self_attn.k_proj.weight')]
    results=[]
    source=Path(manifest['source_path'])
    for module in selected:
        with safe_open(str(source/module['shard']),framework='pt',device='cpu') as stream:
            weight=stream.get_tensor(module['name'])
        n,k=weight.shape
        group=64
        if k%group:
            raise ValueError('Selected CPU storage probe requires complete 64-column groups')
        maxima=weight.float().reshape(n,k//group,group).abs().amax(dim=2)
        scales=torch.where(maxima>0,maxima/7,torch.ones_like(maxima)).T.contiguous()
        expanded=scales.T.repeat_interleave(group,dim=1)
        codes=torch.round(weight.float()/expanded+8).clamp(0,15).to(torch.int32).T.contiguous()
        x=torch.randn((3,k),generator=torch.Generator().manual_seed(787)).mul_(0.1).bfloat16()
        report,tensors=numerical_ladder(x,weight,codes,scales,group)
        with tempfile.TemporaryDirectory(prefix='gptq-storage-probe-',dir=output_parent) as directory:
            path=Path(directory)/'storage.safetensors'
            save_file(tensors,str(path),metadata={'qzero_format':'2','kind':'diagnostic_only'})
            loaded=load_file(str(path),device='cpu')
            if set(loaded)!=set(tensors) or any(not torch.equal(loaded[key],value) for key,value in tensors.items()):
                raise AssertionError('safetensors save/reload changed stored tensors')
            file_hash=sha256_file(path)
        results.append({
            'name':module['name'],'K':k,'N':n,'group_size':group,
            'algorithm':'diagnostic RTN; not GPTQ and not a candidate-quality result',
            'activations':'three deterministic synthetic BF16 rows; not captured model states',
            'source_tensor_sha256':tensor_sha256(weight),
            'packed_tensor_sha256':{key:tensor_sha256(value) for key,value in tensors.items()},
            'temporary_storage_sha256':file_hash,'safetensors_roundtrip_exact':True,
            **report,
        })
        del weight,maxima,scales,expanded,codes,x,tensors,loaded
        gc.collect()
    return results


def cuda_audit(cases, report, save, *, extension_file=None, extension_sha256=None):
    if not cases:
        raise ValueError('An empty CUDA case list cannot certify a kernel')
    import torch
    from gptqmodel.utils._extension_loader import load_extension_module
    from gptqmodel.utils.gptq_pro_contract import pack_int4,v3_reference,error_metrics
    from gptqmodel.utils.torch import tf32_high_precision_guard
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not visible or visible=='-1' or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('CUDA mode requires exactly one gpuq-admitted visible device')
    if torch.cuda.get_device_capability(0)!=(8,6):
        raise RuntimeError('The live contract gate requires sm_86')
    # Import prebuilt only. Do not use ensure_gptq_pro_loaded(), which may remove
    # a cache directory and JIT compile if the extension is unavailable.
    from gptqmodel.utils.gptq_pro_manifest import sha256_file
    if (extension_file is None) != (extension_sha256 is None):
        raise ValueError('extension_file and extension_sha256 must be supplied together')
    if extension_file is not None:
        from gptqmodel.utils.gptq_pro_prebuilt import load_pinned_v3
        extension, identity = load_pinned_v3(extension_file, extension_sha256)
    else:
        extension=load_extension_module('gptqmodel_gptq_pro_kernels_v3')
        if not callable(getattr(extension,'gptq_pro_gemm',None)):
            raise ImportError('Prebuilt V3 extension lacks gptq_pro_gemm')
        extension_path=Path(extension.__file__).resolve(strict=True)
        identity={'path':str(extension_path),'sha256':sha256_file(extension_path),
                  'sha256_pin_verified':False,'source_build_identity_proven':False}
    report['prebuilt_extension']=identity
    report['kernel_source_sha256']={str(path.relative_to(ROOT)):sha256_file(path)
        for path in sorted((ROOT/'gptqmodel_ext/gptq_pro').iterdir())
        if path.suffix in ('.cu','.cuh','.cpp')}
    report['gpu']={'name':torch.cuda.get_device_name(0),'visible_devices':visible,
                   'capability':[8,6],'torch':torch.__version__,'runtime':torch.version.cuda}
    report['tolerances']={'atol':0.002,'rtol':0.02,'max_normalized_rmse':0.002}
    torch.cuda.reset_peak_memory_stats(0)
    results=[]
    shape_key=None
    with torch.inference_mode(),tf32_high_precision_guard():
        for case in cases:
            k,n,group,m=case['K'],case['N'],case['group_size'],case['M']
            key=k,n,group
            if key!=shape_key:
                if shape_key is not None:
                    del codes,scales,qweight,qzeros,idx,x
                generator=torch.Generator().manual_seed(787)
                codes=torch.randint(0,16,(k,n),generator=generator,dtype=torch.int32).cuda()
                scales=(torch.rand(((k+group-1)//group,n),generator=generator)*0.004+0.001).half().cuda()
                qweight=pack_int4(codes)
                qzeros=pack_int4(torch.full(scales.shape,8,device='cuda',dtype=torch.int32),axis=1)
                idx=torch.arange(k,device='cuda',dtype=torch.int32)//group
                x=(torch.randn((64,k),generator=generator)*0.25).half().cuda()
                shape_key=key
            expected=v3_reference(x[:m],qweight,scales,group,qzeros=qzeros,g_idx=idx,qzero_format=2)
            actual=extension.gptq_pro_gemm(x[:m].contiguous(),qweight,scales,group,'auto')
            torch.cuda.synchronize()
            metrics=error_metrics(actual,expected)
            close=bool(torch.isclose(actual.float(),expected.float(),atol=0.002,rtol=0.02).all().item())
            nrms=metrics['normalized_rmse']
            passed=metrics['finite'] and close and (nrms is None or nrms<=0.002)
            results.append({**case,'kernel_mode':'auto','passed':passed,**metrics})
            report['executed_cases']=results
            if not passed:
                save()
                raise AssertionError(f'V3 numerical gate failed for K={k},N={n},g={group},M={m}')
            del actual,expected
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated(0)
    report['live_kernel_shape_parity']=True
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['manifest','cpu','cuda'],default='manifest')
    parser.add_argument('--expected-commit')
    parser.add_argument('--extension-file',type=Path)
    parser.add_argument('--extension-sha256')
    args=parser.parse_args()
    if (args.extension_file is None) != (args.extension_sha256 is None):
        parser.error('--extension-file and --extension-sha256 must be supplied together')
    if args.extension_file is not None and args.mode != 'cuda':
        parser.error('Explicit extension selection is only valid for CUDA mode')
    source=args.source.resolve(strict=True)
    output=args.output.resolve()
    if output.exists() or output.is_relative_to(source):
        parser.error('Output must be a new path outside the source checkpoint')
    if args.mode=='cuda' and not args.expected_commit:
        parser.error('CUDA mode requires --expected-commit in a clean pinned worktree')
    if args.mode!='cuda':
        os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    os.environ.setdefault('OMP_NUM_THREADS','2')
    import torch
    from gptqmodel.utils.gptq_pro_contract import DISPATCH_ROWS,storage_bytes
    from gptqmodel.utils.gptq_pro_manifest import inspect_qwen35_4b,sha256_file
    torch.set_num_threads(2)
    output.parent.mkdir(parents=True,exist_ok=True)
    report={'schema_version':1,'status':'started','mode':args.mode,'started_unix':time.time(),
            'live_kernel_shape_parity':False,'full_model_quality_verified':False,
            'mtp_executed':False,'vision_executed':False,'executed_cases':[]}
    def save():
        temp=output.with_name(output.name+'.tmp')
        temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        temp.replace(output)
    try:
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        diff=subprocess.check_output(['git','diff','HEAD','--binary'],cwd=ROOT)
        status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
        if args.expected_commit and (head!=args.expected_commit or diff or status.strip()):
            raise RuntimeError('Expected a clean pinned source commit')
        report['provenance']={'commit':head,'tracked_diff_sha256':hashlib.sha256(diff).hexdigest(),
                              'implementation_sha256':{str(path.relative_to(ROOT)):sha256_file(path) for path in (
                                Path(__file__),ROOT/'gptqmodel/utils/gptq_pro_contract.py',
                                ROOT/'gptqmodel/utils/gptq_pro_manifest.py',ROOT/'gptqmodel/utils/gptq_pro_prebuilt.py')},
                              'torch':torch.__version__}
        manifest=inspect_qwen35_4b(source)
        report['manifest']=manifest
        cases=[{**shape,'group_size':group,'M':m}
               for shape in manifest['unique_shapes'] for group in (16,32,64,128) for m in DISPATCH_ROWS]
        report['planned_cases']=cases
        report['planned_case_count']=len(cases)
        report['decoder_storage_by_group']={str(g):sum(storage_bytes(m['K'],m['N'],g)['total']
                                                       for m in manifest['modules']) for g in (16,32,64,128)}
        report['storage_scope']='Selected decoder linears only; excludes dense skips, KV, MTP, vision and runtime workspaces'
        save()
        if args.mode=='cpu':
            report['real_weight_audits']=cpu_audit(manifest,output.parent)
        elif args.mode=='cuda':
            cuda_audit(cases,report,save,extension_file=args.extension_file,extension_sha256=args.extension_sha256)
        report['status']='passed'
        report['completed_unix']=time.time()
        save()
        print(json.dumps({'status':report['status'],'mode':args.mode,'modules':manifest['module_count'],
                          'planned_cases':len(cases),'executed_kernel_cases':len(report['executed_cases']),
                          'output':str(output)}))
        return 0
    except Exception as error:
        report['status']='failed'
        report['error']=f'{type(error).__name__}: {error}'
        save()
        raise


if __name__=='__main__':
    raise SystemExit(main())
