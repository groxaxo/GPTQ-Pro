"""CPU oracle tests: independent bit vectors and the existing unmodified packer."""
import json

import numpy as np
import pytest
import torch

from gptqmodel.utils.gptq_pro_contract import (
    DISPATCH_ROWS, SUPPORTED_GROUPS, dequantize_v3, error_metrics,
    numerical_ladder, pack_int4, storage_bytes, unpack_int4,
    validate_v3_layout, v3_reference,
)


def example(group=16,k=32,n=32,fmt=2):
    codes = torch.arange(k*n,dtype=torch.int32).reshape(k,n) % 16
    size = k if group == -1 else group
    g_idx = torch.arange(k,dtype=torch.int32)//size
    scales = torch.full(((k+size-1)//size,n),0.125,dtype=torch.float16)
    zeros = torch.full(scales.shape,7 if fmt == 1 else 8,dtype=torch.int32)
    return codes,pack_int4(codes),scales,pack_int4(zeros,axis=1),g_idx


def validate(q,s,z,i,group=16,fmt=2):
    return validate_v3_layout(q,s,group,qzeros=z,g_idx=i,qzero_format=fmt)


def test_golden_signed_words_and_lane_order():
    codes = torch.arange(16,dtype=torch.int32).reshape(16,1)
    expected = torch.tensor([[0x76543210],[-19088744]],dtype=torch.int32)
    assert torch.equal(pack_int4(codes),expected)
    assert torch.equal(unpack_int4(expected),codes)
    assert torch.equal(pack_int4(codes.T,axis=1),expected.T)
    assert torch.equal(unpack_int4(expected.T,axis=1),codes.T)


@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('dtype',[torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64])
def test_integer_roundtrip_with_noncontiguous_input(axis,dtype):
    codes = (torch.arange(16*32).reshape(16,32)%16).to(dtype).T
    assert not codes.is_contiguous()
    result = unpack_int4(pack_int4(codes,axis=axis),axis=axis)
    assert torch.equal(result,codes.to(torch.int32))
    assert result.is_contiguous()


@pytest.mark.parametrize('code',range(16))
def test_each_nibble_repeated(code):
    codes = torch.full((16,16),code,dtype=torch.int32)
    assert torch.equal(unpack_int4(pack_int4(codes)),codes)


@pytest.mark.parametrize('bad',[-1,16,100])
def test_pack_rejects_out_of_range(bad):
    with pytest.raises(ValueError,match='codes'):
        pack_int4(torch.full((16,16),bad,dtype=torch.int32))


@pytest.mark.parametrize('axis',[True,0.0,2,-1,None])
def test_invalid_axis(axis):
    with pytest.raises(ValueError,match='axis'):
        pack_int4(torch.zeros(16,16,dtype=torch.int32),axis=axis)


@pytest.mark.parametrize('group',SUPPORTED_GROUPS)
@pytest.mark.parametrize('fmt',[1,2])
def test_all_supported_groups_and_qzero_conventions(group,fmt):
    codes,q,s,z,idx = example(group,fmt=fmt)
    result = validate(q,s,z,idx,group,fmt)
    assert result['K']==32 and result['N']==32
    expected = ((codes.float()-8)*s[idx.long()].float()).half()
    got = dequantize_v3(q,s,group,qzeros=z,g_idx=idx,qzero_format=fmt)
    assert torch.equal(got,expected)


@pytest.mark.parametrize('group',[0,True,16.0,7,None])
def test_invalid_groups(group):
    with pytest.raises(ValueError,match='group_size'):
        storage_bytes(32,32,group)


@pytest.mark.parametrize('fmt',[True,1.0,0,3,None])
def test_qzero_format_cannot_be_guessed(fmt):
    _,q,s,z,i = example()
    with pytest.raises(ValueError,match='qzero_format'):
        validate(q,s,z,i,fmt=fmt)


@pytest.mark.parametrize('problem',['wrong_zero','wrong_format','permuted_idx','scales_shape',
    'scales_dtype','negative_scale','nan_scale','inf_scale','qweight_dtype','qzeros_shape','idx_dtype'])
def test_malformed_layout_fails(problem):
    _,q,s,z,i = example()
    fmt=2
    if problem=='wrong_zero': z.fill_(0)
    elif problem=='wrong_format': fmt=1
    elif problem=='permuted_idx': i=i.flip(0)
    elif problem=='scales_shape': s=s[:1]
    elif problem=='scales_dtype': s=s.float()
    elif problem=='negative_scale': s[0,0]=-1
    elif problem=='nan_scale': s[0,0]=float('nan')
    elif problem=='inf_scale': s[0,0]=float('inf')
    elif problem=='qweight_dtype': q=q.long()
    elif problem=='qzeros_shape': z=z[:,:1]
    elif problem=='idx_dtype': i=i.long()
    with pytest.raises((ValueError,FloatingPointError)):
        validate(q,s,z,i,fmt=fmt)


@pytest.mark.parametrize('m',DISPATCH_ROWS)
def test_reference_input_rows_and_bias_stage(m):
    _,q,s,z,i = example()
    x = torch.linspace(-1,1,m*32).reshape(m,32).to(torch.bfloat16)
    bias = torch.linspace(-0.125,0.125,32).half()
    w = dequantize_v3(q,s,16,qzeros=z,g_idx=i,qzero_format=2)
    expected = ((x.half().float()@w.float()).half().float()+bias.float()).half()
    actual = v3_reference(x,q,s,16,qzeros=z,g_idx=i,qzero_format=2,bias=bias)
    assert torch.equal(actual,expected)


def test_zero_scale_and_zero_row_input():
    _,q,s,z,i = example()
    s.zero_()
    out = v3_reference(torch.ones(1,32),q,s,16,qzeros=z,g_idx=i,qzero_format=2)
    assert torch.count_nonzero(out)==0
    out = v3_reference(torch.empty(0,32),q,s,16,qzeros=z,g_idx=i,qzero_format=2)
    assert out.shape==(0,32)


@pytest.mark.parametrize('problem',['input','dequant','output'])
def test_fp16_range_errors_cannot_look_like_success(problem):
    _,q,s,z,i = example()
    x=torch.ones(1,32)
    if problem=='input': x.fill_(70000)
    elif problem=='dequant': s.fill_(10000)
    else:
        q=pack_int4(torch.full((32,32),15,dtype=torch.int32));s.fill_(3000);x.fill_(4)
    with pytest.raises(FloatingPointError):
        v3_reference(x,q,s,16,qzeros=z,g_idx=i,qzero_format=2)


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_metrics_remain_json_safe(bad):
    result=error_metrics(torch.tensor([bad]),torch.ones(1))
    assert result['finite'] is False and result['rmse'] is None
    json.dumps(result,allow_nan=False)


def test_zero_reference_metrics_and_known_error():
    result=error_metrics(torch.ones(2),torch.zeros(2))
    assert result['rmse']==1 and result['normalized_rmse'] is None
    exact=error_metrics(torch.ones(2),torch.ones(2))
    assert exact['max_abs']==exact['rmse']==0


def test_storage_accounting_matches_actual_tensor_bytes():
    _,q,s,z,i=example()
    total=sum(t.numel()*t.element_size() for t in (q,s,z,i))
    assert total==storage_bytes(32,32,16)['total']
    assert total==validate(q,s,z,i)['stored_bytes']


def test_numerical_ladder_does_not_claim_kernel_execution():
    codes,q,s,z,idx=example()
    scales=s.float()*1.0003
    source=((codes.float()-8)*scales[idx.long()]).T.contiguous().bfloat16()
    report,tensors=numerical_ladder(torch.randn(3,32),source,codes,scales,16)
    assert report['integer_roundtrip_exact']
    assert report['live_kernel']=={'status':'not_run','certified':False}
    assert report['weight_scale_cast_error']['max_abs']>0
    assert torch.equal(tensors['qweight'],q)
    json.dumps(report,allow_nan=False)


@pytest.mark.parametrize('group',[16,32,64])
def test_existing_cpu_pack_original_matches_independent_oracle(group):
    from gptqmodel.nn_modules.qlinear import PackableQuantLinear
    # Execute the real unchanged packer with only its storage receiver. No CUDA
    # qlinear construction or runtime validation is bypassed for inference.
    receiver=torch.nn.Module()
    receiver.bits=4;receiver.pack_factor=8;receiver.pack_dtype_bits=32
    receiver.pack_np_math_dtype=np.uint32;receiver.pack_np_dtype=np.int32
    codes,q,scales,zeros,idx=example(group,k=64,n=64)
    weight=((codes.float()-8)*scales[idx.long()].float()).T.contiguous()
    dense=torch.nn.Linear(64,64,bias=False)
    dense.weight.data.copy_(weight)
    PackableQuantLinear.pack_original(receiver,dense,scales.T.float(),
                                     torch.full(scales.T.shape,8.0),idx)
    assert torch.equal(receiver.qweight,q)
    assert torch.equal(receiver.qzeros,zeros)
    assert torch.equal(receiver.scales,scales)
    validate(receiver.qweight,receiver.scales,receiver.qzeros,receiver.g_idx,group=group)
