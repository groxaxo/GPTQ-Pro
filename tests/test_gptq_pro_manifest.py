"""Header-only tests use sparse fake payloads, never allocating full model weights."""
import json
import struct

import pytest

from gptqmodel.utils.gptq_pro_manifest import inspect_qwen35_4b


def fixture_data():
    kinds=['linear_attention']*3+['full_attention']
    config={'model_type':'qwen3_5','text_config':{
        'num_hidden_layers':32,'hidden_size':2560,'intermediate_size':9216,
        'layer_types':kinds*8,'attn_output_gate':True,
        'num_attention_heads':16,'num_key_value_heads':4,'head_dim':256,
        'linear_num_key_heads':16,'linear_num_value_heads':32,
        'linear_key_head_dim':128,'linear_value_head_dim':128}}
    shapes={'mlp.gate_proj':[9216,2560],'mlp.up_proj':[9216,2560],'mlp.down_proj':[2560,9216],
            'linear_attn.in_proj_qkv':[8192,2560],'linear_attn.in_proj_z':[4096,2560],
            'linear_attn.out_proj':[2560,4096],'self_attn.q_proj':[8192,2560],
            'self_attn.k_proj':[1024,2560],'self_attn.v_proj':[1024,2560],
            'self_attn.o_proj':[2560,4096]}
    header={};offset=0
    for layer,kind in enumerate(config['text_config']['layer_types']):
        for suffix,shape in shapes.items():
            if suffix.startswith('linear_attn') and kind!='linear_attention':continue
            if suffix.startswith('self_attn') and kind!='full_attention':continue
            name=f'model.language_model.layers.{layer}.{suffix}.weight'
            size=shape[0]*shape[1]*2
            header[name]={'dtype':'BF16','shape':shape,'data_offsets':[offset,offset+size]}
            offset+=size
    return config,header,offset


def write_fixture(path,config,header,payload_size):
    path.mkdir(exist_ok=True)
    (path/'config.json').write_text(json.dumps(config))
    (path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{
        name:'model.safetensors' for name in header}}))
    raw=json.dumps(header).encode()
    with (path/'model.safetensors').open('wb') as stream:
        stream.write(struct.pack('<Q',len(raw)));stream.write(raw)
        stream.truncate(8+len(raw)+payload_size)  # Sparse holes; only headers occupy disk.


def test_complete_inventory_is_derived_not_a_hardcoded_count(tmp_path):
    cfg,header,size=fixture_data();write_fixture(tmp_path,cfg,header,size)
    report=inspect_qwen35_4b(tmp_path)
    assert report['module_count']==200
    assert len(report['unique_shapes'])==6
    assert report['source_full_weight_hash_verified'] is False
    assert report['mtp_tensor_count']==0


@pytest.mark.parametrize('problem',['architecture','layer_type','quantized','attention_dimensions','output_gate'])
def test_invalid_source_configuration_fails(tmp_path,problem):
    cfg,header,size=fixture_data()
    if problem=='architecture':cfg['text_config']['hidden_size']=4096
    elif problem=='layer_type':cfg['text_config']['layer_types'][0]='unknown'
    elif problem=='quantized':cfg['quantization_config']={'bits':4}
    elif problem=='attention_dimensions':cfg['text_config']['head_dim']=None
    else:cfg['text_config']['attn_output_gate']='yes'
    write_fixture(tmp_path,cfg,header,size)
    with pytest.raises(ValueError):inspect_qwen35_4b(tmp_path)


@pytest.mark.parametrize('problem',['missing_module','offset','byte_size','overlap','wrong_shape','dtype','trailing'])
def test_invalid_shard_layout_fails(tmp_path,problem):
    cfg,header,size=fixture_data();name=next(iter(header))
    if problem=='missing_module':
        first=header.pop(name);header['unrelated.weight']=first
    elif problem=='offset':header[name]['data_offsets'][1]=size+1
    elif problem=='byte_size':header[name]['shape']=[16,16]
    elif problem=='overlap':header[name]['data_offsets'][0]=1
    elif problem=='wrong_shape':header[name]['shape']=header[name]['shape'][::-1]
    elif problem=='dtype':header[name]['dtype']='I16'
    else:size+=1
    write_fixture(tmp_path,cfg,header,size)
    with pytest.raises(ValueError):inspect_qwen35_4b(tmp_path)


def test_shard_index_cannot_escape_source(tmp_path):
    root=tmp_path/'source';cfg,header,size=fixture_data();write_fixture(root,cfg,header,size)
    (tmp_path/'outside.safetensors').write_bytes(b'not a model')
    (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{
        name:'../outside.safetensors' for name in header}}))
    with pytest.raises(ValueError,match='escaped'):inspect_qwen35_4b(root)


def test_duplicate_json_is_rejected(tmp_path):
    cfg,header,size=fixture_data();write_fixture(tmp_path,cfg,header,size)
    (tmp_path/'config.json').write_text('{"model_type":"qwen3_5","model_type":"other"}')
    with pytest.raises(ValueError,match='Duplicate'):inspect_qwen35_4b(tmp_path)


def test_unresolved_index_tensor_is_rejected(tmp_path):
    cfg,header,size=fixture_data();write_fixture(tmp_path,cfg,header,size)
    index=tmp_path/'model.safetensors.index.json';data=json.loads(index.read_text())
    data['weight_map']['missing.weight']='model.safetensors';index.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Missing tensor'):inspect_qwen35_4b(tmp_path)
