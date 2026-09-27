"""Offline extraction of only the user-selected vision encoder, no policy server."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','.22')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys
sys.path.insert(0,'/home/user/Project/openpi_sonic/src')
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import traverse_util
from openpi.models import siglip
from openpi.shared.image_tools import resize_with_pad
from prepare import OUT, CHECKPOINT, sha, write_json
from metrics import descriptors


def restore_vision(checkpoint=CHECKPOINT):
    p=checkpoint/'params'
    shard=jax.sharding.SingleDeviceSharding(jax.devices()[0])
    with ocp.PyTreeCheckpointer() as c:
        meta=c.metadata(p)
        item={'params':{'PaliGemma':{'img':meta['params']['PaliGemma']['img']}}}
        args=jax.tree.map(lambda _:ocp.ArrayRestoreArgs(sharding=shard,restore_type=jax.Array,dtype=jnp.bfloat16),item)
        result=c.restore(p,args=ocp.args.PyTreeRestore(item=item,restore_args=args,transforms={}))
    flat=traverse_util.flatten_dict(result['params']['PaliGemma']['img'])
    assert all(k[-1]=='value' for k in flat)
    flat={k[:-1]:v for k,v in flat.items()}
    h=hashlib.sha256(); sizes=[]
    for k,v in sorted(flat.items()):
        x=np.asarray(v)
        h.update('/'.join(k).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
        sizes.append({'name':'/'.join(k),'shape':list(x.shape),'dtype':str(x.dtype)})
    return traverse_util.unflatten_dict(flat),h.hexdigest(),sizes


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);a=parser.parse_args()
    samples=json.loads((OUT/'inputs/samples.json').read_text())
    checkpoint=Path(json.loads((OUT/'analysis_config.json').read_text())['checkpoint'])
    if a.limit:samples=samples[:a.limit]
    print('device',jax.devices(),flush=True)
    params,parameter_hash,sizes=restore_vision(checkpoint)
    print('vision restored',parameter_hash,flush=True)
    module=siglip.Module(num_classes=2048,variant='So400m/14',pool_type='none',scan=True,dtype_mm='bfloat16')
    @jax.jit
    def forward(p,images):
        rgb=resize_with_pad(images,224,224)
        rgb=rgb.astype(jnp.float32)/255.*2.-1.
        y,_=module.apply({'params':p},rgb,train=False)
        return y.astype(jnp.float32)
    embeddings=[];start=time.monotonic();repeat_error=None
    for i,s in enumerate(samples):
        assert sha(s['image'])==s['image_sha256']
        bgr=cv2.imread(s['image']);assert bgr is not None and bgr.shape==(480,640,3)
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)[None]
        y=np.asarray(forward(params,rgb))
        assert y.shape==(1,256,2048) and np.isfinite(y).all()
        if i==0:
            yr=np.asarray(forward(params,rgb));repeat_error=float(np.max(np.abs(y-yr)))
            assert repeat_error==0
        embeddings.append(y[0])
        if (i+1)%20==0 or i==0 or i+1==len(samples):
            print(json.dumps({'done':i+1,'total':len(samples),'seconds':round(time.monotonic()-start,1)}),flush=True)
    tokens=np.stack(embeddings)
    desc=descriptors(tokens)
    dest=OUT/('features_probe' if a.limit else 'features');dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'embeddings.npz',sample_ids=np.array([s['sample_id'] for s in samples]),
                        spatial=desc['spatial'],global_mean=desc['global'],patch_tokens=tokens.astype(np.float16))
    sources=['src/openpi/models/siglip.py','src/openpi/models/pi0.py','src/openpi/models/model.py',
             'src/openpi/shared/image_tools.py','src/openpi/policies/g1_sonic_policy.py']
    write_json(dest/'provenance.json',{
        'checkpoint':str(checkpoint),'checkpoint_metadata_sha256':sha(checkpoint/'params/_METADATA'),
        'vision_bfloat16_parameter_sha256':parameter_hash,'parameter_tensors':sizes,
        'jax_version':jax.__version__,'orbax_version':ocp.__version__,
        'devices':[str(d) for d in jax.devices()], 'token_shape':list(tokens.shape),
        'descriptor_shapes':{k:list(v.shape) for k,v in desc.items()},
        'determinism_max_abs_error':repeat_error,'runtime_seconds':time.monotonic()-start,
        'preprocessing':'opencv RGB -> OpenPI uint8 bilinear resize/pad224 -> float32 [-1,1]',
        'source_hashes':{s:sha(Path('/home/user/Project/openpi_sonic')/s) for s in sources},
        'feature_file_sha256':sha(dest/'embeddings.npz'),
        'analysis_config_sha256':sha(OUT/'analysis_config.json'),
    })
    print('SAVED',dest,flush=True)


if __name__=='__main__':main()
