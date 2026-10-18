#!/usr/bin/env python3
"""Small external reference check; raw tensors are bounded and never uploaded."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'v100'))
from model_baseline import PACK, ENGINE, ROOT, fixed_env, launch, normalize_logprobs
from model_bringup import NAMES, SIZES
from strata_tokenizer import Tokenizer
from gguf_reader import GGUFFile
import p0


def main():
    out = Path(os.environ['EVIDENCE']).absolute()
    out.mkdir(parents=True, exist_ok=True)
    for name, size in zip(NAMES, SIZES):
        if (Path('/models') / name).stat().st_size != size:
            raise ValueError('wrong model size')
    gg = GGUFFile(Path('/models') / NAMES[0])
    vocab = len(gg.metadata['tokenizer.ggml.tokens'])
    tok = Tokenizer.from_dir(PACK / 'tokenizer')
    text = ('A server has two memory controllers, but a processor may read pages attached to another socket. '
            'Locality can reduce latency while interleaving can increase aggregate bandwidth. '
            'A faster isolated kernel does not guarantee a faster application. Consider a database that stores '
            'weather observations: each record includes a station identifier, time, temperature, and confidence. '
            'An index speeds up selective queries; a full scan may still win when most records qualify. '
            'Floating point addition is not associative. A numerical comparison should examine probability '
            'distributions, rare outliers, and prediction margins rather than only generated strings. '
            'A chef adjusts salt after tasting the soup, and a musician practices a difficult passage slowly.')
    ids = [0] + tok.encode(text)[:129]
    assert len(ids) == 130
    tokens = out / 'input.txt'
    tokens.write_text(','.join(map(str, ids)) + '\n')
    (out / 'source-text.txt').write_text(text)
    provenance = {'abyss_commit': p0.command(['git','rev-parse','HEAD']),
                  'llama_commit': p0.command(['git','-C','third_party/llama.cpp','rev-parse','HEAD']),
                  'model_integrity': 'exact byte sizes; model SHA256 not verified',
                  'cpu_oracle_binary_sha256': p0.sha256('build-v100-oracle/v100_cpu_oracle'),
                  'abyss_binary_sha256': p0.sha256(ENGINE),
                  'pack_marker': json.loads((PACK/'v100_pack_complete.json').read_text())}
    p0.write_json(out / 'provenance.json', provenance)
    with tempfile.TemporaryDirectory(prefix='abyss-oracle-', dir=os.environ.get('RUNNER_TEMP')) as tmp:
        tmp = Path(tmp)
        reference, candidate = tmp/'reference.f32', tmp/'abyss.f32'
        for name in ['reference','abyss']:
            (out/name).mkdir()
        env = fixed_env()
        launch([str(ROOT/'build-v100-oracle/v100_cpu_oracle'), str(Path('/models')/NAMES[0]),
                str(tokens), str(reference)], env, out/'reference', timeout=1800)
        env.update(STRATA_GLM_SCORE='0', STRATA_GLM_SCORE_DUMP=str(candidate))
        launch([str(ENGINE), '--glm-pack', str(PACK), '--tokens-file', str(tokens),
                '--max-new','1','--max-context','4096','--prefill','auto','--temperature','0'],
               env, out/'abyss', timeout=900)
        arrays, raw = [], {}
        for name, path in [('reference',reference),('abyss',candidate)]:
            if path.stat().st_size != 129*vocab*4:
                raise ValueError('wrong logit dump geometry')
            values = np.fromfile(path, dtype='<f4').reshape(129,vocab)
            arrays.append(normalize_logprobs(values))
            raw[name] = {'bytes': path.stat().st_size, 'sha256': p0.sha256(path)}
        ref, test = arrays
        kl = (np.exp(ref)*(ref-test)).sum(axis=1)
        top = ref.argmax(axis=1) == test.argmax(axis=1)
        target = np.array(ids[1:])
        rows = [{'input_position_zero_based':i,'kl':float(kl[i]),'same_top1':bool(top[i]),
                 'reference_top1':int(ref[i].argmax()), 'abyss_top1':int(test[i].argmax()),
                 'reference_nll':float(-ref[i,target[i]]),'abyss_nll':float(-test[i,target[i]])}
                for i in range(129)]
        result = {'scope':'pinned llama.cpp CPU vs Abyss V100 token path; 129 positions, including raw-window wrap',
                  'mean_kl':float(kl.mean()), 'max_kl':float(kl.max()), 'top1_agreement':float(top.mean()),
                  'reference_mean_nll':float(np.mean([r['reference_nll'] for r in rows])),
                  'abyss_mean_nll':float(np.mean([r['abyss_nll'] for r in rows])),
                  'raw_dumps':raw, 'positions':rows,
                  'quality_gate':'measurement only; no universal threshold inferred from upstream README',
                  'raw_dump_retention':'temporary; hashes and per-position metrics uploaded'}
        p0.write_json(out/'external_reference.json', result)
        print(json.dumps({k:v for k,v in result.items() if k!='positions'}),flush=True)


if __name__ == '__main__':
    main()
