#!/usr/bin/env python3
"""Read-only profiler provenance and SM70 counter-access probe; never changes driver settings."""
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v100'))
import p0


def main():
    out = Path(os.environ['EVIDENCE']).absolute()
    out.mkdir(parents=True, exist_ok=True)
    ncu = shutil.which('ncu')
    binary = Path('build-v100-diagnostics/strata-device').absolute()
    manifest = p0.manifest(os.environ.get('ABYSS_GPU', '0'))
    p0.write_json(out / 'toolchain_manifest.json', manifest)
    report = {'ncu': ncu, 'counter_access': 'not tested', 'nsys_available': shutil.which('nsys') is not None,
              'source': 'existing runner installation; no profiler downloaded or redistributed',
              'driver_configuration_changed': False, 'production_kernel_counters': False,
              'source_license_gate': 'partial until package/license evidence and image provenance are reviewed'}
    if not ncu or not manifest['ncu_version_compatible']:
        report['counter_access'] = 'blocked: missing or incompatible Nsight Compute'
    else:
        resolved = Path(ncu).resolve()
        p0.write_json(out / 'package_provenance.json', {
            'package_owner': p0.command(['dpkg-query', '-S', str(resolved)]),
            'installed_packages': p0.command(['dpkg-query', '-W', '-f=${Package} ${Version}\n', '*nsight*']),
            'cuda_image_recipe': Path('deploy/v100-runner/Dockerfile').read_text(),
            'image_digest': 'not exposed inside this runner',
            'license_files': [{'path': str(p), 'sha256': p0.sha256(p)} for p in resolved.parent.glob('*')
                              if p.is_file() and any(x in p.name.lower() for x in ('license', 'eula'))]})
        if not binary.exists():
            raise ValueError('build the SM70 device smoke target before the profiler probe')
        report['test_binary_sha256'] = p0.sha256(binary)
        argv = [ncu, '--clock-control', 'none', '--set', 'basic', '--launch-count', '1',
                '--export', str(out / 'sm70-counter-smoke'), str(binary), '--selftest']
        result = p0.command(argv, timeout=180)
        p0.write_json(out / 'counter_smoke_process.json', result)
        log = result.get('stdout', '') + result.get('stderr', '')
        (out / 'counter_smoke.log').write_text(log)
        if 'ERR_NVGPUCTRPERM' in log or 'permission' in log.lower():
            report['counter_access'] = 'blocked: driver/container denies performance counters'
        elif result['returncode'] == 0 and (out / 'sm70-counter-smoke.ncu-rep').is_file():
            report['counter_access'] = 'SM70 kernel capture succeeded'
            raw = p0.command([ncu, '--import', str(out / 'sm70-counter-smoke.ncu-rep'), '--csv', '--page', 'raw'], timeout=60)
            (out / 'sm70-counter-smoke.csv').write_text(raw.get('stdout', '') + raw.get('stderr', ''))
            report['counter_report_decode_returncode'] = raw['returncode']
        else:
            report['counter_access'] = 'blocked: profiler smoke failed; see process/log evidence'
    p0.write_json(out / 'profiler_gate.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
