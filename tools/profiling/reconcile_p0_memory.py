#!/usr/bin/env python3
"""Reconcile captured CUDA calls with host physical categories and PSS attribution.

Never adds CUDA pin bytes to RSS or host Shmem. Shared backing identities dedupe
aliases without exposing PFNs; residency is constrained by successful CUDA pin
API calls and independently observed PSS, not inferred from virtual size alone.
"""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v100'))
from allocation_ledger import union_bytes


def live_at(events, stamp_ns):
    live = {'device': {}, 'host_allocated': {}, 'host_registered': {}}
    alloc = {'cudaMalloc': 'device', 'cudaHostAlloc': 'host_allocated',
             'cudaMallocHost': 'host_allocated', 'cudaHostRegister': 'host_registered'}
    free = {'cudaFree': 'device', 'cudaFreeHost': 'host_allocated', 'cudaHostUnregister': 'host_registered'}
    for e in events:
        if e['monotonic_ns'] > stamp_ns:
            break
        if e['result']:
            continue
        if e['api'] in alloc:
            live[alloc[e['api']]][e['ptr']] = e
        elif e['api'] in free:
            live[free[e['api']]].pop(e['ptr'], None)
    return live


def backing_union(ranges, maps, page=4096):
    mappings = []
    for line in maps.splitlines():
        f = line.split()
        lo, hi = (int(v, 16) for v in f[0].split('-'))
        mappings.append((lo, hi, int(f[2], 16), f[3], int(f[4]), f[1]))
    backing = defaultdict(list)
    unresolved = []
    for address, size in ranges:
        lo, hi = address // page * page, (address + size + page - 1) // page * page
        covered = []
        for a, b, offset, device, inode, permissions in mappings:
            left, right = max(a, lo), min(b, hi)
            if right <= left:
                continue
            covered.append((left, right - left))
            if inode and permissions.endswith('s'):
                backing[(device, inode)].append((offset + left - a, right - left))
            else:
                unresolved.append((left, right - left))
        if union_bytes(covered) != hi - lo:
            raise ValueError('live pinned range not completely covered by captured process maps')
    return dict(shared_backing_page_bytes=sum(union_bytes(v) for v in backing.values()),
                shared_backing_objects=len(backing), unresolved_virtual_page_bytes=union_bytes(unresolved),
                virtual_page_union_bytes=union_bytes(ranges, page),
                method='Union by shared mapping (device,inode,file offset), excluding device-pointer aliases. '
                'Not PFN/pagemap measurement. Unresolved private/anonymous mappings are separate.')


def physical_partition(worksheet):
    host = worksheet['host_disjoint_bytes']
    proc = worksheet['overlapping_process_view_bytes']
    app = {'app_anonymous_pss': proc['Pss_Anon'], 'app_shared_backing_pss': proc['Pss_Shmem'],
           'app_file_cache_pss': proc['Pss_File']}
    if sum(app.values()) != proc['Pss']:
        raise ValueError('PSS subtype sum is inconsistent')
    parts = dict(app)
    parts.update(other_anonymous=host['anonymous'] - app['app_anonymous_pss'],
                 other_shared_tmpfs=host['shared_tmpfs'] - app['app_shared_backing_pss'],
                 other_file_cache=host['file_cache_excluding_shmem'] - app['app_file_cache_pss'])
    for key in ('free', 'buffers', 'slab', 'kernel_stacks', 'page_tables',
                'unassigned_kernel_driver_reserved_and_other'):
        parts[key] = host[key]
    if min(parts.values()) < 0 or sum(parts.values()) != worksheet['host_total_bytes']:
        raise ValueError('physical/PSS attribution does not conserve host total')
    return dict(bytes=parts, sum_bytes=sum(parts.values()),
                interpretation='Disjoint measured host categories with proportional PSS attribution inside '
                'anonymous/shmem/file-cache categories. PSS assigns shared-page fractions, not ownership. '
                'CUDA API pin bytes are overlapping evidence and are NOT added. Free is the observed '
                'physical safety floor; MemAvailable is separate reclaim-aware headroom. '
                'Observed file cache is not proof of the minimum required working set.')


def dsv4_components(geometry, context):
    def al(n): return (n + 255) // 256 * 256
    g = geometry
    D = g['deepseek4.attention.key_length']
    ratios = g['deepseek4.attention.compress_ratios'][:g['deepseek4.block_count']]
    if len(ratios) != g['deepseek4.block_count']:
        raise ValueError('missing layer ratios')
    idx = g['deepseek4.attention.indexer.key_length']
    result = defaultdict(int)
    for ratio in ratios:
        result['raw_window_rings'] += al((g['deepseek4.attention.sliding_window'] + 4) * D * 2)
        if ratio == 4:
            result['csa_compressed_kv'] += al((context // 4 + 1) * D * 2)
            result['csa_indexer_keys'] += al((context // 4 + 1) * idx * 2)
            result['compressor_recurrence_rings'] += 2 * al(8 * 2 * D * 4) + 2 * al(8 * 2 * idx * 4)
        elif ratio == 128:
            result['hca_compressed_kv'] += al((context // 128 + 1) * D * 2)
            result['compressor_recurrence_rings'] += 2 * al(128 * D * 4)
        elif ratio != 0:
            raise ValueError('unsupported compression ratio')
    HD = g['deepseek4.attention.head_count'] * D
    OL = g['deepseek4.attention.output_group_count'] * g['deepseek4.attention.output_lora_rank']
    result['attention_activation_and_selection'] = (4 * al(2 * D * 4) + al(HD * 4) + al(HD // 32 * 36)
        + al(OL * 4) + al(OL // 32 * 36) + al(g['deepseek4.attention.indexer.top_k'] * 4))
    return dict(bytes=dict(result), sum_bytes=sum(result.values()), context=context,
                evidence='Source-derived aligned suballocations in dsv4_alloc; independently match a '
                'measured cudaMalloc byte count before accepting this decomposition. kSpecSlack=4.')


def gpu_memory(sample, pid):
    gpu = ET.fromstring(sample['gpu_xml']).find('gpu')
    def mib(path):
        value = gpu.findtext(path)
        return int(value.split()[0]) * 1024**2 if value and value.endswith('MiB') else None
    used = None
    for process in gpu.findall('processes/process_info'):
        if process.findtext('pid') == str(pid):
            value = process.findtext('used_memory')
            if value and value.endswith('MiB'): used = int(value.split()[0]) * 1024**2
    processes = [dict(host_pid=p.findtext('pid'), type=p.findtext('type'),
                      used_memory=p.findtext('used_memory')) for p in gpu.findall('processes/process_info')]
    return dict(total_bytes=mib('fb_memory_usage/total'), used_bytes=mib('fb_memory_usage/used'),
                reserved_bytes=mib('fb_memory_usage/reserved'),
                free_bytes=mib('fb_memory_usage/free'), process_used_bytes=used, processes=processes,
                process_attribution='PID match only. NVIDIA may report host PIDs while /proc uses container PIDs; '
                'do not substitute a sole process without a verified namespace mapping.', resolution_bytes=1024**2)


def reconcile(root, geometry, inventory):
    components = dsv4_components(geometry, 135168)
    tensors = list(csv.DictReader(inventory.open()))
    roles = defaultdict(int)
    for row in tensors: roles[row['role']] += int(row['bytes'])
    model = json.loads((root/'model-identity.json').read_text())
    result = dict(schema=1,source_run=39194149121,geometry_artifact_id=12140800382,
        memory_artifact_id=12161799551,
        memory_artifact_zip_sha256='fa1995c21d2fba4cacb9fa877d5c447def118fe47575d617466d2fa8f8c6e8e2',
        model_shard_sha256=model['sha256'],matches_local_SHA256SUMS=model['matches_local_SHA256SUMS'],
        allocator_source_comparison='dsv4_alloc function text is identical in the historical working source '
            'and upstream 7c893ac17e61ec45c934b3278356807fc4626c3c; metadata-derived total matches '
            'the observed 948249856-byte allocation in both immutable binaries.',
        gguf_tensor_bytes_by_role=dict(roles),dsv4_allocation=components,bases={})
    for base in ('historical', 'current'):
        with gzip.open(root / (base + '-allocations.jsonl.gz'), 'rt') as f:
            events = [json.loads(s) for s in f]
        with gzip.open(root / base / 'telemetry.jsonl.gz', 'rt') as f:
            samples = [json.loads(s) for s in f]
        sample = max((s for s in samples if s.get('memory_worksheet')),
                     key=lambda s:s['memory_worksheet']['overlapping_process_view_bytes']['Pss'])
        live = live_at(events, int(sample['monotonic_s'] * 1e9))
        host = [(e['ptr'],e['bytes']) for kind in ('host_allocated','host_registered') for e in live[kind].values()]
        mallocs = [e for e in live['device'].values()]
        matches = [e for e in mallocs if e['bytes'] == components['sum_bytes']]
        if len(matches) != 1:
            raise ValueError('source-derived attention arena does not match observed allocation')
        driver = gpu_memory(sample, events[0]['pid'])
        requested = sum(e['bytes'] for e in mallocs)
        device_parts = dict(application_live_api_requested=requested,
                            unattributed_used=driver['used_bytes']-requested,
                            driver_reserved=driver['reserved_bytes'], free=driver['free_bytes'])
        if any(v is None or v < 0 for v in device_parts.values()):
            raise ValueError('GPU capacity budget is incomplete or inconsistent')
        rounding = driver['total_bytes'] - sum(device_parts.values())
        if abs(rounding) > 3 * driver['resolution_bytes']:
            raise ValueError('GPU capacity conservation exceeds NVML quantization tolerance')
        result['bases'][base] = dict(sample_utc=sample['utc'],sample_monotonic_s=sample['monotonic_s'],
            host_physical_partition=physical_partition(sample['memory_worksheet']),
            headroom_bytes=sample['memory_worksheet']['headroom_bytes'],
            swap=sample['memory_worksheet']['swap'],
            host_pinned_api_requested_bytes=union_bytes(host),
            host_pinned_backing=backing_union(host,sample['process_maps']),
            host_registered_calls=sum(e['api']=='cudaHostRegister' for e in events),
            cuda_alloc_calls=[dict(seq=e['seq'],api=e['api'],bytes=e['bytes'],caller=e['caller'])
                              for e in events if e['api'] in ('cudaMalloc','cudaHostAlloc','cudaHostRegister')],
            device_live_api_bytes=requested,device_nvml=driver,
            device_capacity_partition=dict(bytes=device_parts,rounding_adjustment_bytes=rounding,
                sum_with_rounding_bytes=sum(device_parts.values())+rounding,
                interpretation='Application requested bytes are inside NVML used. The remaining used bytes '
                'include context/modules/vendor pools/other clients/alignment; they are not split by guesswork. '
                'Reserved and free are separate. Signed rounding reflects MiB-granularity NVML reporting.'),
            device_nvml_minus_api_bytes=(driver['process_used_bytes']-requested if driver['process_used_bytes'] is not None else None),
            device_residual_scope='Combined CUDA context/modules/vendor-internal/reserved/rounding; '
                'not separately measured occupancy or a fabricated per-component split.',
            source_attention_arena_event_seq=matches[0]['seq'],
            node_host_memory_bytes=sample['nodes'],process_numa_mapped_bytes=sample['numa_mapped_bytes'],
            realized_anon_huge_bytes=sample['memory_worksheet']['overlapping_process_view_bytes'].get('AnonHugePages'),
            cgroup_memory_current=sample['cgroup_memory_current'],cgroup_events=sample['cgroup_memory_events'],
            page_cache_scope='Whole-host observed file-cache working footprint; per-file resident pages '
                'and a minimum sufficient cache size are not measured.',
            numa_scope='Node placement counts and node meminfo; QPI bytes/latency not measured.',
            units='All numeric memory values bytes; GiB=bytes/1073741824, GB=bytes/1000000000')
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path);ap.add_argument('geometry',type=Path);ap.add_argument('inventory',type=Path)
    ap.add_argument('--output', type=Path, required=True);args=ap.parse_args()
    args.output.write_text(json.dumps(reconcile(args.root,json.loads(args.geometry.read_text()),args.inventory),indent=2)+'\n')


if __name__ == '__main__': main()
