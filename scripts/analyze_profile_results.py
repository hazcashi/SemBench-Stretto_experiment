"""Offline Stretto report. Requires matplotlib/numpy; never calls models or APIs.

python scripts/analyze_profile_results.py --results-root stretto-results/results \
    --output stretto-results/analysis
Charts use seconds, bytes converted explicitly to GiB, and observed counters.
"""
import argparse
import collections
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import html
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

GIB = 2 ** 30
COLORS = ['#2764ad', '#11a5a0', '#ec9940', '#9155a6', '#d85863', '#92a1b3', '#bfceea']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'figure.facecolor': '#ffffff',
                     'savefig.facecolor': '#ffffff', 'svg.fonttype': 'path'})


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def jsonl(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def union_length(intervals):
    """Union, not thread-sum. Callers supply one shared clock/domain."""
    end = None
    total = 0.
    for a, b in sorted(intervals):
        assert b >= a
        total += max(0., b - max(a, end if end is not None else a))
        end = max(end if end is not None else b, b)
    return total


def save_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def trace_stats(path, meta):
    events = read_json(path)['traceEvents']
    gpu = [e for e in events if e.get('ph') == 'X' and e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    assert gpu, f'No CUDA events: {path}'
    start = min(e['ts'] for e in gpu)
    end = max(e['ts'] + e['dur'] for e in gpu)
    interval = lambda group: [(e['ts'], e['ts'] + e['dur']) for e in group]
    kernel = [e for e in gpu if e['cat'] == 'kernel']
    copy = [e for e in gpu if e['cat'] != 'kernel']
    kernel_us = union_length(interval(kernel))
    active_us = union_length(interval(gpu))
    # Copy time overlapping kernels is already included in kernel_us.
    copy_only_us = active_us - kernel_us
    gap_us = end - start - active_us
    assert gap_us >= -1e-3 and copy_only_us >= -1e-3
    return dict(file=path.name, role=meta['role'], route=meta['route'], phase=meta['phase'],
                kernel_events=len(kernel), memcpy_events=sum(e['cat'] == 'gpu_memcpy' for e in copy),
                envelope_s=(end-start)/1e6, kernel_union_s=kernel_us/1e6,
                copy_only_s=copy_only_us/1e6, no_recorded_gpu_event_s=max(0, gap_us)/1e6,
                scope='One representative batch; first-to-last recorded GPU event; not whole run')


def load_run(path):
    manifest = read_json(path/'manifest.json')
    assert manifest['status'] == 'completed' and not manifest.get('backend_errors')
    assert not (path/'failure.txt').exists()
    query = read_json(path/'queries.json')[0]
    answers = []
    for name in ('benchmark_0.csv', 'silver_0.csv'):
        with (path/name).open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        keyed = {(r['name'], r['game_id']): r for r in rows}
        assert len(keyed) == len(rows), 'Duplicate player/game key requires explicit alignment'
        answers.append(keyed)
    pred, silver = answers
    assert pred.keys() == silver.keys()
    normalize = lambda value: Decimal(value) if value.strip() else None
    differences = []
    for key in pred:
        for column in ('points', 'assists'):
            if normalize(pred[key][column]) != normalize(silver[key][column]):
                differences.append(dict(name=key[0], game_id=key[1], column=column,
                                        benchmark_value=pred[key][column], silver_value=silver[key][column]))
    quality = dict(rows=len(pred), mismatch_rows=len({(d['name'], d['game_id']) for d in differences}),
                   points_mismatches=sum(d['column']=='points' for d in differences),
                   assists_mismatches=sum(d['column']=='assists' for d in differences),
                   missing_cells=sum(not r[c].strip() for a in answers for r in a.values() for c in ('points','assists')))
    # Dataset-specific cross-check, not a replacement definition of upstream F1.
    assert abs((1-quality['mismatch_rows']/quality['rows'])-query['f1_score']) < 1e-8
    events = list(jsonl(path/'events.jsonl'))
    ends = {e['data']['executor']: e['data'] for e in events if e['type'] == 'query_end'}
    components = query['cost']['component_times']
    requests = []
    lookup = {}
    for role in ('small', 'large'):
        for r in jsonl(path/f'{role}_requests.jsonl'):
            r['role'] = role
            r['route'] = (r.get('inference_stats') or {}).get('path', 'unknown')
            requests.append(r)
            lookup[role, r['request_id']] = r
    prepare = sum(r['elapsed_s'] for r in requests if r['stage'] == 'benchmark' and r['path'] == '/prepare_caches')
    # The phase hierarchy was checked against events: profiling and solve are
    # children of tuning. Cache preparation is outside configuring/tuning/execution.
    parts = {'Cache preparation': prepare, 'Configuration': components.get('configuring', 0),
             'Candidate sampling': components.get('profiling', 0),
             'Optimizer solve': components.get('optimizer_solve', 0),
             'Other tuning': components['tuning']-components.get('profiling', 0)-components.get('optimizer_solve', 0),
             'Execution': components['execution']}
    parts['Other query work'] = components['end_to_end'] - sum(parts.values())
    assert min(parts.values()) >= -0.01, parts
    assert abs(sum(parts.values()) - components['end_to_end']) < 1e-5
    aggregates = {}
    generations = collections.Counter()
    io = collections.defaultdict(lambda: collections.Counter())
    for role in ('small', 'large'):
        for r in jsonl(path/f'profiling/{role}_phases.jsonl'):
            req = lookup.get((role, r['request_id']), {})
            route = req.get('route', 'unknown')
            key = (role, r.get('stage'), r.get('phase'), route)
            if r['kind'] == 'generation':
                generations[key+('calls',)] += 1
                generations[key+('output_tokens',)] += r['output_tokens']
            elif r['kind'] == 'span':
                agg = aggregates.setdefault(key+(r['name'],), collections.Counter())
                agg['observations'] += 1
                for field in ('cpu_wall_s', 'gpu_elapsed_s', 'tensor_bytes', 'file_bytes', 'input_tokens',
                              'thread_minor_faults', 'thread_major_faults'):
                    if r.get(field) is not None:
                        agg[field] += r[field]
                        agg[field+'_measured'] += 1
                for field in ('peak_allocated_bytes', 'peak_reserved_bytes'):
                    if r.get(field) is not None:
                        agg[field] = max(agg[field], r[field])
                if r.get('error_type'):
                    agg['errors'] += 1
            elif r['kind'] == 'cache_io':
                row = io[key]
                row['file_bytes'] += r['file_bytes']
                delta = r.get('delta', {})
                row['status_'+delta.get('status', 'unknown')] += 1
                if delta.get('status') == 'measured':
                    for field in ('read_bytes', 'write_bytes', 'minor_faults', 'major_faults'):
                        row[field] += delta[field]
    resources = list(jsonl(path/'resources.jsonl'))
    sample_rows, faults = [], collections.defaultdict(collections.Counter)
    origin = datetime.fromisoformat(manifest['started_utc']).timestamp()
    seen = set()
    for s in resources:
        item = dict(elapsed_s=s['at']-origin, stage=s.get('stage', 'unknown'),
                    process_rss_gib=s.get('process_tree_rss_bytes', 0)/GIB,
                    process_pss_gib=sum(p.get('pss_bytes', 0) for p in s['processes'])/GIB,
                    cgroup_gib=(s.get('cgroup_memory', {}).get('used_bytes') or 0)/GIB,
                    cgroup_limit_gib=(s.get('cgroup_memory', {}).get('limit_bytes') or 0)/GIB,
                    host_used_gib=s.get('host_memory', {}).get('used_bytes', 0)/GIB,
                    host_total_gib=s.get('host_memory', {}).get('total_bytes', 0)/GIB)
        gpu = list(csv.reader([s.get('gpu_csv') or '']))
        if gpu and len(gpu[0]) >= 5:
            item.update(gpu_used_gib=float(gpu[0][-3])/1024,
                        gpu_total_gib=float(gpu[0][-2])/1024, gpu_util_pct=float(gpu[0][-1]))
        for role in ('runner', 'small', 'large', 'embedding'):
            item[role+'_rss_gib'] = sum(p.get('rss_bytes', 0) for p in s['processes'] if p['role'] == role)/GIB
        for proc in s['processes']:
            identity = (proc['pid'], proc['created_at'])
            key = (item['stage'], proc['role'])
            delta = proc.get('delta', {})
            status = delta.get('status', 'unknown')
            assert status in ('measured', 'first_sample'), 'Report unavailable counter status explicitly'
            assert all(k in proc for k in ('rss_bytes','pss_bytes')), 'Memory data unavailable'
            faults[key]['status_'+status] += 1
            if status == 'measured':
                assert identity in seen, 'Delta without initial process sample'
                for field in ('minor_faults', 'major_faults', 'read_bytes', 'write_bytes'):
                    faults[key][field] += delta[field]
            seen.add(identity)
        sample_rows.append(item)
    summary = read_json(path/'profiling/summary.json')
    traces = [trace_stats(path/'profiling/traces'/Path(t['path']).name, t) for t in summary['traces']]
    plans = [s for text in query['tuned_pipelines'] for s in json.loads(text)]
    executed = [e['data'] for e in events if e['type'] == 'operator_run']
    # Raw event records remain available in the source. Buckets have stable fields.
    buckets = read_json(path/'telemetry_summary.json')['operator_buckets']
    return dict(path=path, reports=manifest['profile']['reports_limit'], manifest=manifest, query=query,
                components=components, parts=parts, reference=ends['silver_8b']['component_times'],
                requests=requests, aggregates=aggregates, generations=generations, io=io,
                samples=sample_rows, faults=faults, traces=traces, plans=plans, buckets=buckets,
                quality=quality, differences=differences)


def span(run, role, stage, phase, route, name, field='gpu_elapsed_s'):
    return run['aggregates'].get((role, stage, phase, route, name), {}).get(field, 0)


def request_group(run, role, phase, route):
    return [r for r in run['requests'] if r['role'] == role and r['stage'] == 'benchmark'
            and r['phase'] == phase and r['route'] == route and r['path'] == '/text_qa']


def group_latency(run, role, phase, route):
    rows = request_group(run, role, phase, route)
    n = sum(r['inference_stats']['n_items'] for r in rows)
    return sum(r['elapsed_s'] for r in rows)/n if n else None


def generate_report(runs, out, input_root):
    out.mkdir(parents=True, exist_ok=True)
    charts = []
    pdf = PdfPages(out/'charts.pdf')
    def save(fig, name, title, note):
        fig.savefig(out/(name+'.png'), dpi=170, bbox_inches='tight')
        fig.savefig(out/(name+'.svg'), bbox_inches='tight')
        pdf.savefig(fig, bbox_inches='tight')
        charts.append(dict(name=name, title=title, note=note))
        plt.close(fig)
    def style(ax, xlabel=None):
        ax.grid(axis='x', alpha=.16)
        ax.set_axisbelow(True)
        if xlabel: ax.set_xlabel(xlabel)

    labels = [f"{r['reports']} reports" for r in runs]
    # Two distinct wall clocks: benchmark query vs entire run incl. reference.
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for ax, mode in zip(axs, ('query', 'run')):
        left = np.zeros(len(runs))
        names = list(runs[0]['parts']) if mode == 'query' else ['Benchmark query', '8B silver reference', 'Setup / other']
        for j, name in enumerate(names):
            if mode == 'query': vals = [r['parts'][name]/60 for r in runs]
            elif j == 0: vals = [r['components']['end_to_end']/60 for r in runs]
            elif j == 1: vals = [r['reference']['end_to_end']/60 for r in runs]
            else: vals = [(r['manifest']['total_wall_s']-r['components']['end_to_end']-r['reference']['end_to_end'])/60 for r in runs]
            ax.barh(labels, vals, left=left, label=name, color=COLORS[j], height=.5)
            for i, v in enumerate(vals):
                if v > 1.5: ax.text(left[i]+v/2, i, f'{v:.1f}', ha='center', va='center', fontsize=9, color='white')
            left += vals
        for i, v in enumerate(left): ax.text(v+.25, i, f'{v:.2f} min', va='center', fontsize=9)
        ax.set_xlim(0, max(left)*1.22)
        ax.set_title('Benchmark query: non-overlapping components' if mode=='query' else 'Whole run: benchmark + silver + setup')
        style(ax, 'Wall time (minutes)')
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.18), ncol=2, fontsize=8, frameon=False)
    save(fig, '01_wall_time', '實驗時間：查詢與整輪要分開看',
         '左圖扣除 tuning 的子階段，避免重複加總。Cache preparation 是 /prepare_caches 的請求時間；其他工作是剩餘時間，未細分原因。右圖包含另外產生 8B 參考答案的時間。100 篇沒有重新建立 cache；728 篇有，因此不是相同 cold/warm 條件。')

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
    for ax, r in zip(axs, runs):
        ys = np.arange(2)
        pre = [span(r, role, 'benchmark', 'execution', 'vanilla', 'vanilla_prefill') for role in ('small','large')]
        dec = [span(r, role, 'benchmark', 'execution', 'vanilla', 'decode') for role in ('small','large')]
        ax.barh(ys, pre, label='Prefill: GPU event intervals', color=COLORS[0])
        ax.barh(ys, dec, left=pre, label='Decode: GPU event intervals', color=COLORS[1])
        for i,(a,b) in enumerate(zip(pre,dec)):
            ax.text(a/2,i,f'{a:.1f}s',ha='center',va='center',color='white')
            ax.text(a+b/2,i,f'{b:.1f}s',ha='center',va='center',color='white')
        ax.set_yticks(ys, ['3B (points)', '8B (assists)'])
        ax.set_title(f"{r['reports']} reports | actual execution only")
        style(ax, 'Sum of forward GPU intervals (seconds)')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),fontsize=8,frameon=False)
    save(fig, '02_prefill_decode', '正式執行的 prefill／decode',
         '只取 stage=benchmark、phase=execution；兩模型各處理一次每個球員—報導組合。GPU event interval 包含 launch gaps，不能稱為純計算時間。此次答案很短，通常只有一個 decode forward；不要與長文生成直接類比。')

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.7), layout='constrained')
    for ax,r in zip(axs,runs):
        y=np.arange(2)
        for j,route in enumerate(('vanilla','kv')):
            vals=[group_latency(r,role,'profiling',route)*1000 for role in ('small','large')]
            bars=ax.barh(y+(j-.5)*.33, vals, height=.30, color=COLORS[j], label=route if route=='vanilla' else 'KV cr=0.8')
            ax.bar_label(bars, fmt='%.1f ms', padding=3, fontsize=9)
        ax.set_yticks(y,['3B','8B']); ax.set_xlim(0,210)
        ax.set_title(f"{r['reports']} reports | optimizer samples")
        style(ax, 'Backend request wall time / item (milliseconds)')
        ax.legend(loc='lower right',frameon=False)
    save(fig,'03_candidate_latency','候選方案：KV 在本次採樣較慢',
         '每模型每路徑兩個 request、合計 200 items（points 與 assists 各 100）；圖為總 request wall／總 items，不是每 token 吞吐。包含少量首批 trace 開銷與不同快取狀態，不能推論所有資料或 batch size 都如此。正式 execution 沒有 KV 請求。')

    # Shared timeline of the full run: resources are sampled rather than continuous.
    for r in runs:
        fig,axs=plt.subplots(3,1,figsize=(13,8),sharex=True,layout='constrained')
        samples=r['samples']; ts=np.array([s['elapsed_s']/60 for s in samples])
        for key,label,color in [('gpu_used_gib','GPU used',COLORS[0])]:
            axs[0].plot(ts,[s.get(key,np.nan) for s in samples],label=label,color=color)
        axs[0].axhline(max(s.get('gpu_total_gib',0) for s in samples),ls='--',color='#8a97a8',label='GPU total')
        axs[0].set_ylabel('GPU memory (GiB)'); axs[0].set_ylim(bottom=0)
        for key,label,color in [('process_rss_gib','Process-tree RSS (shared pages may repeat)',COLORS[0]),
                                ('process_pss_gib','Process-tree PSS',COLORS[1]),('cgroup_gib','Visible cgroup usage (incl. cache)',COLORS[2])]:
            axs[1].plot(ts,[s[key] for s in samples],label=label,color=color)
        axs[1].set_ylabel('CPU memory (GiB)'); axs[1].set_ylim(bottom=0)
        axs[2].plot(ts,[s.get('gpu_util_pct',np.nan) for s in samples],color=COLORS[0],lw=.8,label='nvidia-smi GPU utilization')
        axs[2].set_ylabel('GPU utilization (%)');axs[2].set_ylim(0,105);axs[2].set_xlabel('Minutes since run start')
        for stage,color in [('benchmark','#e7effa'),('reference','#e8f4ef')]:
            indices=[i for i,s in enumerate(samples) if s['stage']==stage]
            if indices:
                a,b=ts[min(indices)],ts[max(indices)]
                for ax in axs: ax.axvspan(a,b,color=color,zorder=-1)
                axs[0].text((a+b)/2,.98,stage,transform=axs[0].get_xaxis_transform(),ha='center',va='top',fontsize=9)
        for ax in axs:
            ax.grid(alpha=.15);ax.legend(loc='upper left',fontsize=8,frameon=False)
        axs[0].set_title(f"{r['reports']} reports | resource samples across the whole run")
        save(fig,f"04_resources_{r['reports']}",f"{r['reports']} 篇：顯存、RAM 與 GPU 使用率",
             '背景藍色為 benchmark、綠色為 reference。約一秒採樣會漏掉瞬間峰值；顯卡 utilization 不是 SM occupancy。RSS、PSS、cgroup 是不同口徑，不能相加；cgroup 包含 page cache 等，主機 RAM 可能含其他租戶。')

    r=runs[-1]
    fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    metrics=[('cache_file_load','cpu_wall_s','Load / deserialize (CPU thread-sum)'),
             ('cpu_to_cuda','cpu_wall_s','H2D call intervals (CPU)'),
             ('cpu_to_cuda','gpu_elapsed_s','H2D GPU event intervals'),
             ('cache_assembly','cpu_wall_s','Cache assembly (CPU, inclusive)'),
             ('prefetch_wait','cpu_wall_s','Prefetch wait (CPU)'),
             ('gpu_sync_wait','cpu_wall_s','Explicit sync wait (CPU)')]
    y=np.arange(len(metrics))
    for j,role in enumerate(('small','large')):
        vals=[span(r,role,'benchmark','profiling','kv',n,f) for n,f,_ in metrics]
        bars=axs[0].barh(y+(j-.5)*.33,vals,.30,label='3B' if role=='small' else '8B',color=COLORS[j])
        axs[0].bar_label(bars,fmt='%.2f',padding=2,fontsize=8)
    axs[0].set_yticks(y,[x[2] for x in metrics]);axs[0].invert_yaxis()
    style(axs[0],'Seconds, 200 KV items per model (overlapping spans)');axs[0].legend(frameon=False)
    for j,role in enumerate(('small','large')):
        k=(role,'benchmark','profiling','kv'); io=r['io'][k]
        vals=[io['file_bytes']/GIB,span(r,role,'benchmark','profiling','kv','cpu_to_cuda','tensor_bytes')/GIB,io['read_bytes']/GIB]
        bars=axs[1].barh(np.arange(3)+(j-.5)*.33,vals,.30,label='3B' if role=='small' else '8B',color=COLORS[j])
        axs[1].bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    axs[1].set_yticks(np.arange(3),['Cache file bytes\n(repeated loads included)','H2D tensor bytes','read_bytes in load spans\n(process I/O accounting)'])
    axs[1].invert_yaxis();style(axs[1],'GiB');axs[1].legend(frameon=False)
    fig.suptitle('728 reports | KV candidate profiling only')
    save(fig,'05_cache_transfer','KV 搬送：讀檔不等於 SSD 實體讀取',
         '左圖各量測區間可能包含或重疊，不可相加成總時間。右圖 cache 檔案量含重複載入，不是唯一 cache 大小；read_bytes 是載入區間的程序 I/O 計帳，全部為 0，符合 OS page cache 命中的情況，不能據此估 SSD 頻寬。')

    fig,axs=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for ax,r in zip(axs,runs):
        y=np.arange(4)
        for j,stage in enumerate(('benchmark','reference')):
            vals=[r['faults'].get((stage,role),{}).get('minor_faults',0) for role in ('runner','small','large','embedding')]
            bars=ax.barh(y+(j-.5)*.33,vals,.30,label=stage,color=COLORS[j])
            ax.bar_label(bars,labels=[f'{v/1000:.1f}k' for v in vals],padding=2,fontsize=8)
        major=sum(v.get('major_faults',0) for (st,_),v in r['faults'].items() if st in ('benchmark','reference'))
        ax.set_yticks(y,['Runner','3B','8B','Embedding']); ax.ticklabel_format(axis='x',style='sci',scilimits=(3,3))
        ax.set_title(f"{r['reports']} reports | major faults observed: {major:,}")
        style(ax,'Minor faults: sum of measured per-process deltas');ax.legend(frameon=False)
    save(fig,'06_linux_faults','Linux 缺頁：minor 與 major 分開解讀',
         '按 PID＋建立時間使用每次採樣的增量，不加總累計值，也不再疊加 thread/request 的重複計數。第一筆／最後一筆之外的活動未涵蓋，跨 stage 邊界增量歸給後一個樣本。minor fault 不等於讀 SSD；這不是 CUDA Unified Memory faults。')

    fig,axs=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for ax,r in zip(axs,runs):
        ts=sorted(r['traces'],key=lambda t:(t['role'],t['route']))
        y=np.arange(len(ts));left=np.zeros(len(ts))
        for key,label,color in [('kernel_union_s','Kernel activity union',COLORS[0]),('copy_only_s','Copy / memset, excluding kernel overlap',COLORS[1]),('no_recorded_gpu_event_s','No recorded GPU event in this window',COLORS[2])]:
            vals=[t[key]*1000 for t in ts];ax.barh(y,vals,left=left,label=label,color=color);left+=vals
        ax.set_yticks(y,[('3B' if t['role']=='small' else '8B')+' '+t['route'] for t in ts])
        ax.set_title(f"{r['reports']} reports | representative batch only");style(ax,'First-to-last GPU event window (ms)')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),fontsize=7,frameon=False)
    save(fig,'07_trace_intervals','代表 trace：GPU 區間不等於純 kernel 計算',
         '以 trace 中第一個至最後一個 GPU event 作窗口，對 kernels／copies 取時間聯集，排除重疊。橘色表示此窗口沒有記錄到 GPU 工作，不等同已證明 CPU 是瓶頸。Trace 本身有開銷，且 small_kv 來自 configuring、其他來自 optimizer profiling，不能外推全場。')
    fig,axs=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for ax,r in zip(axs,runs):
        q=r['quality']; diffs=[q['points_mismatches'],q['assists_mismatches']]
        agreements=[100*(1-v/q['rows']) for v in diffs]
        ax.barh(['Points (3B vs 8B)','Assists (8B vs 8B)'],agreements,color=COLORS[0],label='Agreement')
        ax.barh(['Points (3B vs 8B)','Assists (8B vs 8B)'],[100-v for v in agreements],left=agreements,color=COLORS[4],label='Disagreement')
        for i,(a,d) in enumerate(zip(agreements,diffs)):
            ax.text(2,i,f'{a:.2f}% agree | {d:,} / {q["rows"]:,} differ',va='center',color='white',fontsize=10)
        ax.set_xlim(0,100);style(ax,'Agreement with 8B silver (%)')
        ax.set_title(f"{r['reports']} reports | aligned by player + game ID")
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2,frameon=False)
    save(fig,'08_silver_agreement','答案差異：points 與 assists 分別核對',
         '由原始 CSV 以 name＋game_id 對齊，將數值 12 與 12.0 視為相同；此次鍵值唯一且無缺值。8B 參考答案也可能出錯，所以這是模型間一致性，不是人工標註準確率。差異明細另存 answer_differences.csv。')
    r=runs[-1]
    fig,ax=plt.subplots(figsize=(11,4.7),layout='constrained')
    y=np.arange(4)
    for j,role in enumerate(('small','large')):
        prepare=sum(q['elapsed_s'] for q in r['requests'] if q['role']==role and q['stage']=='benchmark' and q['path']=='/prepare_caches')
        vals=[prepare,span(r,role,'benchmark','end_to_end','prepare','cache_build_prefill'),
              span(r,role,'benchmark','end_to_end','prepare','cuda_to_cpu'),
              span(r,role,'benchmark','end_to_end','prepare','cache_file_save','cpu_wall_s')]
        bars=ax.barh(y+(j-.5)*.33,vals,.30,label='3B' if role=='small' else '8B',color=COLORS[j])
        ax.bar_label(bars,fmt='%.2f s',padding=3,fontsize=9)
    ax.set_yticks(y,['Prepare-cache request wall\n(inclusive total)','Cache-build prefill\n(GPU event intervals)','Cache D2H\n(GPU event intervals)','Cache save calls\n(CPU wall intervals)'])
    ax.invert_yaxis();ax.set_xlim(0,135);style(ax,'Seconds (different scopes; do not add these bars)')
    ax.set_title('728 reports | eager cache preparation before candidate selection');ax.legend(frameon=False)
    save(fig,'09_cache_build','先建立 cache 的成本：兩模型共約 203 秒',
         '728 篇各模型實際出現 728 個 cache_build_prefill；100 篇此次沒有新建 cache 的 span。Prepare request 包含其餘子工作，所以不可再加上 prefill／D2H／save。Cache save 是寫入呼叫時間，不代表資料已落到實體 SSD；這些 cache 有用於候選採樣，但正式 execution 最後沒有選 KV。')
    pdf.close()

    # Export grouped, auditable values. Missing GPU measurements remain None.
    overview=[];span_rows=[];request_rows=[];fault_rows=[];trace_rows=[];candidate_rows=[];memory=[];differences=[]
    for r in runs:
        m=r['manifest'];s=r['samples'];g=r['query'];n=r['reports']
        row=dict(reports=n,run_id=m['run_id'],result_rows=g['rows'],run_wall_s=m['total_wall_s'],
                 query_wall_s=r['components']['end_to_end'],execution_s=r['components']['execution'],
                 reference_wall_s=r['reference']['end_to_end'],silver_precision=g['precision'],silver_recall=g['recall'],silver_f1=g['f1_score'],
                 gpu_sample_peak_gib=max(x.get('gpu_used_gib',0) for x in s),process_rss_sample_peak_gib=max(x['process_rss_gib'] for x in s),
                 process_pss_sample_peak_gib=max(x['process_pss_gib'] for x in s),cgroup_sample_peak_gib=max(x['cgroup_gib'] for x in s),
                 cache_directory_bytes=m['cache_bytes_after_run'])
        overview.append(row)
        differences.extend(dict(reports=n,**d) for d in r['differences'])
        for stage in dict.fromkeys(x['stage'] for x in s):
            selected=[x for x in s if x['stage']==stage]
            mem=dict(reports=n,stage=stage,samples=len(selected))
            for field in ('gpu_used_gib','process_rss_gib','process_pss_gib','cgroup_gib','host_used_gib','host_total_gib','cgroup_limit_gib'):
                mem[field+'_peak']=max(x.get(field,0) for x in selected)
            mem['gpu_util_sample_mean_pct']=float(np.mean([x['gpu_util_pct'] for x in selected if 'gpu_util_pct' in x]))
            memory.append(mem)
        save_csv(out/f'resource_samples_{n}.csv',s)
        for key,v in r['aggregates'].items():
            role,stage,phase,route,name=key
            a=dict(reports=n,role=role,stage=stage,phase=phase,route=route,name=name,**v)
            if not v.get('gpu_elapsed_s_measured'): a['gpu_elapsed_s']=None
            span_rows.append(a)
        for x in r['requests']:
            stat=x.get('inference_stats') or {}
            request_rows.append(dict(reports=n,role=x['role'],stage=x['stage'],phase=x['phase'],route=x['route'],endpoint=x['path'],
                                     request_id=x['request_id'],elapsed_s=x['elapsed_s'],n_items=stat.get('n_items'),status=x['status']))
        for (stage,role),v in r['faults'].items():fault_rows.append(dict(reports=n,stage=stage,role=role,**v))
        trace_rows.extend(dict(reports=n,**t) for t in r['traces'])
        for role in ('small','large'):
            for route in ('vanilla','kv'):
                reqs=request_group(r,role,'profiling',route)
                candidate_rows.append(dict(reports=n,role=role,route=route,request_count=len(reqs),n_items=sum(x['inference_stats']['n_items'] for x in reqs),
                    mean_request_wall_per_item_s=group_latency(r,role,'profiling',route)))
    for name,rows in [('overview',overview),('phase_spans',span_rows),('requests',request_rows),('linux_faults',fault_rows),('trace_intervals',trace_rows),('candidate_latency',candidate_rows),('resource_stage_summary',memory),('answer_differences',differences)]:save_csv(out/(name+'.csv'),rows)

    full=runs[-1];o=overview[-1]
    pre=sum(span(full,role,'benchmark','execution','vanilla','vanilla_prefill') for role in ('small','large'))
    dec=sum(span(full,role,'benchmark','execution','vanilla','decode') for role in ('small','large'))
    h2d=sum(span(full,role,'benchmark','execution','vanilla','cpu_to_cuda') for role in ('small','large'))
    d2h=sum(span(full,role,'benchmark','execution','vanilla','cuda_to_cpu') for role in ('small','large'))
    rs={role:group_latency(full,role,'profiling','kv')/group_latency(full,role,'profiling','vanilla') for role in ('small','large')}
    summary=dict(schema_version=1,generated_utc=datetime.now(timezone.utc).isoformat(),overview=overview,
                 query_components=[dict(reports=r['reports'],**r['parts']) for r in runs],
                 actual_execution_gpu_intervals_s=dict(prefill=pre,decode=dec,h2d=h2d,d2h=d2h),candidate_kv_to_vanilla_latency_ratio=rs,
                 quality_alignment=[dict(reports=r['reports'],**r['quality']) for r in runs],
                 resource_stage_summary=memory,
                 provenance=dict(source='Downloaded results only; no new experiment',input_root=str(input_root),
                                 python=platform.python_version(),matplotlib=matplotlib.__version__,numpy=np.__version__,
                                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                 limitations=['One run per size; no repeated trials or confidence intervals','Silver reference is 8B, not human labels',
                              'No unprofiled baseline; instrumentation overhead unknown','100 and 728 cache states differ',
                              'CUDA event intervals include launch gaps; spans overlap',
                              'No Nsight Unified Memory faults, CPU utilization, or full-run kernel trace',
                              'Memory peaks are sampled; cgroup and host scopes differ from process RAM',
                              'Trace small_kv is configuring; other traces are optimizer profiling'])
    (out/'analysis_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    # Save actual chosen plans, rather than treating a listed fallback as an executed call.
    (out/'chosen_plans.json').write_text(json.dumps([dict(reports=r['reports'],plans=r['plans'],operator_buckets=r['buckets']) for r in runs],ensure_ascii=False,indent=2),encoding='utf-8')

    table='| 指標 | 100 篇 | 728 篇 |\n|---|---:|---:|\n'
    for title,key,fmt in [('球員—報導結果列數','result_rows',',.0f'),('整輪含參考答案（分鐘）','run_wall_s','.2f'),('benchmark 查詢（分鐘）','query_wall_s','.2f'),
                          ('正式 execution（分鐘）','execution_s','.2f'),('GPU 採樣峰值（GiB）','gpu_sample_peak_gib','.2f'),
                          ('程序 RSS 加總峰值（GiB）','process_rss_sample_peak_gib','.2f'),('可見 cgroup 使用峰值（GiB）','cgroup_sample_peak_gib','.2f'),
                          ('對 8B silver 的 F1','silver_f1','.4f')]:
        vals=[x[key]/60 if '分鐘' in title else x[key] for x in overview]
        table+=f'| {title} | {format(vals[0],fmt)} | {format(vals[1],fmt)} |\n'
    findings=[
      f"完整 728 篇產生 {o['result_rows']:,} 列球員—報導組合，不是只問 728 個問題：每列都要抽 points 與 assists。正式 execution 實際呼叫 3B vanilla 抽 points、8B vanilla 抽 assists，各 {o['result_rows']:,} items。計畫中雖保留 points 的 8B fallback，但 operator/backend 紀錄顯示這次沒有收到執行資料。",
      f"728 篇的 benchmark 查詢 {o['query_wall_s']/60:.2f} 分鐘，其中 execution {o['execution_s']/60:.2f} 分鐘（{o['execution_s']/o['query_wall_s']:.1%}）；cache preparation {full['parts']['Cache preparation']:.1f} 秒、候選採樣 {full['parts']['Candidate sampling']:.1f} 秒、optimizer solve {full['parts']['Optimizer solve']:.1f} 秒。整輪 {o['run_wall_s']/60:.2f} 分鐘包含另外的 8B silver reference，不能全算成 benchmark 成本。",
      f"正式 execution 的 forward GPU intervals：prefill 合計 {pre:.1f} 秒、decode {dec:.1f} 秒，prefill 佔兩者 {pre/(pre+dec):.1%}。顯式 H2D＋D2H GPU intervals 合計僅 {h2d+d2h:.2f} 秒。就目前紀錄，時間主要落在模型 forward／生成路徑，沒有證據顯示正式執行被顯式 PCIe 搬送主導；但不能進一步聲稱是純 GPU 算力瓶頸。",
      f"候選採樣中，3B 的 KV/vanilla 每 item 時間比為 {rs['small']:.2f}，8B 為 {rs['large']:.2f}，即 KV 反而較慢。正式 execution 兩輪都沒有 KV 呼叫。KV 省下的 prefill 並未使這次端到端候選請求更快；讀檔／反序列化、多筆 tensor 搬送與呼叫間隙是值得後續隔離測量的原因，尚未證明個別因果。",
      f"728 篇在選定最終算子之前，已為兩模型各建立 728 篇的 cache，prepare request 共 {full['parts']['Cache preparation']:.1f} 秒，佔 benchmark 查詢 {full['parts']['Cache preparation']/o['query_wall_s']:.1%}。其中 cache-build prefill 的 GPU intervals 為 3B {span(full,'small','benchmark','end_to_end','prepare','cache_build_prefill'):.1f} 秒、8B {span(full,'large','benchmark','end_to_end','prepare','cache_build_prefill'):.1f} 秒。這些 cache 確實供候選採樣使用，但本輪正式 execution 沒用到；之後可研究 cache 建立時機，不能直接把這段成本刪掉當作已取得的加速。",
      f"728 篇整卡顯存採樣峰值 {o['gpu_sample_peak_gib']:.2f} GiB；程序 RSS 加總峰值 {o['process_rss_sample_peak_gib']:.2f} GiB（含模型載入期），benchmark 階段 RSS 加總峰值為 {max(s['process_rss_gib'] for s in full['samples'] if s['stage']=='benchmark'):.2f} GiB；可見 cgroup 峰值 {o['cgroup_sample_peak_gib']:.2f} GiB。這次 batch=1 有餘裕，不能由此保證 24GB 顯卡或更大 batch 能跑；RSS 可能重複計共享頁，cgroup 包含 cache，不能直接當模型本身 RAM。",
      "兩輪 KV 載入區間的程序 read_bytes 都是 0，代表有邏輯讀檔但沒有記錄到該區間的實體讀取計帳，符合 page cache 命中情況。因此現在不能把 cache_file_load 說成 SSD→CPU 的硬體傳輸時間，也無法估 SSD 頻寬。",
      f"728 篇對 8B vanilla silver 的 precision／recall／F1 都為 {o['silver_f1']:.4f}（約 {o['silver_f1']:.2%}）。逐欄對齊後，points 有 {full['quality']['points_mismatches']} 列不同、assists 有 {full['quality']['assists_mismatches']} 列不同，無缺值。這是列級結果一致性，不是人工標註正確率；precision=recall 是兩邊唯一結果列數相同，不代表兩個指標被獨立驗證。",
      "代表 trace 的 CUDA kernels 之間有明顯空隙，尤其 KV；這支持調查 batch=1 的框架／launch／同步與 cache 整理成本。Trace 是有額外開銷的首批樣本，且沒有 CPU 使用率紀錄，不能直接宣稱整場 CPU-bound 或用這四批推估全場 GPU idle 比例。",
      f"資料量由 {overview[0]['result_rows']} 列增至 {o['result_rows']} 列（{o['result_rows']/overview[0]['result_rows']:.2f} 倍），正式 execution 時間增加 {o['execution_s']/overview[0]['execution_s']:.2f} 倍；每列約 {overview[0]['execution_s']/overview[0]['result_rows']:.3f}→{o['execution_s']/o['result_rows']:.3f} 秒。這兩點顯示目前正式執行大致隨結果列數增加，但單次兩點不能建立普遍的 scaling law。"
    ]
    caveats=[
      '本次只測上游固定 Rotowire 的一個 join＋兩個抽取查詢，並非整篇論文的所有 benchmark。3B/8B 單張 RTX 6000 Ada、BF16、batch=1，不能直接與原論文硬體／模型設定比較。',
      '100 篇沿用已存在的 cache；728 篇在查詢中實際建立兩模型各 728 份 cache。整個 cache 目錄大小也含先前實驗，不能當本輪新增量。',
      'phase=profiling 是 optimizer 的候選資料採樣；run action=profile 才是這次打開的儀器量測模式，兩者名稱相近但含義不同。',
      'Prefill input_tokens 依實際 forward 計算；cached_query_prefill 只計補算，不等於整篇報導重新 prefill。生成紀錄 kind=generation 與 span name=generation 是同段工作，不能加兩次。',
      '每秒資源採樣可能漏峰值；fault/I/O 增量排除 first_sample，跨階段增量依樣本結束 stage 歸類。數值為觀測範圍內，不是整個程序生命週期總量。',
      f"主機 RAM 約 {max(s['host_total_gib'] for s in full['samples']):.1f} GiB、可見 cgroup limit 約 {max(s['cgroup_limit_gib'] for s in full['samples']):.1f} GiB，與先前租借介面數字不同；只報告觀測口徑，不視為租用保證容量。",
      '沒有 Nsight Unified Memory faults，沒有未開 profiling 的同條件 baseline，沒有重複試驗；不能報 profiling overhead、統計顯著性或對論文的 speedup。8B silver 僅是參考流程，cache／查詢設定與執行順序不同，不能直接當公平效能 baseline。'
    ]
    text='# Stretto 單卡 3B／8B：100 與 728 篇結果分析\n\n'+table+'\n## 主要發現\n\n'+'\n\n'.join(f'{i+1}. {s}' for i,s in enumerate(findings))
    text+='\n\n## 圖表\n\n'
    for c in charts:text+=f"### {c['title']}\n\n![{c['title']}]({c['name']}.png)\n\n{c['note']}\n\n"
    text+='## 量測限制\n\n'+'\n\n'.join('- '+s for s in caveats)
    text+='\n\n## 下一步\n\n先用這份結果建立 baseline。若再租 GPU，優先在相同 warm-cache 條件下比較 run/profile 的重複試驗，並把 vanilla／KV 的同一批問題固定下來；再獨立測試 cache load、tensor copies 與 generation，以驗證 KV 較慢的原因。Nsight 與 batch-size sweep 是後續獨立實驗，不要混成這次已觀測的結論。\n\n圖表可用 charts.pdf 分享，PNG 放投影片，SVG 保留向量。overview.csv、phase_spans.csv、requests.csv、resource_samples_*.csv、linux_faults.csv、trace_intervals.csv 與 analysis_summary.json 保存衍生數據；原始結果不修改。\n'
    (out/'report.md').write_text(text,encoding='utf-8')
    # One self-contained offline HTML: inline SVG, no CDN or server dependency.
    table_html='<table><thead><tr><th>指標</th><th>100 篇</th><th>728 篇</th></tr></thead><tbody>'
    for line in table.splitlines()[2:]:
        cells=[s.strip() for s in line.strip('|').split('|')]
        table_html+='<tr>'+''.join('<td>'+html.escape(s)+'</td>' for s in cells)+'</tr>'
    table_html+='</tbody></table>'
    cards=''
    for c in charts:
        svg=(out/(c['name']+'.svg')).read_text(encoding='utf-8');svg=svg[svg.index('<svg'):]
        cards+=f'<section class="chart" id="{c["name"]}"><h2>{c["title"]}</h2><p>{c["note"]}</p>{svg}</section>'
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stretto 實驗結果｜100 / 728 篇</title><style>
body{margin:0;background:#edf2f7;color:#172b45;font:16px/1.8 "Microsoft JhengHei",system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:32px 24px}h1{line-height:1.3}h2{font-size:23px}section,.card{background:white;border-radius:14px;padding:24px;margin:22px 0;box-shadow:0 2px 8px #142b4510}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:10px 14px;border-bottom:1px solid #e2e8f0}th{background:#f1f5f9}svg{width:100%;height:auto}li{margin:14px 0}.tag{display:inline-block;background:#e1eef9;color:#20518b;border-radius:20px;padding:3px 12px;margin:0 8px 6px 0}nav{position:sticky;top:0;background:#edf2f7ed;padding:10px 0;z-index:9}nav a{color:#2462a0;margin-right:18px;text-decoration:none}.note{color:#52677d;font-size:14px}@media print{nav{display:none}section{break-inside:avoid}body{background:white}main{padding:0}}</style>
<main><div class="tag">RTX 6000 Ada · 3B + 8B · BF16</div><div class="tag">batch=1 · 固定 Rotowire 查詢</div><div class="tag">離線分析 · 未新增推論</div>
<h1>Stretto：時間花在哪裡？<br>100 篇與 728 篇實驗報告</h1>
<p>正式執行選用了 vanilla。KV 候選有被量測，但沒有進入最終大量推論。下方將正式執行、候選採樣與參考答案分開解讀。</p>
<nav><a href="#overview">總覽</a><a href="#findings">發現</a><a href="#01_wall_time">圖表</a><a href="#limits">限制</a><a href="charts.pdf">下載圖表 PDF</a></nav>
'''
    page+='<section id="overview"><h2>量測總覽</h2>'+table_html+'</section><section id="findings"><h2>主要發現與依據</h2><ol>'+''.join('<li>'+html.escape(s)+'</li>' for s in findings)+'</ol></section>'+cards
    page+='<section id="limits"><h2>哪些結論還不能下？</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in caveats)+'</ul></section><p class="note">圖表使用秒、分鐘與 GiB；測試日期 2026-10-22。衍生表與 JSON 在相同資料夾。此 HTML 內嵌圖表，不需連網。</p></main></html>'
    (out/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    suites=sorted(args.results_root.glob('suites/*/suite.json'))
    complete=[read_json(s) for s in suites if read_json(s).get('status')=='completed']
    if not complete: raise SystemExit('No completed suite')
    stages=[s for s in complete[-1]['stages'] if s['name'] in ('100-profile','728-profile')]
    assert len(stages)==2 and all(s['status']=='completed' for s in stages)
    assert union_length([(0,2),(1,3),(4,5)])==4
    runs=[load_run(args.results_root/Path(s['results']).name) for s in stages]
    generate_report(runs,args.output,args.results_root)
    input_files={}
    for r in runs:
        for f in sorted(r['path'].rglob('*')):
            if not f.is_file(): continue
            digest=hashlib.sha256()
            with f.open('rb') as handle:
                for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
            input_files[f.relative_to(args.results_root).as_posix()]={'bytes':f.stat().st_size,'sha256':digest.hexdigest()}
    inventory=args.results_root.parent/'remote-inventory.json'
    if inventory.exists():
        original=read_json(inventory)['files']
        assert all(original.get(k)==v for k,v in input_files.items()), 'Input changed since backup verification'
    (args.output/'analysis_inputs.json').write_text(json.dumps(input_files,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
