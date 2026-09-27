"""Combine manual review evidence; does not alter any video."""
import html
import json
from pathlib import Path

B = Path(__file__).resolve().parent
rows = []
def add(cid, windows, position, kind, decision, reason, screenshot, sample=None, boxes=None):
    rows.append(dict(id=cid, windows=windows, position=position, type=kind,
                     recommendation=decision, reason=reason, screenshot=str(screenshot),
                     example_time=sample if sample is not None else windows[0][0],
                     bbox_xyxy=boxes or []))

for r in json.loads((B/'mosaics/findings.json').read_text())['findings']:
    add(r['id'], [r['time_seconds']], r['position'], r['type'], r['recommendation'],
        r['reason'], r['screenshot'], r['example_time'], r['bbox_xyxy'])
preview = json.loads((B/'preview/review.json').read_text())
for r in preview['findings']:
    kind = r['kind']
    reason = r['reason']
    if r['id'] in ['P1','P8','P9','P10']:
        kind = '纸箱二维码及相邻码区'
        reason = reason.replace('两个二维码', '密集方格二维码及旁边圆形图案标签')
        reason = reason.replace('双码', '两块码/标签区域')
        reason += ' 密集方码确认有二维码外观；相邻圆形图案未验证编码类型，一并遮挡属于保守处理。'
    add(r['id'], [r['time']], r['panel'], kind, r['decision'], reason,
        r['screenshot'], r['example_time'], [r['bbox']])
t2 = json.loads((B/'t2/face_findings.json').read_text())
rows.extend(t2['findings'])
add('T2-Q01',[[71,118]],'主画面：办公桌旁箱子、冰箱旁搬运箱的长侧面',
    '纸箱二维码','必须打码','长侧面密集方码随相机移动、纸箱旋转反复可见；79.3秒前景较大。该区间是保守跟踪窗，并非每帧都露出码面。旁边圆形标签可随码区一并遮挡。',B/'t2/annotated_frame_79.3.jpg',79.3)
add('T2-Q02',[[73.8,78.8]],'HEAD：上缘/左上经过的纸箱长侧面',
    '纸箱二维码','建议打码','已知长侧面方码区进入小窗，短暂出框或遮挡时按可见性跟踪。',B/'t2/head_all_2.jpg',76.3)
add('T2-Q03',[[72.5,76.1]],'CHEST：右侧叠箱的长侧面',
    '纸箱二维码','建议打码','转向扫过叠箱，两块白框中包含已知密集方码；小窗需要单独处理。',B/'t2/chest_3.jpg',75.3)
add('T2-Q04',[[77,79.5],[86.1,89.6],[92.1,92.7]],'CHEST：桌椅后/冰箱旁箱体',
    '疑似纸箱码区','保守建议','77–79.5秒是小背景已知码区；后两段主要为斜侧面、短侧面图案，不能确认清晰二维码，仅建议跟随源片已知方码核查。',B/'t2/chest_5.jpg',88.3)
add('T2-N02',[[90,106.8],[107,118]],'HEAD：搬运箱短侧面、箱内、袋子',
    '排除项','非目标','90–106.8秒主要显示卡通图案和圆形包装标志，107秒后主要看箱内；不能将这些整段报告为持续二维码。只在长侧面方码真实转入时处理。',B/'t2/code_type_comparison.jpg',104.3)

t4 = [
 ('F01',119.15,119.45,'保守建议','左侧桌后局部脸；16×扫描短暂出现。','119.267'),
 ('F02',124.05,124.70,'保守建议','蹲下人员侧脸；小而模糊但确有面部露出。','124.233'),
 ('F03',141.30,143.20,'必须打码','蹲下人员侧脸持续出现，142.5秒脸部轮廓较明确；143秒附近开始转背。','142.500'),
 ('F04',151.35,152.30,'必须打码','转向扫过桌后正脸，期间短暂出画后又进入；另有椅侧小侧脸候选。','151.967'),
 ('F05',154.60,155.30,'必须打码','桌后正脸从CHEST左侧经过，154.7秒附近相对清楚。','154.767'),
 ('F06',155.95,156.20,'必须打码','快速转头再次扫过正脸，仅约0.2秒。','156.100'),
 ('F07',156.65,159.30,'必须打码','桌后正脸从右侧移到左侧；157.5–158.5秒较稳定。','157.500'),
 ('F08',164.20,175.10,'必须打码','坐地/蹲下人员从椅侧露出面部，持续可见，间有低头及椅背遮挡。','168.500'),
 ('F09',175.60,176.25,'保守建议','再次从椅背左侧露出局部侧脸，较模糊；建议与前段合并164.20–176.25秒跟踪。','164.500'),
]
for cid,a,z,level,reason,frame in t4:
    # F09's boundary sheet is more representative than an earlier raw frame.
    shot = B/'t4'/('boundary_02.jpg' if cid=='F09' else f'chest_{frame}.png')
    add('T4-'+cid,[[a,z]],'右中CHEST小窗', '人脸',level,reason,shot,
        175.8 if cid=='F09' else float(frame))
add('T4-Q01',[[151.45,152.25],[154.60,154.85],[156,156.2],[156.65,157.35]],
    'CHEST：桌面纸箱堆的标签区域','疑似纸箱码区','保守建议',
    '纸箱标签过小且模糊，未确认清晰二维码；只建议核对并遮已知码区，不应把所有箱面图案升级为二维码。',B/'t4/chest_151.967.png',151.967)

rows.sort(key=lambda r:(r['windows'][0][0],r['id']))
for r in rows:
    assert all(0 <= a < z <= 179.567 for a,z in r['windows']), r
    assert Path(r['screenshot']).is_file(), r['screenshot']
identity=json.loads((B/'video_identity.json').read_text())
payload=dict(video=identity, notes=[
    '完整成片时间，单位秒；宫格行列均从左上起数。',
    '建议窗包含短时遮挡和出框；实际打码应跟踪可见目标，不能整块小窗永久盖住。',
    '这是密集抽帧人工审查，未对10774帧逐帧人工标注，不保证瞬时露出绝无遗漏。',
    '不识别人身份，不解码二维码；不因无法扫码而断言码区无须处理。',
    '同款箱长侧面密集方码与邻接圆形图案标签建议一起遮；短侧面卡通/圆形商标不自动视为二维码。',
], findings=rows)
(B/'all_findings.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')

def tc(t):
    return f'{int(t//60):02d}:{t%60:05.2f}'
def window(r):
    return '；'.join(f'{tc(a)}–{tc(z)}' for a,z in r['windows'])
def rel(path):
    return str(Path(path).relative_to(B))

intro = '''以 Full_demo_combined_v3.mp4 为基准，时长 02:59.57，1920×1080 / 60fps。本次只审查，未修改视频。

全部时间对应整片。下表是建议处理窗口，不表示窗口内每一帧都清楚露出目标；遮挡应跟随实际可见的人脸或码区。宫格行列从左上角起数。截图框仅标示示例位置，不能直接用作整段固定遮罩。

人脸分为明确建议处理与小尺寸/裁切/运动模糊的保守项。纸箱密集方格码确认具有二维码外观；相邻白框内的圆形图案未验证编码类型，一并处理属于保守扩大。箱子短侧面的卡通图案和圆形商标不能直接算作二维码。
'''
md=['# 全片打码审查清单\n',intro,'\n[带视频定位与截图的审查页](review.html) · [结构化清单](all_findings.json)\n']
for heading, filt in [('建议处理与保守候选',lambda r:r['recommendation'] not in ['非目标','无需按人脸打码']),('核查后排除的目标',lambda r:r['recommendation'] in ['非目标','无需按人脸打码'])]:
    md += ['\n## '+heading+'\n','|编号|完整视频时间|位置|判断|原因与证据|','|---|---|---|---|---|']
    for r in filter(filt,rows):
        reason=r['reason'].replace('|','／')
        md.append(f"|{r['id']}|{window(r)}|{r['position']}|{r['type']}；{r['recommendation']}|{reason} [截图]({rel(r['screenshot'])})|")
md += ['\n## 覆盖与限制\n',
       '- 封面、片尾：每0.25秒抽帧并补末帧，逐宫格放大，30张成片样本。',
       '- 四宫格：每0.5秒覆盖，共81帧、324分区；人脸边界补0.1/0.25秒复查。OVMM每0.5秒20帧。',
       '- T2长任务：主画面、HEAD每1秒，CHEST每0.5秒；重点扫描与出入画边界加密到0.1秒，另做8张箱码交叉核对。',
       '- T4长任务：120个0.5秒样本，16×区间157个不大于0.1秒样本，26个边界样本；原帧和小窗放大交叉检查。',
       '- OVMM（00:45.30–00:55.30）未发现真实人脸或纸箱二维码。T4长任务的主画面及HEAD未发现明确目标。',
       '- 这是密集抽样审查，不是逐帧跟踪完成稿。实际遮挡时建议在露脸/露码的进入、离开边界加约0.1–0.2秒余量，并复看快速扫描、边缘及末帧停留。',
       '\n## 分段证据\n',
       '- [首尾宫格详细说明](mosaics/notes.md)',
       '- [四宫格及OVMM详细说明](preview/review.md)',
       '- [T2码区与包装图标对照](t2/code_crosscheck.md)',
       '- [T4小窗详细说明](t4/notes.md)',
       '\n视频SHA-256：`'+identity['sha256']+'`\n']
(B/'打码审查清单.md').write_text('\n'.join(md))

cards=[]
for r in rows:
    excluded=r['recommendation'] in ['非目标','无需按人脸打码']
    cls='excluded' if excluded else ('conservative' if '保守' in r['recommendation'] else 'required')
    buttons=' '.join(f'<button data-seek="{a:.6f}">{tc(a)}–{tc(z)}</button>' for a,z in r['windows'])
    cards.append(f'''<article class="{cls}" data-kind="{'face' if '脸' in r['type'] else 'code'}">
    <div><b>{html.escape(r['id'])} · {html.escape(r['type'])}</b><span>{html.escape(r['recommendation'])}</span></div>
    <p class="times">{buttons}</p><p><strong>{html.escape(r['position'])}</strong></p>
    <p>{html.escape(r['reason'])}</p><a href="{html.escape(rel(r['screenshot']))}" target="_blank"><img loading="lazy" src="{html.escape(rel(r['screenshot']))}" alt="{html.escape(r['id'])} 证据截图"></a>
    <button data-seek="{r['example_time']}">定位示例 {tc(r['example_time'])}</button></article>''')
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Demo 打码审查</title>
<style>body{margin:0;background:#101722;color:#e7edf4;font:16px/1.65 system-ui}header,main{max-width:1440px;margin:auto;padding:22px}h1{font-size:28px}p{margin:9px 0}a{color:#74dbea}video{width:100%;max-height:50vh;background:black}.viewer{position:sticky;top:0;background:#101722;z-index:2;border-bottom:1px solid #35465c;padding:8px 20px;display:grid;grid-template-columns:minmax(400px,1000px) 1fr;gap:15px}.viewer p{font-size:14px}button{background:#22394d;color:#fff;border:1px solid #486178;border-radius:5px;padding:6px 9px;margin:3px;cursor:pointer}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}article{background:#192332;border:1px solid #35465c;border-left:4px solid #f17872;padding:15px;border-radius:8px}article.conservative{border-left-color:#e5b967}article.excluded{border-left-color:#71849b}article span{float:right;color:#b7c9d9}article img{width:100%;max-height:350px;object-fit:contain;background:#080d14}article p{font-size:14px}.hidden{display:none}@media(max-width:850px){.grid{grid-template-columns:1fr}.viewer{position:static;grid-template-columns:1fr}}</style>
<header><h1>Demo 全片打码审查</h1><p>基准：Full_demo_combined_v3.mp4 · 02:59.57 · 审查报告，视频尚未打码。</p>
<p>时间按钮可定位整片；点击截图看原图。红边为明确建议处理，黄边为小脸、裁切脸或疑似码区的保守项，灰边为核查后排除项。建议窗不是整段固定遮罩；仅跟踪目标可见区域。</p>
<p>密集抽样人工审查，非逐帧无遗漏保证。箱子长侧面的方格码与相邻圆形标签可一起遮；短侧面卡通/商标不自动算二维码。</p>
<p><a href="打码审查清单.md">完整文字清单</a> · <a href="all_findings.json">JSON清单</a></p></header>
<section class="viewer"><video id="video" controls preload="metadata" src="../exports/Full_demo_combined_v3.mp4"></video><div><p id="location">选择一个时间段定位视频</p><button data-filter="all">全部</button><button data-filter="face">人脸</button><button data-filter="code">纸箱码区</button><button data-filter="required">明确建议</button><button data-filter="conservative">保守候选</button><button data-filter="excluded">排除项</button><p>宫格行列从左上起算。示例框不能直接用作全段固定遮罩；主画面、HEAD、CHEST各自处理。</p></div></section><main class="grid">'''+''.join(cards)+'''</main>
<script>const v=document.getElementById('video');document.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;if(b.dataset.seek!==undefined){v.currentTime=Number(b.dataset.seek);v.pause();document.getElementById('location').textContent='已定位 '+Number(b.dataset.seek).toFixed(3)+' 秒；按播放检查前后画面';}if(b.dataset.filter){const f=b.dataset.filter;document.querySelectorAll('article').forEach(a=>a.classList.toggle('hidden',!(f==='all'||a.dataset.kind===f||a.classList.contains(f))));}});</script></html>'''
(B/'review.html').write_text(page)
print(f'Wrote {len(rows)} findings, HTML review page, Markdown and JSON; all screenshot paths validated.')
