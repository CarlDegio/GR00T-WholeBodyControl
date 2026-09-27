from pathlib import Path
import json,csv,html,zipfile,hashlib
from collections import Counter
from PIL import Image,ImageDraw,ImageFont
import cv2
R=Path('/home/user/Project/GR00T-WholeBodyControl');B=R/'demo_video/2026-09-26_cover';p=json.load(open(B/'edit/selection.json'));fav=set(sum(p['grid16'],[]));clips=p['clips'];kind={'navigation':'移动','scan':'环视','alignment':'对齐','manipulation':'操作'}
# Record alternate upload names without replacing existing canonical sources.
library=R/'demo_video/2026-09-25/assets/manifest.json';m=json.load(open(library));report=json.load(open(B/'edit/import_report.json'))
for entry in report:
 v=next(v for v in m if v['order']==entry['order']);alias=Path(entry['attachment']).name
 if alias!=v['original_name']:
  v['aliases']=sorted(set(v.get('aliases',[])+[alias]))
library.write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n')
with (B/'selection.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.writer(f);w.writerow(['片段','任务','类别','内容','源素材编号','原片起点秒','原片终点秒','倍速','16宫格入选','导出文件'])
 for c in clips:w.writerow([c['id'],c['task'],kind[c['kind']],c['label'],c['source'],c['start'],c['end'],c['speed'],'是' if c['id'] in fav else '否',c['path']])
font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',22)
contact=Image.new('RGB',(1920,5*264),'#101820');d=ImageDraw.Draw(contact)
(B/'review/thumbs').mkdir(exist_ok=True)
for c in clips:
 cap=cv2.VideoCapture(str(B/c['path']),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);cap.set(cv2.CAP_PROP_POS_MSEC,2300);ok,im=cap.read();assert ok;cap.release()
 pic=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));pic.resize((640,360)).save(B/'review/thumbs'/f"{c['id']:02d}.jpg",quality=88)
 x=(c['id']-1)%5*384;y=(c['id']-1)//5*264;contact.paste(pic.resize((384,216)),(x,y));star='★' if c['id'] in fav else ''
 d.text((x+7,y+218),f"{c['id']:02d} {c['task']} {kind[c['kind']]} {c['speed']:g}× {star}",font=font,fill='white')
contact.save(B/'selected_25_contact_sheet.jpg',quality=93)
readme='''# Demo 封面 · 2026-09-26

两版均为 1920×1080、60 fps、5.000 秒、H.264，无音频。

- `exports/cover_25_5s.mp4`：3×3 保持 1 秒 → 1 秒连续缩放为 5×5 → 播放 3 秒。
- `exports/cover_16_5s.mp4`：2×2 保持 1 秒 → 1 秒连续缩放为 4×4 → 播放 3 秒。
- 第 2–3 秒标题由模糊渐显，中心两行；第 3–5 秒清晰保持。
- `clips/` 为全部 25 条实际入选片段，各 5 秒，与封面使用的剪辑和倍速一致。
- `gallery.html` 可逐条播放，★ 表示同时入选 16 条版。`selection.csv` / `edit/selection.json` 记录原片起止、倍速、来源哈希及宫格位置。

标题采用当前论文主稿 `/home/user/Project/Overleaf/6a925da904325a8d60afac0b/root.tex`：
Beyond Arrival: Geometric Alignment for Agentic Humanoid Loco-manipulation

25 条构成：指定移动 4 条、T4 环视 1 条；操作 10 条、对齐 10 条。
T1/T3 各操作 3 + 对齐 2；T2/T4 各操作 2 + 对齐 3。
16 条版是其中的子集，保留指定移动与环视 5 条，另含操作 7 条、对齐 4 条。
同一原片可使用不同、不重叠的动作阶段；并非 25 个独立试验。所有片段连续截取，无动作循环或合成补帧。

21 个本次附件：新增 8 条，13 条哈希与库内内容一致；其中两条仅文件名不同。
同时把此前 T4 长程 demo 原片补存入同一素材库，用于指定导航与环视。
素材库：`../2026-09-25/assets/originals/`，索引为 `../2026-09-25/assets/manifest.json`。
库内现有 22 个唯一原始文件（含先前标记 superseded 的历史文件）。本封面不修改实验标注或成功率。

复现：`.venv_inference/bin/python demo_video/2026-09-26_cover/edit/select_and_render.py`，随后运行 `edit/compose.py`、`edit/package.py`、`edit/verify.py`。
'''
(B/'README.md').write_text(readme)
head='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Beyond Arrival · Demo 封面与入选片段</title><style>body{background:#10151c;color:#eef4fa;font:16px system-ui;margin:0;padding:28px;max-width:1500px;margin:auto}h1{font-size:30px}p{color:#aebdcc;line-height:1.7}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px}article{background:#19232f;padding:12px;border-radius:12px}video{width:100%;display:block;border-radius:6px}h2{font-size:19px;margin:12px 0}a{color:#81dccc}.badge{color:#81dccc}small{color:#aebdcc}</style><h1>Beyond Arrival · Demo 封面</h1><p>两版均为 5 秒 · 1080p · 60 fps。第 1–2 秒缩放，第 2–3 秒标题由模糊渐显。★ 为优选 16 条。</p><div class="grid">'''
for n,label in [(25,'25 条版 · 3×3 → 5×5'),(16,'16 条版 · 2×2 → 4×4')]:head+=f'<article><h2>{label}</h2><video controls playsinline preload="metadata" poster="exports/cover_{n}_5s.jpg" src="exports/cover_{n}_5s.mp4"></video></article>'
head+='</div><h1>全部 25 条入选片段</h1><p><a href="selection.csv">下载剪辑清单</a> · 每条 5 秒，已应用成片倍速；时间范围为原视频位置。</p><div class="grid">'
for c in clips:
 star='★ ' if c['id'] in fav else ''
 head+=f'<article><h2>{star}{c["id"]:02d} · {c["task"]} {html.escape(c["label"])}</h2><video controls playsinline preload="none" poster="review/thumbs/{c["id"]:02d}.jpg" src="{c["path"]}"></video><p>{kind[c["kind"]]} · {c["speed"]:g}× · 原片 {c["start"]:.1f}–{c["end"]:.1f} s<br><small>源素材 #{c["source"]:02d}</small></p></article>'
(B/'gallery.html').write_text(head+'</div></html>')
print('index, gallery, contact sheet, README written')
