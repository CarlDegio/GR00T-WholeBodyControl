"""Static, explicitly exploratory nearfield figures and reproducibility report."""
from pathlib import Path
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

from nearfield_t2_20260921 import OUT, ROOT, read, write_json, sha
from metrics import neighbor_distances

INK='#223949';GRAY='#667d8c';GREEN='#178a78';RED='#d45c58';GOLD='#d6b972'
FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':FONT.get_name(),'font.size':11,'axes.labelcolor':INK,
                     'text.color':INK,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'svg.fonttype':'path','axes.unicode_minus':False})


def save(fig,name):
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=190,facecolor='white',bbox_inches='tight')
    plt.close(fig)


def main():
    (OUT/'figures').mkdir(exist_ok=True)
    rows=read(OUT/'results/approximate_paired_scores.json');stats=read(OUT/'results/approximate_statistics.json')
    refs=np.load(OUT/'references/head_reference_features.npz')['spatial']
    loo,_=neighbor_distances(refs,refs,exclude=np.arange(len(refs)))
    q25,q50,q75=np.quantile(loo,[.25,.5,.75])
    write_json(OUT/'results/training_leave_one_out.json',dict(n=179,q25=float(q25),median=float(q50),q75=float(q75),
                   definition='Each training head first frame vs nearest 3 OTHER training episodes; same spatial descriptor'))
    x=np.linspace(0,1,301);sig=1/(1+np.exp(-10*(x-.5)));blend=(sig-sig[0])/(sig[-1]-sig[0])
    ymax=max(r['pre_spatial'] for r in rows)*1.07
    ymin=max(0,min(float(q25),min(r['post_spatial'] for r in rows))*.85)
    fig,axs=plt.subplots(1,3,figsize=(17.8,7.8),sharey=True,gridspec_kw={'width_ratios':[1.25,1,1]})
    fig.subplots_adjust(left=.068,right=.985,top=.76,bottom=.26,wspace=.17)
    for ax,group,title in zip(axs,['combined','V2','V3'],['保留样本合并','第二版：仅保留成功','第三版：全部记录中的可配对样本']):
        rr=[r for r in rows if group=='combined' or r['version']==group];s=stats[group]
        ax.axhspan(q25,q75,color=GOLD,alpha=.22,zorder=0)
        ax.axhline(q50,color='#b39851',ls=':',lw=1.5,zorder=1)
        ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
        for r in rr:
            a,b=r['pre_spatial'],r['post_spatial'];c=GREEN if b<a else RED
            ys=a+(b-a)*blend
            assert np.allclose(ys[[0,-1]],[a,b],rtol=0,atol=1e-15)
            ax.plot(x,ys,color=c,lw=7,alpha=.25,solid_capstyle='round',zorder=2)
            ax.scatter([0,1],[a,b],color=c,s=29,alpha=.72,linewidths=0,marker='o' if r['version']=='V2' else 's',zorder=3)
        for xx,key in enumerate(['pre_spatial','post_spatial']):
            v=np.median([r[key] for r in rr]);ax.plot([xx-.07,xx+.07],[v,v],color=INK,lw=3,zorder=4)
        ax.set_title(f'{title}\n{s["reduced"]}/{s["n"]} 组距离下降 · 配对降幅中位数 {abs(s["median_paired_percent"]):.1f}%',loc='left',fontsize=12,pad=14)
        ax.set_xlim(-.13,1.13);ax.set_ylim(ymin,ymax)
        ax.set_xticks([0,1],['对齐前\n近似恢复图','对齐后\n原图，VLA 启动前'])
        ax.tick_params(axis='x',length=0,pad=12);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    axs[0].set_ylabel('头图视觉距离（3 近邻平均余弦距离）',labelpad=12)
    fig.text(.065,.935,'T2 近场：对齐前后的视觉距离',fontsize=23,weight='bold')
    fig.text(.065,.882,'近似恢复版 · 同一 T2 编码器 · 179 条训练示范首帧 · 26 条保留记录中有 22 组前后图片',fontsize=13,color=GRAY)
    legend=[Line2D([],[],color=GREEN,lw=6,alpha=.55,label='距离下降'),Line2D([],[],color=RED,lw=6,alpha=.55,label='距离上升'),
            Line2D([],[],ls='',marker='o',color=GRAY,label='第二版'),Line2D([],[],ls='',marker='s',color=GRAY,label='第三版'),
            Patch(facecolor=GOLD,alpha=.3,label='训练留一法距离：中间 50%'),Line2D([],[],color=INK,lw=3,label='组内中位数')]
    fig.legend(handles=legend,loc='lower left',bbox_to_anchor=(.065,.118),ncol=6,frameon=False,fontsize=10,handlelength=1.5,columnspacing=1.6)
    fig.text(.065,.096,'对齐前使用颜色反算与局部线条修补；含不可逆信息损失。第二版按成功筛选，本图仅描述所选样本。',fontsize=10.5,color=GRAY)
    fig.text(.065,.06,'S 形线只连接前后两个端点，不是实测时间轨迹。缺少结束头图的 4 条仍保留归档：1 条成功、3 条待标注。',fontsize=10.5,color=GRAY)
    save(fig,'T2_近场对齐前后趋势_近似恢复版')

    # Example fixed by chronological order, not selected by effect size.
    r=rows[1];audit={t['trial_id']:t for t in read(OUT/'inputs/restoration_audit.json')};raw=audit[r['trial_id']]['source']
    fig,axs=plt.subplots(1,3,figsize=(15,5))
    for ax,path,label in zip(axs,[raw,r['pre_image'],r['post_image']],['原始诊断图：带分割与边沿标记','近似恢复：颜色反算＋局部修补','对齐结束：VLA 启动前原图']):
        ax.imshow(Image.open(path));ax.set_title(label,fontsize=13);ax.axis('off')
    fig.suptitle(f'恢复效果示例 · {r["trial_id"]}',fontsize=17,x=.055,ha='left')
    fig.text(.055,.055,'中图为近似恢复数据，无法还原被实色线条覆盖的真实像素；原始文件完整保留。',fontsize=11,color=GRAY)
    fig.subplots_adjust(left=.04,right=.98,top=.82,bottom=.13,wspace=.04)
    save(fig,'T2_去标记效果示例')

    fig,axs=plt.subplots(1,2,figsize=(12,5.2));fig.subplots_adjust(left=.09,right=.98,top=.80,bottom=.25,wspace=.25)
    delta=np.array([r['delta_spatial'] for r in rows])
    for key,label,color,marker in [('delta_threshold70_spatial','更保守的紫色阈值',GREEN,'o'),('delta_unfilled_spatial','不修补实色边沿线',GRAY,'s')]:
        alt=np.array([r[key] for r in rows]);axs[0].scatter(delta,alt,c=color,s=30,alpha=.65,label=label,marker=marker)
    lo=min(delta.min(),min(r['delta_unfilled_spatial'] for r in rows))-.0003;hi=max(delta.max(),.0003)+.0003
    axs[0].plot([lo,hi],[lo,hi],color=GRAY,ls=':',lw=1);axs[0].axhline(0,color=GRAY,lw=.7);axs[0].axvline(0,color=GRAY,lw=.7)
    axs[0].set_xlabel('默认恢复：对齐后 − 对齐前');axs[0].set_ylabel('敏感性处理：对齐后 − 对齐前');axs[0].legend(frameon=False,fontsize=9)
    axs[0].set_title('恢复处理敏感性：负值代表距离下降',loc='left',fontsize=12)
    controls=list(__import__('csv').DictReader((OUT/'results/synthetic_recovery_controls.csv').open(encoding='utf-8-sig')))
    err=np.array([abs(float(c['recovery_score_error'])) for c in controls]);axs[1].scatter(np.ones(len(err)),err,color=GRAY,s=25,alpha=.6)
    axs[1].scatter(np.zeros(len(delta)),np.abs(delta),c=[GREEN if d<0 else RED for d in delta],s=25,alpha=.7)
    axs[1].set_xticks([0,1],['观察到的配对变化\n22 组','合成恢复误差\n44 张']);axs[1].set_xlim(-.45,1.45)
    axs[1].set_ylabel('视觉距离绝对变化');axs[1].grid(axis='y',alpha=.12);axs[1].set_title('合成检验仅提供误差量级参考',loc='left',fontsize=12)
    fig.suptitle('恢复误差与敏感性检查',x=.08,ha='left',fontsize=19)
    fig.text(.08,.08,'合成检验并非真实恢复误差的上界或置信区间；不修补边沿线的版本仍含标记，仅作为敏感性参考。',fontsize=10,color=GRAY)
    save(fig,'T2_恢复误差与敏感性')

    s=stats['combined'];control=stats['synthetic_recovery_check']
    lines=['# T2 近场对齐前后视觉距离：近似恢复版','',
      '## 结果与口径','',
      f'22 组可配对试验中，{s["reduced"]} 组对齐后距离下降；逐试验相对变化的中位数为 {s["median_paired_percent"]:.2f}%。这是对所选样本、近似恢复图的描述，不是原始图像结果或因果结论。','',
      '| 范围 | 保留 | 成功 | 失败 | 待标注 | 已标注成功率 | 可配对 |',
      '|---|---:|---:|---:|---:|---:|---:|',
      '| 第二版：仅成功 | 11 | 11 | 0 | 0 | 11/11（100%） | 11 |',
      '| 第三版：全部 | 15 | 11 | 1 | 3 | 11/12（91.7%） | 11 |',
      '| T2 near_dual 本批＝累计 | 26 | 22 | 1 | 3 | 22/23（95.7%） | 22 |','',
      '另有 21 条 smoke；22 条成功中有 1 条没有完成对齐后的头图，不能进入配对分析。待标注不计作失败。原始共有 47 条，按 trial_id 唯一处理。成功率经过用户指定的成功筛选，不可作为无偏方法评测。','',
      '第二版头/胸目标距离 1.20/1.10 m；第三版 1.10/1.15 m。修改仅追加排除/纳入元数据，未修改人工成败、Progress 分子分母或完成时间；已运行标准归档程序。','',
      '## 方法','',
      '使用此前用户确认的 carry_box_big_179/29999 视觉编码器。训练参考为 CarryBoxBig 全部 179 条示范的头部第一帧；从 agent_full 分支及本机 Git LFS 对象恢复并核对 SHA256，无需重新访问服务器。新提取的视觉权重哈希与历史参考一致。','',
      'SigLIP So400m/14 视觉投影输出（语言融合前），OpenPI uint8 resize/pad224，去掉上下各2行 padding patch，12×16 有效 patch 分成3×4空间格，每格平均并归一化，拼接再归一化。各图计算距训练参考最近3个不同 episode 的平均余弦距离；前后允许选择不同近邻。','',
      '对齐前取同 trial 第一次 ALIGN 的 initial.jpg；对齐后取首个 VLA start 命令之前、最后允许通过的 ALIGN gate 引用的头图。校验图片在控制器 aligned 之后、VLA 命令之前。一次包含对齐重试的试验保留最初起点到最终对齐后的变化。','',
      '## 为什么只能近似恢复','',
      '原始诊断渲染是 uint8(0.55×RGB + 0.45×[255,0,255])，保存为 JPEG95。依据颜色检测紫色区域并逆变换；黄色/绿色线条位置用局部 Telea 插值修补。未生成新的场景或物体，但插值像素不是观测数据。原始分割 mask 未保存，JPEG 和实色线造成不可逆损失。21 张前图有去标记处理，1 张初始检测失败的图未检出覆盖、保留像素。','',
      '参数在计算 embedding 之前固定，没有按距离改善程度或成败调参。更保守阈值仍为21/22下降；不修补边沿线版本22/22下降，但仍含标记。全局平均特征敏感性为21/22下降。','',
      f'合成检验：22 张干净后图和22张均匀抽取的训练图，加推断的覆盖模板、JPEG95编码，再执行同一恢复过程。参考距离绝对误差中位数 {control["median_absolute_distance_error"]:.6f}，95分位 {control["p95_absolute_distance_error"]:.6f}，最大 {control["max_absolute_distance_error"]:.6f}；真实配对变化绝对值的中位数为 {np.median(np.abs(delta)):.6f}。合成模板不等于真实掩膜，误差不能作为真实恢复误差的上界或显著性保证。','',
      '## 图与数据','',
      '- [配对趋势 PNG](figures/T2_近场对齐前后趋势_近似恢复版.png) · [PDF](figures/T2_近场对齐前后趋势_近似恢复版.pdf)',
      '- [去标记效果](figures/T2_去标记效果示例.png) · [误差检查](figures/T2_恢复误差与敏感性.png)',
      '- [逐条距离与近邻 CSV](results/approximate_paired_scores.csv)',
      '- [恢复检验 CSV](results/synthetic_recovery_controls.csv)',
      '- [筛选清单](selection.json) · [缺失图片清单](image_availability.csv)',
      '- [逐图恢复参数及哈希](inputs/restoration_audit.json) · [提取权重与代码来源](features/provenance.json)','',
      '## 论文使用建议','',
      '目前适合作为内部探索或明确披露恢复步骤的补充分析；不建议将这张图单独作为论文核心 claim。第二版成功筛选、同场景重复和恢复误差均限制外推。若要形成原图级证据，需要新的实验直接记录对齐前未标注头图。S形连接仅用于展示两个端点，不代表过程轨迹、拟合或置信带。训练背景带为179个训练首帧留一法距离的四分位范围，不是成功判据。','',
      '## 重现','',
      '脚本：tools/t1_visual_handoff/nearfield_t2_20260921.py（select/references/audit），nearfield_restore.py（prepare/analyze），extract.py，render_nearfield_t2.py。先prepare，再设置 T1_VISUAL_OUT 为本目录运行 extract，最后 analyze 与 render。select 已完成且以 batch_id 幂等，不需再次标注。','']
    (OUT/'README.md').write_text('\n'.join(lines))
    # Check summaries against raw logs and archived records; use original log prefix backups.
    sel=read(OUT/'selection.json')['trials'];expected={r['trial_id'] for r in sel if r['selected']}
    for p in (OUT/'source_backups').glob('*/events.before_selection.jsonl'):
        original=ROOT/'outputs/nearfield_experiments/T2/align_vla'/p.parent.name/'events.jsonl'
        assert original.read_bytes().startswith(p.read_bytes())
    archive=read(ROOT/'experiment data/manifest.json')['trials']
    archived={r['trial_id'] for r in archive if r['method']=='near_dual' and r['task_id']=='T2'}
    assert archived==expected
    assert len(rows)==22 and len({r['trial_id'] for r in rows})==22
    # Independent scipy implementation verifies all reported 3-neighbor distances.
    from scipy.spatial.distance import cdist
    f=np.load(OUT/'features/embeddings.npz');ix={s:i for i,s in enumerate(f['sample_ids'])}
    discrepancy=0.
    for r in rows:
        for stage in ['pre','post']:
            d=cdist(f['spatial'][ix[stage+'_'+r['trial_id']]][None],refs,metric='cosine')[0]
            error=abs(np.sort(d)[:3].mean()-r[f'{stage}_spatial']);discrepancy=max(discrepancy,error)
    assert discrepancy<1e-12
    write_json(OUT/'verification.json',dict(original_logs_preserved_as_prefix=True,archive_matches_selection=26,
        selected=26,smoke=21,paired=22,missing_pairs=4,physical_success=22,labeled=23,pending=3,
        independent_distance_max_error=discrepancy,training_frames_sha256_verified=179,
        endpoint_preserving_sigmoid=True,exact_pre_recovery_claimed=False,
        inputs=[{'path':str(p),'sha256':sha(p)} for p in [OUT/'selection.json',OUT/'inputs/samples.json',OUT/'features/embeddings.npz',OUT/'results/approximate_paired_scores.csv']]))
    print('Saved figures, README and verification to',OUT)


if __name__=='__main__':main()
