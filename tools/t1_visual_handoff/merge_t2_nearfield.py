"""Merge the displayed historical T2 data with the selected nearfield cohort."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-t2-merged')
import hashlib
import subprocess
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from render_nearfield_t2 import INK, GRAY, GREEN, RED
from nearfield_t2_20260921 import OUT, ROOT, read, write_json, write_csv, sha

DEST=OUT/'merged_with_full_vln'
BASE='experiment data/T2_垃圾袋入箱并搬箱/review/'


def main():
    DEST.mkdir(exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','agent_full'],cwd=ROOT,text=True).strip()
    provenance={}
    def historical(folder,name):
        import json
        path=BASE+folder+'/'+name
        raw=subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
        dest=DEST/'sources'/folder/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
        provenance[path]={'commit':commit,'sha256':hashlib.sha256(raw).hexdigest()}
        return json.loads(raw)
    folder='visual_handoff_20260918_pre_vla_head';paired='alignment_head_change_20260918'
    old=[t for t in historical(folder,'results/trials.json') if t['has_image']]
    old_pairs=[t for t in historical(paired,'results/trials.json') if t['align_allowed']]
    assert len(old)==26 and len(old_pairs)==8
    cfg=read(OUT/'analysis_config.json');new_prov=read(OUT/'features/provenance.json')
    for f,pro in [(folder,'combined_features/provenance.json'),(paired,'features/provenance.json')]:
        c=historical(f,'analysis_config.json');p=historical(f,pro)
        assert c['checkpoint']==cfg['checkpoint'] and c['reference_episode_ids']==cfg['reference_episode_ids']==list(range(179))
        assert c['descriptor_primary']==cfg['descriptor_primary']
        assert p['vision_bfloat16_parameter_sha256']==new_prov['vision_bfloat16_parameter_sha256']
    near=read(OUT/'results/approximate_paired_scores.json');assert len(near)==22
    assert not ({r['trial_id'] for r in old}&{r['trial_id'] for r in near})
    methods=['full_vln','nav_direct_vla','geometric_vla','near_dual']
    names=['Full VLN','导航后直接 VLA','几何对齐 + VLA','近场 BasePose + VLA']
    points=[dict(trial_id=t['trial_id'],method=t['method'],success=t['success'],distance=t['head_spatial'],source='historical_clean_handoff') for t in old]
    points += [dict(trial_id=t['trial_id'],method='near_dual',success=t['physical_success'],distance=t['post_spatial'],source='nearfield_clean_post_alignment') for t in near]
    pairs=[]
    for t in old_pairs:
        r=t['metrics'];assert abs(next(p['distance'] for p in points if p['trial_id']==t['trial_id'])-r['post_spatial'])<1e-12
        pairs.append(dict(trial_id=t['trial_id'],cohort='Full VLN',version=None,success=t['success'],pre=r['pre_spatial'],post=r['post_spatial'],pre_image_kind='original',post_image_kind='original'))
    pairs += [dict(trial_id=t['trial_id'],cohort='nearfield',version=t['version'],success=t['physical_success'],pre=t['pre_spatial'],post=t['post_spatial'],pre_image_kind=t['pre_source'],post_image_kind='original') for t in near]
    for r in pairs:r.update(delta=r['post']-r['pre'],percent=100*(r['post']-r['pre'])/r['pre'])
    def stats(rows):
        return dict(n=len(rows),decreased=sum(r['delta']<0 for r in rows),median_paired_percent=float(np.median([r['percent'] for r in rows])),median_pre=float(np.median([r['pre'] for r in rows])),median_post=float(np.median([r['post'] for r in rows])))
    summary={name:stats([r for r in pairs if name=='combined' or r['cohort']==name]) for name in ['combined','Full VLN','nearfield']}
    assert len(points)==48 and len(pairs)==30 and len({r['trial_id'] for r in pairs})==30
    assert summary['Full VLN']['decreased']==8 and round(summary['Full VLN']['median_paired_percent'],1)==-33.7
    config_path=ROOT/'outputs/analysis/visual_distance_three_questions_20260919/plot_config_and_counts.json'
    h=read(config_path)['T2']['kde_absolute_bandwidth']
    fig,axs=plt.subplots(1,2,figsize=(17,7.8));fig.subplots_adjust(left=.07,right=.98,top=.76,bottom=.27,wspace=.27)
    ax=axs[0];bounds=[]
    for i,m in enumerate(methods):
        # Draw the rarer failure marker last so a centered success cannot hide it.
        rr=sorted([p for p in points if p['method']==m],key=lambda r:not r['success'])
        v=np.array([r['distance'] for r in rr])
        kde=gaussian_kde(v,bw_method=h/np.std(v,ddof=1));y=np.linspace(max(0,v.min()-4*h),v.max()+4*h,600)
        w=kde(y);w=.35*w/w.max();w[[0,-1]]=0
        ax.fill_betweenx(y,i-w,i+w,facecolor='#76a7b8',edgecolor='#588d9f',alpha=.48,linewidth=1.3)
        dots=ax.scatter(np.full(len(rr),i),v,c=[GREEN if r['success'] else RED for r in rr],s=35,edgecolors='white',linewidths=.6,zorder=4)
        assert np.all(dots.get_offsets()[:,0]==i);bounds.extend([y[0],y[-1]])
    ax.set_xticks(range(4),[name+f'\nn={sum(r["method"]==m for r in points)}'+('（筛选后）' if m=='near_dual' else '') for m,name in zip(methods,names)],fontsize=10)
    ax.set_xlim(-.6,3.6);span=max(bounds)-min(bounds);ax.set_ylim(max(0,min(bounds)-.04*span),max(bounds)+.04*span)
    ax.set_title('② 哪些启动画面更接近训练参考？\n原图的三组 + 近场对齐后，共 48 张干净头图',loc='left',fontsize=14,pad=14)
    ax.legend(handles=[Line2D([],[],ls='',marker='o',color=c,label=l) for c,l in [(GREEN,'物理成功'),(RED,'物理失败')]],frameon=False,fontsize=10)
    ax=axs[1];x=np.linspace(0,1,301);sig=1/(1+np.exp(-10*(x-.5)));blend=(sig-sig[0])/(sig[-1]-sig[0])
    for r in pairs:
        c=GREEN if r['delta']<0 else RED;y=r['pre']+r['delta']*blend
        assert np.allclose(y[[0,-1]],[r['pre'],r['post']],rtol=0,atol=1e-15)
        ax.plot(x,y,color=c,lw=7,alpha=.24,solid_capstyle='round',zorder=2)
        ax.scatter([0,1],[r['pre'],r['post']],color=c,s=28,alpha=.72,marker='o' if r['cohort']=='Full VLN' else 's',linewidths=0,zorder=3)
    for x0,key in enumerate(['pre','post']):
        v=np.median([r[key] for r in pairs]);ax.plot([x0-.085,x0+.085],[v,v],color=INK,lw=3,zorder=4)
    s=summary['combined'];ax.set_title(f'③ 合并后的对齐前后趋势 · n=30\n{s["decreased"]}/30 组下降；配对相对变化中位数 {s["median_paired_percent"]:+.1f}%',loc='left',fontsize=14,pad=14)
    ax.set_xlim(-.2,1.2);ax.set_ylim(.0052,max(r['pre'] for r in pairs)*1.20)
    ax.set_xticks([0,1],['对齐前\nFull VLN 导航结束／近场起点','对齐结束\n尚未启动 VLA'],fontsize=10)
    handles=[Line2D([],[],color=GREEN,lw=6,alpha=.4,label='距离下降'),Line2D([],[],color=RED,lw=6,alpha=.4,label='距离上升'),
             Line2D([],[],ls='',marker='o',color=GRAY,label='原 Full VLN：8 组'),Line2D([],[],ls='',marker='s',color=GRAY,label='近场：22 组'),
             Line2D([],[],color=INK,lw=3,label='合并时点中位数')]
    ax.legend(handles=handles,frameon=False,fontsize=9,ncol=2,loc='upper right')
    ax.text(.03,.035,'原 Full VLN：8/8 下降，变化中位数 −33.7%\n近场：21/22 下降，变化中位数 −28.0%',transform=ax.transAxes,color=GRAY,fontsize=10)
    for ax in axs:
        ax.set_ylabel('头图视觉距离（3 近邻平均余弦距离）',fontsize=11);ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
    fig.text(.07,.93,'T2：原实验与近场实验合并观察',fontsize=23,weight='bold')
    fig.text(.07,.875,'同一视觉编码器 · 同一 179 条训练首帧参考 · 相同特征与距离定义',fontsize=13,color=GRAY)
    fig.text(.07,.145,'左图全部使用未修补的交接头图；近场单列为第四组，以保留不同实验条件。各小提琴采用相同绝对带宽。',fontsize=10.5,color=GRAY)
    fig.text(.07,.105,'右图包含近场的近似恢复前图；第二版仅保留成功，合并统计只作描述，不视为同一无偏样本。',fontsize=10.5,color=GRAY)
    fig.text(.07,.065,'小提琴尾部是密度估计；S 形曲线只连接两个端点，不表示实测过程；叠色不代表置信带。',fontsize=10.5,color=GRAY)
    for ext in ['png','pdf','svg']:fig.savefig(DEST/f'T2_原实验与近场合并.{ext}',dpi=190,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    write_csv(DEST/'paired_30.csv',pairs);write_csv(DEST/'handoff_images_48.csv',points);write_json(DEST/'summary.json',summary)
    write_json(DEST/'verification.json',dict(historical_sources=provenance,nearfield_source_sha256=sha(OUT/'results/approximate_paired_scores.json'),
        matching_encoder_weights=True,matching_reference_episodes=179,matching_descriptor=True,
        no_duplicate_trial_ids=True,violin_n=48,paired_n=30,violin_absolute_bandwidth=h,
        no_horizontal_jitter=True,no_boxplots=True,endpoint_normalized_sigmoid=True,pooled_confidence_interval_computed=False,
        counts={m:sum(r['method']==m for r in points) for m in methods},code_sha256=sha(__file__)))
    (DEST/'README.md').write_text(f'''# T2 原实验与近场合并观察

原图数据从 agent_full 的固定提交 {commit} 读取，逐文件哈希见 verification.json。不是从截图估算点位。

小提琴图保留原三个方法（8、9、9），另列近场 BasePose + VLA（22），合计48张；所有后图/交接头图均为未修补原图。近场可配对样本21成功、1失败。完整尾部，相同带宽 {h}，散点一列，红绿表示人工成败。

趋势图合并原 Full VLN 的8组与近场22组，共30组，{s['decreased']}/30距离下降，逐试验相对变化中位数 {s['median_paired_percent']:.6f}%。没有把两组中位数取平均；用30条原始差值重新统计。圆点为Full VLN、方点为近场；线色表示距离变化方向，不表示成败。

近场中21张前图经过近似恢复；不同起始条件、第二版成功筛选及恢复误差限制合并结果的推断。合并是描述性展示，不新增整体置信区间或显著性结论。参考此前近场目录的恢复检验。

[PNG](T2_原实验与近场合并.png) · [PDF](T2_原实验与近场合并.pdf) · [30条配对](paired_30.csv) · [48张交接头图](handoff_images_48.csv)
''')
    print(summary)
    print(DEST)


if __name__=='__main__':main()
