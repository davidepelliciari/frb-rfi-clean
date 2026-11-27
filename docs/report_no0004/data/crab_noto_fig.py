import pandas as pd, numpy as np, glob, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
P=33.8414264e-3
def load(d):
    r=pd.read_csv(glob.glob(f'{d}/obs/*_processed_*/results_a.csv')[0])
    for k,p in (('t',r'tcand_([\d.]+)_'),('dm',r'_dm_([\d.]+)_'),('snr',r'_snr_([\d.]+)\.h5')): r[k]=r.candidate.str.extract(p).astype(float)
    c=pd.read_csv(glob.glob(f'{d}/obs/*_processed_*/*_all_candidates.csv')[0])
    r=r.merge(c[['stime','width']].rename(columns={'stime':'t'}).round({'t':4}).drop_duplicates('t'),left_on=r.t.round(4),right_on='t',how='left',suffixes=('','_y'))
    r['ph']=(r.t/P)%1; r['gp']=(abs(r.dm-56.72)<1)&((abs(r.ph-0.31)<0.03)|(abs(r.ph-0.71)<0.03))
    return r
o,c=load('orig'),load('clean')
fig,ax=plt.subplots(1,3,figsize=(15,4.6))
for r,lab,col,mk in ((o,'originale','tab:red','o'),(c,'pulito','tab:blue','x')):
    g=r[abs(r.dm-56.72)<1]
    ax[0].scatter(g.ph,g.t,s=8+2*g.snr,**(dict(facecolors='none',edgecolors=col) if mk=='o' else dict(c=col)),marker=mk,label=f'{lab} ({int(g.gp.sum())} in fase)')
    ax[2].scatter(r.t,r.dm,s=4+g.snr.max()*0+r.snr*0.8,c=col,marker=mk,alpha=0.6,label=f'{lab} ({len(r)})')
ax[0].set_xlabel('fase (P = 33.84143 ms)'); ax[0].set_ylabel('tempo (s)'); ax[0].set_title('candidati a DM 55.7–57.7'); ax[0].legend(fontsize=8,loc='center')
for x in (0.31,0.71): ax[0].axvline(x,color='0.8',zorder=0,lw=6)
go,gc=o[o.gp],c[c.gp]
m=pd.merge_asof(gc.sort_values('t'),go.sort_values('t')[['t','snr','probability']],on='t',tolerance=0.005,direction='nearest',suffixes=('_c','_o'))
mm=m.dropna(subset=['snr_o'])
acc=lambda p:p>=0.5
for sel,lab,col in ((acc(mm.probability_o)&acc(mm.probability_c),'FETCH sì / sì','k'),(~acc(mm.probability_o)&acc(mm.probability_c),'FETCH no / sì','tab:green'),(~acc(mm.probability_c),'FETCH – / no','0.6')):
    ax[1].scatter(mm.snr_o[sel],mm.snr_c[sel],c=col,s=25,label=lab)
oc=m[m.snr_o.isna()]
ax[1].scatter(np.full(len(oc),7.3),oc.snr_c,marker='>',c=['tab:green' if p>=0.5 else '0.6' for p in oc.probability_c],s=25,label=f'solo nel pulito ({len(oc)})')
lim=[7,50]; ax[1].plot(lim,lim,'k:',lw=1); ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlim(7,50); ax[1].set_ylim(7,50)
ax[1].set_xlabel('S/N originale'); ax[1].set_ylabel('S/N pulito'); ax[1].set_title(f'impulsi giganti (mediana pulito/orig. {np.median(mm.snr_c/mm.snr_o):.2f})'); ax[1].legend(fontsize=8,title='FETCH≥0.5 orig./pulito',title_fontsize=8)
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import NullFormatter
for a in ax[1:3]:
    for axis in (a.xaxis,a.yaxis): axis.set_minor_formatter(NullFormatter())
ax[2].yaxis.set_major_formatter(ScalarFormatter()); ax[2].set_yticks([30,56.7,100,200,400,700]); ax[2].set_yticklabels(['30','56.7','100','200','400','700'])
for a in ax[1:2]:
    for axis in (a.xaxis,a.yaxis): axis.set_major_formatter(ScalarFormatter()); 
    a.set_xticks([8,10,15,20,30,50]); a.set_yticks([8,10,15,20,30,50])
ax[2].set_yscale('log'); ax[2].set_xlabel('tempo (s)'); ax[2].set_ylabel('DM'); ax[2].set_title('tutti i candidati Heimdall'); ax[2].axhline(56.72,color='0.7',lw=0.8); ax[2].legend(fontsize=8,loc='upper right')
ax[2].yaxis.set_minor_formatter(NullFormatter()); ax[2].set_yticks([30,56.7,100,200,400,700]); ax[2].set_yticklabels(['30','56.7','100','200','400','700'])
plt.tight_layout(); plt.savefig('/home/user/frb-rfi-clean/docs/report_no0004/fig/pipeline/crab_noto_gp.png',dpi=150)
# numbers
for n,r in (('orig',o),('clean',c)):
    ng=r[~r.gp]
    print(n,len(r),'gp',int(r.gp.sum()),'fetch gp',int((r.gp&(r.probability>=0.5)).sum()),'nongp',len(ng),'nongp w>=2^12',int((ng.width>=12).sum()),'DM<50',int((ng.dm<50).sum()),'fetch nongp',int((~r.gp&(r.probability>=0.5)).sum()))
print('rejected clean snr', sorted(c[c.gp&(c.probability<0.5)].snr.round(1)))
print('accepted clean snr range', c[c.gp&(c.probability>=0.5)].snr.min(), c[c.gp&(c.probability>=0.5)].snr.max())
print('ip count', int((abs(c.ph-0.71)<0.03).sum()), int((abs(o.ph-0.71)<0.03).sum()))
print('lost:', go[~go.t.apply(lambda x:(abs(gc.t-x)<0.005).any())][['t','snr','probability']].values)
