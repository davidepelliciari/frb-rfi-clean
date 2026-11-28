import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
P=np.load('gp_profiles.npy'); names=['originale','originale + rfifind','step 1','step 2','step 3','step 4']
snr=[4.4,9.7,9.1,9.8,9.5,8.4]
nb=128; pk=39; sh=int(nb*0.25)-pk
fig,ax=plt.subplots(figsize=(7,6.5))
ph=(np.arange(nb)+0.5)/nb
for i,(p,n) in enumerate(zip(P,names)):
    off=np.array([j for j in range(nb) if min((j-pk)%nb,(pk-j)%nb)>8])
    b=np.median(p[off]); m=1.4826*np.median(abs(p[off]-b)); z=np.roll((p-b)/m,sh)
    y=-12*i; ax.step(ph,z+y,where='mid',color='k',lw=0.8)
    ax.text(1.01,y,f'{n}\nS/N {snr[i]}',va='center',fontsize=8,transform=ax.get_yaxis_transform())
ax.axvspan(0.25-1.5/nb,0.25+1.5/nb,color='0.85',zorder=0)
ax.set_xlim(0,1); ax.set_yticks([]); ax.set_xlabel('fase (impulso principale a 0.25)'); ax.set_ylabel('S/N per intervallo (sfalsati)')
ax.set_title('Crab a Noto: profilo ripiegato (P = 33.841426 ms, DM 56.72, 128 intervalli)',fontsize=9)
plt.tight_layout(rect=(0,0,0.84,1)); plt.savefig('/home/user/frb-rfi-clean/docs/report_no0004/fig/noto/crab_profiles.png',dpi=150)
