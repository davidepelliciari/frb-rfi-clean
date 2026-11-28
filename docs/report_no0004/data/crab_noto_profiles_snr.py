import numpy as np, glob, re
names=['orig','orig_rfifind','step1','step2','step3','step4']
P={}
for n in names:
    f=glob.glob(f'gp_{n}_*.bestprof')[0]; t=open(f).read()
    prof=np.array([float(l.split()[1]) for l in t.splitlines() if l and not l.startswith('#')])
    psd=float(re.search(r'Profile StdDev\s*=\s*([\d.e+]+)',t).group(1))
    P[n]=(prof,psd)
pk=int(np.argmax(P['step1'][0])); print('peak bin',pk, 'phase',pk/128)
nb=128
def snr(prof,psd,won):
    on=[(pk+k)%nb for k in won]
    off=np.array([i for i in range(nb) if min((i-pk)%nb,(pk-i)%nb)>8 and min((i-pk-51)%nb,(pk+51-i)%nb)>6])
    base=np.median(prof[off]); mad=1.4826*np.median(abs(prof[off]-np.median(prof[off])))
    sig=(prof[on]-base).sum()/(mad*np.sqrt(len(on)))
    return sig, (prof[pk]-base)/mad, mad, mad/psd, (prof[on]-base).sum()
ref=None
for n in names:
    prof,psd=P[n]
    s1,pkv,mad,ratio,S=snr(prof,psd,[0])
    s3,_,_,_,S3=snr(prof,psd,[-1,0,1])
    if n=='step2': ref=(S3)
    print(f'{n:14s} SNRpicco {pkv:6.1f}  SNR(3bin) {s3:6.1f}  rumore/bianco(prepfold) {ratio:5.2f}  S3 {S3:.3e}')
# radiometric for steps: 8 counts * sqrt(Nlive) * sqrt(Nsamp/bin)
rad=8*np.sqrt(924)*np.sqrt(73420800/128); print('radiometrico step1-4', rad)
for n in names[2:]:
    prof,psd=P[n]; _,_,mad,_,S3=snr(prof,psd,[-1,0,1]); print(n,'rumore/radiometrico',round(mad/rad,2),'segnale/step2',round(S3/ref,3))
# interpulse: look at profile around pk+~51 (0.4)
pr=P['step1'][0]; base=np.median(pr); m=1.4826*np.median(abs(pr-base))
z=(pr-base)/m; print('profilo step1 z, bins pk-3..pk+3', np.round(z[[(pk+k)%nb for k in range(-3,4)]],1))
ipw=[(pk+k)%nb for k in range(45,58)]; print('interimpulso z', np.round(z[ipw],1), 'max at +', 45+int(np.argmax(z[ipw])))
for n in names: 
    pr=P[n][0]; b=np.median(pr); mm=1.4826*np.median(abs(pr-b)); zz=(pr-b)/mm; print(n,'IP z max',round(zz[ipw].max(),1),'sum IP(3)', round(zz[[(pk+k)%nb for k in (50,51,52)]].sum()/np.sqrt(3),1))
np.save('../gp_profiles.npy',np.array([P[n][0] for n in names]))
