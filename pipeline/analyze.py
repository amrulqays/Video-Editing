import numpy as np, soundfile as sf, pyloudnorm as pyln, sys
from scipy import signal
x, sr = sf.read(sys.argv[1] if len(sys.argv)>1 else 'score_mix.wav')
m = pyln.Meter(sr, block_size=0.4)
print('integrated', round(pyln.Meter(sr).integrated_loudness(x),2))
# short-term loudness every 0.5 s (3 s window approximated by 1s)
row=[]
for t in np.arange(0, len(x)/sr-1, 0.5):
    seg = x[int(t*sr):int((t+1.0)*sr)]
    try: l = m.integrated_loudness(seg)
    except Exception: l = -99
    row.append(f'{t:4.1f}:{l:5.1f}')
for i in range(0,len(row),8): print('  '.join(row[i:i+8]))
# band energy balance
f, P = signal.welch(x.mean(axis=1), sr, nperseg=8192)
bands=[(20,60),(60,150),(150,400),(400,1000),(1000,3000),(3000,6000),(6000,12000),(12000,20000)]
tot=P.sum()
print('band balance (dB rel total):', ' '.join(f'{a}-{b}:{10*np.log10(P[(f>=a)&(f<b)].sum()/tot):.1f}' for a,b in bands))
# click detection: large 2nd-difference relative to local rms
mono = x.mean(axis=1)
d2 = np.abs(np.diff(mono,2))
loc = np.sqrt(signal.convolve(mono**2, np.ones(480)/480, 'same'))[1:-1]+1e-4
r = d2/loc
idx = np.argsort(r)[-8:]
print('top click ratios:', [(round(i/sr,3), round(r[i],1)) for i in sorted(idx)])
print('stereo corr', round(np.corrcoef(x[:,0],x[:,1])[0,1],3))
