import sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from skimage import io as skio
from xrdpipe.io import load_image
from xrdpipe.features import extract_features
P, OUT = sys.argv[1], sys.argv[2]
raw=skio.imread(P); f,d=extract_features(load_image(P))
BG='#151515'; INK='#e6e6e3'; MUTED='#8f8e86'; GRID='#2a2a27'; ACC='#3987e5'; HL='#eb6834'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':INK,'axes.labelcolor':MUTED,
 'xtick.color':MUTED,'ytick.color':MUTED,'axes.edgecolor':GRID})
fig=plt.figure(figsize=(12,5),facecolor=BG)
a=fig.add_axes([0.02,0.06,0.38,0.86]); b=fig.add_axes([0.49,0.14,0.48,0.74])
a.imshow(raw,cmap='gray',vmin=0,vmax=255); a.axis('off')
cy,cx=d['center']; r=d['r']; p=d['profile']; pk=d['peaks']
prom=np.array([p[k]/d['baseline'][k] for k in pk]); top=pk[np.argsort(prom)[::-1][:2]]
for k in top:
    # partial arc, offset outward, so the ring itself stays visible
    a.add_patch(Arc((cx,cy),2*(r[k]+9),2*(r[k]+9),theta1=200,theta2=340,ec=HL,lw=2))
a.set_title('Raw RefleX image',color=INK,fontsize=12,loc='left')
b.set_facecolor(BG); b.semilogy(r,p,color=ACC,lw=2,label='azimuthal average')
b.semilogy(r,d['baseline'],color=MUTED,lw=1.2,ls='--',label='smooth background')
b.plot(r[top],p[top]*1.12,'v',color=HL,ms=9,label='detected ice rings')
b.axvspan(0,32,color=GRID,alpha=0.6,lw=0)
b.text(36,np.nanmin(p[p>0])*1.05,'beamstop region (ignored)',color=MUTED,fontsize=9,va='bottom')
b.set_xlim(0,r.max()); b.set_xlabel('radius from beam centre (px)'); b.set_ylabel('mean intensity (inverted counts)')
b.grid(color=GRID,lw=0.6,which='major'); [b.spines[s].set_visible(False) for s in ('top','right')]
b.legend(frameon=False,labelcolor=INK,loc='upper right')
b.set_title('Ring profile: ice rings stand out as sharp spikes',color=INK,fontsize=12,loc='left')
fig.savefig(OUT,dpi=130,facecolor=BG)
