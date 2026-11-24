"""Liesegang banding in a sandstone block, Griff 2026-11-24.
Oxygen (a) seeps in from the joints, ferrous iron (b) sits in the pores,
their product (c) only precipitates past a supersaturation threshold cs,
then a nucleated cell keeps eating c down to cp. Bands make themselves."""
import numpy as np, sys
def smooth(N,sc,rng):
    from PIL import Image
    n=rng.standard_normal((N//sc+2,N//sc+2)).astype(np.float32)
    im=np.asarray(Image.fromarray(n,'F').resize((N,N),Image.BICUBIC))
    return (im-im.mean())/im.std()
def lap(u):
    return (np.roll(u,1,0)+np.roll(u,-1,0)+np.roll(u,1,1)+np.roll(u,-1,1)-4*u)
def run(src, N=260, a0=12., b0=1., Da=1., Db=0.05, Dc=0.4, k=0.2, cs=0.5, cp=0.05,
        g=1., cn=0.45, T=30000, dt=0.1, perm=None, seed=1, snap_every=0):
    rng=np.random.default_rng(seed)
    a=np.zeros((N,N)); b=b0*(1+0.03*rng.standard_normal((N,N))); c=np.zeros((N,N)); p=np.zeros((N,N))
    nuc=np.full((N,N),np.nan)
    Dm = 1. if perm is None else perm
    cs_map = cs*(1+0.03*smooth(N,24,rng))
    snaps=[]
    for s in range(T):
        a[src]=a0
        r=k*a*b
        a+=dt*(Da*Dm*lap(a)-r); b+=dt*(Db*Dm*lap(b)-r); c+=dt*(Dc*Dm*lap(c)+r)
        a[src]=a0; b[src]=0
        new=(p==0)&(c>cs_map)
        nuc[new]=s*dt
        nb_=(np.roll(p,1,0)+np.roll(p,-1,0)+np.roll(p,1,1)+np.roll(p,-1,1))>0
        seeded=(p==0)&nb_&(c>cn)   # a band spreads sideways along itself
        nuc[seeded&np.isnan(nuc)]=s*dt
        grow=((p>0)|new|seeded)&(c>cp)
        q=np.minimum(np.where(grow,g*(c-cp),0)*dt,c); c-=q; p+=q
        if snap_every and s%snap_every==0: snaps.append(p.copy())
    return p,nuc,snaps
