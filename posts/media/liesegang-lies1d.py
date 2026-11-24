import numpy as np
def run(N=600,a0=12.,b0=1.,Da=1.,Db=0.05,Dc=0.4,k=1.,cs=0.32,cp=0.06,g=1.,T=60000,dt=0.2):
    a=np.zeros(N);b=np.full(N,b0);c=np.zeros(N);p=np.zeros(N)
    nuc=np.full(N,-1.)
    def lap(u):
        l=np.empty_like(u); l[1:-1]=u[2:]-2*u[1:-1]+u[:-2]; l[0]=u[1]-u[0]; l[-1]=u[-2]-u[-1]; return l
    for s in range(T):
        a[0]=a0
        r=k*a*b
        a+=dt*(Da*lap(a)-r); b+=dt*(Db*lap(b)-r); c+=dt*(Dc*lap(c)+r)
        a[0]=a0
        new=(p==0)&(c>cs)
        nuc[new]=s*dt
        grow=((p>0)|new)&(c>cp)
        q=np.where(grow,g*(c-cp),0)*dt
        q=np.minimum(q,c); c-=q; p+=q
    return p,nuc
if __name__=="__main__":
    p,nuc=run()
    # find bands: contiguous p>thr
    m=p>0.05*p.max()
    edges=np.flatnonzero(np.diff(m.astype(int)))
    print("bands starts/ends",edges[:60])
