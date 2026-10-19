# Griff's rough sonification of the Halberstadt ORGAN2/ASLSP schedule, part 1 (2001-2072).
# Schedule copied from https://www.aslsp.org/das-projekt.html . One second per year. Crude additive tone, not the real organ.
import numpy as np, datetime as dt, wave
base={'c':0,'cis':1,'des':1,'d':2,'dis':3,'es':3,'e':4,'f':5,'fis':6,'ges':6,'g':7,'gis':8,'as':8,'a':9,'ais':10,'b':10,'h':11}
def midi(tok):
    sixteen=tok.endswith('16'); tok=tok[:-2] if sixteen else tok
    primes=tok.count("'"); name=tok.replace("'","")
    octv = 2 if name[0].isupper() else 3+primes
    m=12*(octv+1)+base[name.lower()]
    return m-12 if sixteen else m   # 16' stop sounds an octave lower
ev=[]
for line in open('halberstadt-schedule.txt'):
    d,notes=line.strip().split('|'); dd,mm,yy=map(int,d.split('.'))
    ev.append((dt.date(2000+yy,mm,dd),[midi(t) for t in notes.split()]))
start=ev[0][0]; end=ev[-1][0]
sr=22050; spy=1.0  # one second per year
def t(d): return (d-start).days/365.2425*spy
N=int(t(end)*sr)+sr
out=np.zeros(N)
for i in range(len(ev)-1):
    a,b=int(t(ev[i][0])*sr),int(t(ev[i+1][0])*sr)
    tt=np.arange(a,b)/sr
    for m in ev[i][1]:
        f=440*2**((m-69)/12)
        tone=sum(w*np.sin(2*np.pi*f*k*tt) for k,w in [(1,1),(2,.5),(3,.25),(4,.12)])
        out[a:b]+=tone
# smooth seams 15ms
out/= np.max(np.abs(out))+1e-9
k=int(.015*sr); win=np.ones(k)/k; out=np.convolve(out,win,'same')
out=out/np.max(np.abs(out))*0.6
with wave.open('part1.wav','wb') as w:
    w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr);w.writeframes((out*32767).astype(np.int16).tobytes())
print('seconds',N/sr,'events',len(ev))
# stats
durs=[((ev[i+1][0]-ev[i][0]).days,ev[i][0],len(ev[i][1])) for i in range(len(ev)-1)]
print('longest',max(durs),'shortest',min(durs))
print('max notes',max(len(e[1]) for e in ev))
for d,n in ev:
    if len(n)==8: print('8-note', d)
# SVG piano roll
lo=min(m for _,n in ev for m in n); hi=max(m for _,n in ev for m in n)
W,H,px=700,260,9.6
X=lambda d:40+(d-start).days/((end-start).days)*(W-50)
Y=lambda m:H-30-(m-lo)*((H-50)/(hi-lo+1))
r=[]
for i in range(len(ev)-1):
    for m in ev[i][1]:
        r.append(f'<rect x="{X(ev[i][0]):.1f}" y="{Y(m)-5:.1f}" width="{max(X(ev[i+1][0])-X(ev[i][0]),0.8):.1f}" height="4.5" fill="#4a6b3a"/>')
today=dt.date(2026,10,19)
r.append(f'<line x1="{X(today):.1f}" y1="10" x2="{X(today):.1f}" y2="{H-25}" stroke="#b5562a" stroke-width="1.5"/><text x="{X(today)+4:.1f}" y="20" font-size="12" fill="#b5562a">today</text>')
for y in range(2001,2073,10):
    x=X(dt.date(y,1,1) if y>2001 else start); r.append(f'<text x="{x:.1f}" y="{H-8}" font-size="11" fill="#777">{y}</text>')
r.append(f'<text x="2" y="{Y(hi)+4:.1f}" font-size="10" fill="#777">high</text><text x="2" y="{Y(lo):.1f}" font-size="10" fill="#777">low</text>')
svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="Piano roll of the Halberstadt chord schedule, 2001 to 2072"><rect width="{W}" height="{H}" fill="#fbfaf7"/>'+''.join(r)+'</svg>'
open('part1.svg','w').write(svg)
el=(today-dt.date(2001,9,5)).days; tot=(dt.date(2640,9,5)-dt.date(2001,9,5)).days
print('elapsed days',el,'total',tot,'pct',100*el/tot)
print('slowdown vs 70min', tot*24*60/70, 'vs 20min', tot*24*60/20)
print('days to next change',(dt.date(2027,10,5)-today).days)
