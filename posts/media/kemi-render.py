"""Kemi coastline: PSMSL station 229 (annual RLR means) rendered as one rising tone.
Data: Permanent Service for Mean Sea Level (PSMSL), https://psmsl.org/data/obtaining/stations/229.php
Rules:
  pitch   - an 11-year centred running mean of the annual values that exist (missing years are
            skipped, never filled), inverted (lower sea = land higher = higher pitch), mapped
            linearly from 220 Hz (highest smoothed level) up 24 semitones (lowest).
            The tone glides from one year's value to the next. Smoothing is there because
            single years swing by +/-73 mm (1 SD) while the land rises ~6.6 mm a year.
  gaps    - a missing year (-99999) is silence. When readings resume, the tone starts at the
            smoothed level of the first year back, so the pitch jumps by however much the land rose unrecorded.
  storms  - a year whose residual from the straight-line trend exceeds 1.5 standard deviations
            gets a tremolo (7 Hz), deeper the bigger the residual. No storm catalogue is used.
  ending  - a tick, one real second of uplift at true speed (silence), a tick.
"""
import numpy as np, wave, json
rows=[l.split(';') for l in open('kemi-229-annual.txt') if l.strip() and not l.startswith('#')]
yr=np.array([int(r[0]) for r in rows]); v=np.array([int(r[1]) for r in rows],float)
ok=v!=-99999
slope,icpt=np.polyfit(yr[ok],v[ok],1)
res=np.where(ok, v-np.polyval([slope,icpt],yr), 0.0)
sd=res[ok].std(ddof=1); K=1.5
storm=ok & (np.abs(res)>K*sd)
H=5
sm=np.array([v[(np.abs(yr-y)<=H)&ok].mean() for y in yr])
vmax,vmin=sm[ok].max(),sm[ok].min(); SEMI=24; F0=220.0
def freq(x): return F0*2**(((vmax-x)/(vmax-vmin))*SEMI/12)

sr=22050; SPY=0.65
lead=0.5
nyears=len(yr)
body=nyears*SPY
tail_pause=2.0; real=1.0; after=1.5
N=int((lead+body+tail_pause+0.05+real+0.05+after)*sr)
out=np.zeros(N)
phase=0.0
for i in range(nyears):
    a=int((lead+i*SPY)*sr); b=int((lead+(i+1)*SPY)*sr); n=b-a
    if not ok[i]:
        phase=0.0; continue
    nxt = sm[i+1] if i+1<nyears and ok[i+1] else sm[i]
    f=np.linspace(freq(sm[i]),freq(nxt),n,endpoint=False)
    ph=phase+2*np.pi*np.cumsum(f)/sr; phase=ph[-1]
    tone=np.sin(ph)+.35*np.sin(2*ph)+.15*np.sin(3*ph)
    env=np.ones(n)
    fade=int(.03*sr)
    if i==0 or not ok[i-1]: env[:fade]=np.linspace(0,1,fade)          # entry after a gap
    if i+1>=nyears or not ok[i+1]: env[-fade:]=np.linspace(1,0,fade)  # exit into a gap
    if storm[i]:
        depth=min(0.9,0.4+0.25*(abs(res[i])/sd-K))
        t=np.arange(n)/sr
        env*=1-depth*(0.5+0.5*np.sin(2*np.pi*7*t))
    out[a:b]+=tone*env
def tick(at):
    a=int(at*sr); n=int(.02*sr); t=np.arange(n)/sr
    out[a:a+n]+=0.8*np.sin(2*np.pi*1800*t)*np.exp(-t*250)
t_tick1=lead+body+tail_pause
tick(t_tick1); tick(t_tick1+0.05+real)
out=out/np.max(np.abs(out))*0.7
with wave.open('kemi.wav','wb') as w:
    w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr);w.writeframes((out*32767).astype(np.int16).tobytes())

SEC_PER_YEAR=365.2425*86400
gaps=[]; i=0
while i<nyears:
    if not ok[i]:
        j=i
        while j+1<nyears and not ok[j+1]: j+=1
        before=sm[i-1]; after_v=sm[j+1]
        gaps.append(dict(years=f"{yr[i]}" if i==j else f"{yr[i]}-{yr[j]}",before=round(before,1),after=round(after_v,1),
                         jump_semitones=round((before-after_v)/(vmax-vmin)*SEMI,2),
                         trend_mm=round(-slope*(yr[j+1]-yr[i-1]),1)))
        i=j+1
    else: i+=1
stats=dict(station="KEMI",psmsl_id=229,first=int(yr[0]),smoothing_window_years=2*H+1,last=int(yr[-1]),years_with_data=int(ok.sum()),
    missing=[int(x) for x in yr[~ok]],trend_mm_per_yr=round(slope,3),
    trend_total_mm=round(-slope*(yr[-1]-yr[0]),0),
    residual_sd_mm=round(sd,1),storm_rule=f"|residual| > {K} x SD = {K*sd:.0f} mm",
    storm_years={int(y):int(round(r)) for y,r in zip(yr[storm],res[storm])},
    real_speed_mm_per_s=-slope/SEC_PER_YEAR, real_speed_semitones_per_s=(-slope/SEC_PER_YEAR)/(vmax-vmin)*SEMI,
    audio_seconds=round(N/sr,2),seconds_per_year=SPY,gaps=gaps,pitch_hz=[round(freq(vmax),1),round(freq(vmin),1)])
json.dump(stats,open('stats.json','w'),indent=1); print(json.dumps(stats,indent=1))

# SVG chart
W,H=720,340; L,R,T,B=60,20,20,60
x=lambda y_: L+(y_-1919.5)/(2025.5-1919.5)*(W-L-R)
lo,hi=6500,7420
yy=lambda s: T+(hi-s)/(hi-lo)*(H-T-B)
s=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Georgia,serif" font-size="12">',
   f'<rect width="{W}" height="{H}" fill="#fbfaf7"/>']
for i in range(nyears):
    if not ok[i]:
        s.append(f'<rect x="{x(yr[i]-.5):.1f}" y="{T}" width="{x(yr[i]+.5)-x(yr[i]-.5):.1f}" height="{H-T-B}" fill="#d9d3c7"/>')
for g in range(6600,7401,200):
    s.append(f'<line x1="{L}" x2="{W-R}" y1="{yy(g):.1f}" y2="{yy(g):.1f}" stroke="#e4e0d8"/><text x="{L-6}" y="{yy(g)+4:.1f}" text-anchor="end" fill="#555">{g}</text>')
for d in range(1920,2026,20):
    s.append(f'<text x="{x(d):.1f}" y="{H-B+16}" text-anchor="middle" fill="#555">{d}</text>')
s.append(f'<line x1="{x(1920)}" x2="{x(2025)}" y1="{yy(np.polyval([slope,icpt],1920)):.1f}" y2="{yy(np.polyval([slope,icpt],2025)):.1f}" stroke="#999" stroke-dasharray="4 3"/>')
seg=[]
def flush():
    if len(seg)>1: s.append('<polyline fill="none" stroke="#8fa3bb" stroke-width="1" points="'+' '.join(seg)+'"/>')
for i in range(nyears):
    if ok[i]: seg.append(f'{x(yr[i]):.1f},{yy(v[i]):.1f}')
    else: flush(); seg=[]
flush()
seg=[]
def flush2():
    if len(seg)>1: s.append('<polyline fill="none" stroke="#2c4a6e" stroke-width="3" points="'+' '.join(seg)+'"/>')
for i in range(nyears):
    if ok[i]: seg.append(f'{x(yr[i]):.1f},{yy(sm[i]):.1f}')
    else: flush2(); seg=[]
flush2()
for i in range(nyears):
    if ok[i]: s.append(f'<circle cx="{x(yr[i]):.1f}" cy="{yy(v[i]):.1f}" r="{3.5 if storm[i] else 1.8}" fill="{"#b5452b" if storm[i] else "#2c4a6e"}"/>')
s.append(f'<text x="14" y="{(T+H-B)/2}" transform="rotate(-90 14 {(T+H-B)/2})" text-anchor="middle" fill="#555">annual mean sea level, mm (PSMSL RLR)</text>')
s.append(f'<text x="{(L+W-R)/2}" y="{H-12}" text-anchor="middle" fill="#555">grey bands: no annual value · thick line: 11-yr mean (the pitch) · red dots: tremolo years · dashed: trend {slope:.1f} mm/yr</text>')
s.append('</svg>')
open('kemi.svg','w').write('\n'.join(s))
