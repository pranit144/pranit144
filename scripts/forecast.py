"""Trains a tiny model on my recent GitHub contributions and draws the forecast as an SVG.
Runs daily from .github/workflows/forecast.yml. Needs only numpy."""
import json, os, sys, datetime as dt, urllib.request
import numpy as np

USER = os.environ.get("GH_USER", "pranit144")
TOKEN = os.environ.get("GITHUB_TOKEN", "")
DAYS, HORIZON, RIDGE = 60, 7, 1.0
OUT = os.environ.get("OUT", "assets/forecast.svg")

def fetch():
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=DAYS)
    query = """query($u:String!,$f:DateTime!,$t:DateTime!){user(login:$u){contributionsCollection(from:$f,to:$t){
      contributionCalendar{weeks{contributionDays{date contributionCount}}}}}}"""
    body = json.dumps({"query": query, "variables": {
        "u": USER, "f": start.isoformat(), "t": end.isoformat()}}).encode()
    req = urllib.request.Request("https://api.github.com/graphql", data=body, headers={
        "Authorization": f"bearer {TOKEN}", "Content-Type": "application/json"})
    data = json.load(urllib.request.urlopen(req))
    weeks = data["data"]["user"]["contributionsCollection"]["contributionCalendar"]["weeks"]
    days = sorted((d["date"], d["contributionCount"]) for w in weeks for d in w["contributionDays"])
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    days = [d for d in days if d[0] <= today][-DAYS:]  # drop future days GitHub pads into the last week
    if len(days) < 21:
        raise SystemExit('Not enough contribution history yet (need at least 21 days).')
    return [dt.date.fromisoformat(d) for d, _ in days], np.array([c for _, c in days], float)

def demo():
    rng = np.random.default_rng(1)
    dates = [dt.date.today() - dt.timedelta(days=DAYS - 1 - i) for i in range(DAYS)]
    y = rng.poisson(3 + 2 * np.sin(np.arange(DAYS) / 4) + (np.array([d.weekday() for d in dates]) < 5) * 2)
    return dates, y.astype(float)

def features(dates, t0):
    X = []
    for d in dates:
        dow = np.zeros(6); 
        if d.weekday() < 6: dow[d.weekday()] = 1
        X.append(np.concatenate([[1.0, (d - t0).days / DAYS], dow]))
    return np.array(X)

def fit(X, y):
    A = X.T @ X + RIDGE * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ np.log1p(y))

def predict(w, X):
    return np.clip(np.expm1(X @ w), 0, None)

def main():
    dates, y = demo() if "--demo" in sys.argv else fetch()
    t0 = dates[0]
    X = features(dates, t0)

    # Honest check: train on everything except the last 7 days, score on those 7.
    w_cv = fit(X[:-HORIZON], y[:-HORIZON])
    mae = float(np.mean(np.abs(predict(w_cv, X[-HORIZON:]) - y[-HORIZON:])))
    base = float(np.mean(np.abs(y[:-HORIZON].mean() - y[-HORIZON:])))

    w = fit(X, y)
    sd = float(np.std(np.log1p(y) - X @ w))
    fdates = [dates[-1] + dt.timedelta(days=i + 1) for i in range(HORIZON)]
    Xf = features(fdates, t0)
    mid = predict(w, Xf)
    lo = np.clip(np.expm1(Xf @ w - sd), 0, None)
    hi = np.expm1(Xf @ w + sd)

    W, H, L, R, T, B = 820, 270, 44, 20, 56, 40
    n = len(dates) + HORIZON
    ymax = max(float(y.max()), float(hi.max()), 1) * 1.15
    px = lambda i: L + (W - L - R) * (i + 0.5) / n
    py = lambda v: H - B - (H - T - B) * v / ymax
    bw = (W - L - R) / n * 0.62

    bars = "".join(f'<rect x="{px(i)-bw/2:.1f}" y="{py(v):.1f}" width="{bw:.1f}" height="{H-B-py(v):.1f}" rx="2" class="bar"/>'
                   for i, v in enumerate(y))
    k = len(dates)
    band = " ".join(f"{px(k+i):.1f},{py(hi[i]):.1f}" for i in range(HORIZON)) + " " + \
           " ".join(f"{px(k+i):.1f},{py(lo[i]):.1f}" for i in reversed(range(HORIZON)))
    line = " ".join(f"{px(k+i):.1f},{py(mid[i]):.1f}" for i in range(HORIZON))
    dots = "".join(f'<circle cx="{px(k+i):.1f}" cy="{py(mid[i]):.1f}" r="3.5" class="dot"/>' for i in range(HORIZON))
    grid = "".join(f'<line x1="{L}" x2="{W-R}" y1="{py(v):.1f}" y2="{py(v):.1f}" class="grid"/>'
                   f'<text x="{L-8}" y="{py(v)+4:.1f}" text-anchor="end" class="muted">{v}</text>'
                   for v in range(0, int(ymax) + 1, max(1, int(ymax // 4))))
    labels = f'<text x="{px(0):.1f}" y="{H-14}" class="muted">{dates[0]:%b %d}</text>' \
             f'<text x="{px(k-1):.1f}" y="{H-14}" text-anchor="end" class="muted">{dates[-1]:%b %d}</text>' \
             f'<text x="{px(n-1):.1f}" y="{H-14}" text-anchor="end" class="muted">{fdates[-1]:%b %d}</text>'
    verdict = "beats" if mae < base else "trails"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace">
<style>
:root{{--bg:#ffffff;--fg:#2b3a37;--mut:#6b7f7a;--grid:#e3e8e4;--bar:#b4c1ba;--fc:#2f6b63}}
.bg{{fill:var(--bg);stroke:#cdd2ca;stroke-width:1.5}}.t{{fill:var(--fg);font-size:15px;font-weight:600}}.muted{{fill:var(--mut);font-size:11px}}
.grid{{stroke:var(--grid);stroke-width:1}}.bar{{fill:var(--bar)}}.band{{fill:var(--fc);opacity:.18}}
.line{{fill:none;stroke:var(--fc);stroke-width:2.5;stroke-dasharray:6 4;stroke-linecap:round}}.dot{{fill:var(--fc)}}
</style>
<rect class="bg" x="1" y="1" width="{W-2}" height="{H-2}" rx="10"/>
<text x="{L}" y="26" class="t">A model that retrains itself every day</text>
<text x="{L}" y="43" class="muted">Ridge regression · last {DAYS} days · forecast next {HORIZON} · updated {dt.date.today():%d %b %Y}</text>
{grid}{bars}<polygon points="{band}" class="band"/><polyline points="{line}" class="line"/>{dots}{labels}
<text x="{W-R}" y="26" text-anchor="end" class="muted">holdout MAE {mae:.2f} · baseline {base:.2f}</text>
<text x="{W-R}" y="43" text-anchor="end" class="muted">model {verdict} the baseline</text>
</svg>'''
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    open(OUT, "w").write(svg)
    print(f"wrote {OUT}  mae={mae:.2f} baseline={base:.2f}")

if __name__ == "__main__":
    main()
