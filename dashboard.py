"""
Trading Games Dashboard — serves on port 8888
Run: python dashboard.py
"""
import json
import re
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

BASE = Path(__file__).parent
SCORES_FILE = BASE / "polymarket" / "data" / "trading_games_scores.jsonl"
PREDS_FILE  = BASE / "polymarket" / "data" / "predictions.jsonl"
LOG_FILE    = BASE / "bot.log"

RANK_EMOJI = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]


def _latest_scores():
    try:
        lines = SCORES_FILE.read_text().strip().splitlines()
        return json.loads(lines[-1])["rankings"] if lines else []
    except Exception:
        return []


def _recent_predictions(n=20):
    try:
        lines = PREDS_FILE.read_text().strip().splitlines()
        preds = [json.loads(l) for l in lines[-n:]]
        return list(reversed(preds))
    except Exception:
        return []


def _log_tail(n=60):
    try:
        lines = LOG_FILE.read_text().splitlines()
        return lines[-n:]
    except Exception:
        return []


def _ig_trades_from_log():
    trades = []
    try:
        for line in LOG_FILE.read_text().splitlines():
            if "CRYPTO MOMENTUM" in line or "place_order" in line.lower() or "IG CRYPTO" in line:
                trades.append(line)
    except Exception:
        pass
    return trades[-20:]


def _next_scan_secs():
    try:
        for line in reversed(LOG_FILE.read_text().splitlines()):
            m = re.search(r"Next scan in (\d+)s", line)
            if m:
                ts_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                if ts_match:
                    scan_time = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S")
                    elapsed = (datetime.utcnow() - scan_time).total_seconds()
                    remaining = int(m.group(1)) - elapsed
                    return max(0, int(remaining))
    except Exception:
        pass
    return None


def _render():
    scores = _latest_scores()
    preds  = _recent_predictions(20)
    ig     = _ig_trades_from_log()
    log    = _log_tail(60)
    next_s = _next_scan_secs()

    now = datetime.utcnow().strftime("%H:%M:%S UTC")

    rows = ""
    for i, a in enumerate(scores):
        pnl = a["simulated_pnl"]
        pnl_col = "#00ff88" if pnl >= 0 else "#ff4444"
        acc = f"{a['accuracy']*100:.1f}%"
        rows += f"""
        <tr>
          <td>{RANK_EMOJI[i] if i < 5 else i+1}</td>
          <td><b>{a['display_name']}</b> <span style="color:#888">{a['token']}</span></td>
          <td style="color:#00ff88">{a['score']:.2f}</td>
          <td style="color:{pnl_col}">${pnl:+.2f}</td>
          <td>{acc}</td>
          <td>{a['correct']}/{a['predictions']}</td>
        </tr>"""

    pred_rows = ""
    for p in preds:
        outcome = p.get("outcome")
        if outcome == "correct":
            oc = '<span style="color:#00ff88">✓ YES</span>'
        elif outcome == "incorrect":
            oc = '<span style="color:#ff4444">✗ NO</span>'
        else:
            oc = '<span style="color:#888">pending</span>'
        pnl = p.get("simulated_pnl", 0) or 0
        pnl_str = f'<span style="color:{"#00ff88" if pnl>=0 else "#ff4444"}">${pnl:+.2f}</span>' if pnl else ""
        q = p.get("question", "")[:60]
        pred_rows += f"""
        <tr>
          <td style="color:#888;font-size:11px">{p.get('agent','?')}</td>
          <td style="font-size:12px">{q}…</td>
          <td>{p.get('side','?')}</td>
          <td style="font-size:11px">{p.get('market_probability',0):.0%} → {p.get('our_probability',0):.0%}</td>
          <td>{oc}</td>
          <td>{pnl_str}</td>
        </tr>"""

    ig_html = "\n".join(
        f'<div style="font-size:11px;color:{"#00ff88" if "BUY" in l else "#ff9900" if "SELL" in l else "#aaa"};padding:2px 0">{l}</div>'
        for l in ig
    ) or '<div style="color:#555">No IG trades yet</div>'

    log_html = "\n".join(
        f'<div style="font-size:11px;color:{"#ff4444" if "[ERROR]" in l else "#ffcc00" if "[WARNING]" in l else "#aaa"};white-space:nowrap">{l}</div>'
        for l in log
    )

    next_html = f'<span style="color:#00ff88">{next_s}s</span>' if next_s is not None else "—"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Trading Games</title>
<meta http-equiv="refresh" content="10">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0a0a0f; color: #e0e0e0; font-family: 'Courier New', monospace; padding: 20px; }}
  h2 {{ color: #00ff88; margin-bottom: 12px; font-size: 14px; letter-spacing: 2px; text-transform: uppercase; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
  .grid3 {{ display: grid; grid-template-columns: 2fr 1fr; gap: 16px; }}
  .card {{ background: #0f0f1a; border: 1px solid #1a1a2e; border-radius: 6px; padding: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th {{ color: #555; font-weight: normal; text-align: left; padding: 4px 8px; border-bottom: 1px solid #1a1a2e; font-size: 11px; }}
  td {{ padding: 5px 8px; border-bottom: 1px solid #111; }}
  .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
  .title {{ font-size: 20px; color: #fff; font-weight: bold; }}
  .meta {{ color: #555; font-size: 11px; }}
  .log {{ height: 220px; overflow-y: auto; background: #070710; padding: 10px; border-radius: 4px; }}
  .score-big {{ font-size: 28px; color: #00ff88; font-weight: bold; }}
  .stat {{ display: inline-block; margin-right: 24px; }}
  .stat-val {{ font-size: 20px; color: #fff; }}
  .stat-lbl {{ font-size: 10px; color: #555; display: block; }}
</style>
</head>
<body>
<div class="header">
  <div class="title">⚡ TRADING GAMES</div>
  <div class="meta">Day {scores[0]['rank'] if scores else '?'} live &nbsp;|&nbsp; {now} &nbsp;|&nbsp; next scan {next_html} &nbsp;|&nbsp; <span style="color:#555">auto-refresh 10s</span></div>
</div>

<div class="grid">
  <div class="card">
    <h2>🏆 Leaderboard</h2>
    <table>
      <tr><th>Rank</th><th>Agent</th><th>Score</th><th>P&L</th><th>Acc</th><th>Bets</th></tr>
      {rows}
    </table>
  </div>
  <div class="card">
    <h2>⚡ Top Agent Stats</h2>
    {''.join(f'''
    <div style="margin-bottom:16px">
      <div style="color:#888;font-size:11px;margin-bottom:4px">{a["display_name"]} {a["token"]}</div>
      <div class="score-big">{a["score"]:.1f}</div>
      <div style="margin-top:8px">
        <span class="stat"><span class="stat-val" style="color:{"#00ff88" if a["simulated_pnl"]>=0 else "#ff4444"}">${a["simulated_pnl"]:+.2f}</span><span class="stat-lbl">P&L</span></span>
        <span class="stat"><span class="stat-val">{a["accuracy"]*100:.0f}%</span><span class="stat-lbl">accuracy</span></span>
        <span class="stat"><span class="stat-val">{a["predictions"]}</span><span class="stat-lbl">predictions</span></span>
      </div>
    </div>
    ''' for a in scores[:1]) if scores else '<div style="color:#555">No data</div>'}
  </div>
</div>

<div class="grid3">
  <div class="card">
    <h2>📋 Recent Predictions</h2>
    <table>
      <tr><th>Agent</th><th>Market</th><th>Side</th><th>Prob</th><th>Result</th><th>P&L</th></tr>
      {pred_rows or '<tr><td colspan="6" style="color:#555;text-align:center;padding:20px">No predictions yet</td></tr>'}
    </table>
  </div>
  <div class="card">
    <h2>📈 IG Trades</h2>
    {ig_html}
  </div>
</div>

<div class="card" style="margin-top:16px">
  <h2>📡 Live Log</h2>
  <div class="log" id="log">{log_html}</div>
</div>
<script>document.getElementById('log').scrollTop = document.getElementById('log').scrollHeight;</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        html = _render().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8888), Handler)
    print("Dashboard → http://45.32.183.161:8888")
    server.serve_forever()
