# Polymarket Intelligence Playbook
> Sources: 2025–2026 only. (a) proven on-chain, (b) claimed, (c) theoretical.
> UK (GB) is geoblocked — see Execution Options below.

## Market Size
- $44B volume 2025 | $2B ICE investment | $9B valuation | $21B/month March 2026

---

## Category Edge Map

| Category | Maker-Taker Gap | Action |
|---|---|---|
| World Events | 7.32pp | PRIMARY TARGET — Forage entity graph has structural advantage |
| Media | 7.28pp | PRIMARY TARGET — Forage people-graph detects signal early |
| Entertainment | 4.79pp | Secondary |
| Crypto | 2.69pp | On-chain metrics give edge |
| Sports | 2.23pp | SKIP — no Forage advantage |
| Politics | 1.02pp | SKIP — well-calibrated community |
| Finance | 0.17pp | AVOID — near-efficient |

---

## Proven Edges (on-chain)

### 1. Cross-Platform Arb (a)
- $39.59M extracted Apr 2024–Apr 2025; top address $2.01M across 4,049 transactions
- Bitcoin Reserve: 14-cent spread (51% Polymarket vs 37% Kalshi) persisting hours
- Tool: `pmxt-dev/pmxt` — unified SDK covering Polymarket, Kalshi, Limitless, Opinion

### 2. Insider Copy-Follow (a)
- 210,718 suspicious wallet-market pairs; 69.9% win rate; ~$143M anomalous profit
- Tools: `suislanchez/polymarket-insider-detector` + `pselamy/polymarket-insider-tracker`
- Tactic: copy-follow 30-60s delay, 1/3 of position size, $50 test first

### 3. Optimism Tax Market-Making (a)
- NO contracts outperform YES by up to 64pp at equivalent prices
- Sell NO on longshot YES (1–15¢) in World Events / Media / Entertainment
- Kelly 2-3% per position, no directional view needed

### 4. Information Arbitrage (b)
- Average crowd update lag: 29 minutes post-event
- Forage `search_web` + `scrape_page` on primary sources (FDA.gov, SEC EDGAR) at 30s poll = faster than Reuters aggregators
- Edge: 35–95% annual for skilled practitioners

---

## Whale Detection Framework

**Fresh wallet signals:**
1. Wallet < 7 days old + first Polymarket trade
2. Single large USDC inflow from bridge/exchange
3. $5K+ position in niche market (< $100K volume), against current odds
4. Multiple fresh wallets, same direction, same market, within 30min window

**Scoring (suislanchez method):**
- Binomial p-value < 0.001 = highly suspicious
- Win rate > 80% = flag
- Check Polygonscan for USDC funding cluster

**Documented cases:**
- `0xafEe` (AlphaRaccoon): Google Year in Search, 22/23, $150K on exact Gemini release date
- Magamyman: $553K, 71min before military strike at 17% implied probability

---

## Signal → Position Pipeline

```
1. Gamma API scan (every 5min) — World Events/Media/Biotech, vol > $50K
2. Forage entity score (skill_job_signals, find_connections, get_signals) — threshold 6/10
3. Insider detection — suislanchez + pselamy on $5K+ trades last 24h
4. MiroFish sim (when score > 6) — 1,000 agents, 30 rounds, P(YES) output
5. Polyseer Bayesian synthesis — parallel to sim
6. Combined prob = 40% MiroFish + 40% Polyseer + 20% insider signal
7. Half-Kelly sizing, max 5% wallet, minimum 8pp divergence from market
8. Execution via py-clob-client (limit order 1-2¢ inside best ask)
9. FalkorDB log: signals, sim output, outcome, P&L
```

---

## Forage → Market Signal Map

| Forage Tool | Signal | Lead Time |
|---|---|---|
| `skill_job_signals` | Biotech regulatory hire spike → FDA approval | Days–weeks |
| `skill_company_dossier` | Executive calendar clear → press conference | 24–72h |
| `skill_competitor_ads` | Competitor spend drop → acquisition target | Days |
| `find_connections` | Legal firm retained → M&A/regulatory action | Days–weeks |
| `get_signals` | LinkedIn anomaly → regulatory signal | 24–48h |

---

## UK Execution Options

UK (GB) is **blocked** from placing orders on Polymarket CLOB. Gamma API (read) is fine.

| Option | Risk | Recommended |
|---|---|---|
| **Kalshi via pmxt** (legal UK alternative) | None — fully regulated | YES — primary execution |
| Railway deployment (Ireland, eu-west-1) | ToS risk if linked to UK identity | Use separate wallet |
| Direct Polygon RPC (no CLOB API) | Grey area — on-chain only | Fallback |

**Recommended architecture:**
- Quinn/agents trade on **Kalshi from London** (legal, pmxt SDK)
- Monitor Polymarket via public Gamma API for price comparison
- Cross-platform arb = Kalshi execution + Polymarket signal = **your primary alpha**
- Kalshi/Polymarket settlement divergence lasts 12–24h on major events vs 2.7s bot races

---

## Open-Source Arsenal

| Tool | Install | Use |
|---|---|---|
| `py-clob-client` | `pip install py-clob-client` | Polymarket CLOB (from non-UK server) |
| `polymarket-gamma` | `pip install polymarket-gamma` | Market discovery (no geoblock) |
| `pmxt-dev/pmxt` | `npm install pmxt` | **Cross-platform arb: Polymarket+Kalshi+Opinion** |
| `suislanchez/polymarket-insider-detector` | GitHub | Insider detection: p-value + timing |
| `pselamy/polymarket-insider-tracker` | GitHub | Real-time alerts + Telegram |
| `yorkeccak/polyseer` | GitHub | Bayesian probability synthesis |
| `TauricResearch/TradingAgents` | GitHub | 7-role trading firm agent |

**Data:** Jon Becker 36GB dataset — 400M+ trades, MIT licensed, free via Cloudflare R2

---

## Key People
- **Jon Becker** (@beckerrjon) — 36GB dataset, Optimism Tax paper
- **Joshua Mitts** — Harvard/Columbia insider trading study ($143M anomalous profit)
- **suislanchez** — insider detector (steal their binomial p-value scoring)
- **pselamy** — insider tracker (wire to TheWatcherSees alerts)
- **Shayne Coplan** (@shayne_coplan) — Polymarket CEO (POLY token + ICE integration)

---

## Railway Deployment (from non-UK server)

```bash
# In project root — run interactively:
railway add --service          # name it "trading-games"
railway up                     # deploys from current directory
```

Set env vars in Railway dashboard. Region: eu-west-1 (Ireland = unblocked).
