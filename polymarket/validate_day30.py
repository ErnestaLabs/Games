"""
Day 30 validation — run this to decide whether to go live.

  python polymarket/validate_day30.py

Reads from local polymarket/data/predictions.jsonl (no graph required).
Also prints the Cypher query to run against FalkorDB directly if needed.
"""

from __future__ import annotations

from polymarket.resolution_checker import ResolutionChecker
from polymarket.prediction_store import PredictionStore

CYPHER_QUERY = """
MATCH (p:PredictionRecord)
WHERE p.outcome IS NOT NULL
RETURN
  count(p) AS total_predictions,
  avg(CASE WHEN p.outcome = 'correct' THEN 1.0 ELSE 0.0 END) AS accuracy,
  sum(p.simulated_pnl) AS total_simulated_pnl,
  avg(abs(p.our_probability - p.market_probability)) AS avg_edge_taken
"""

GO_LIVE_RULES = """
GO LIVE if:
  accuracy    > 0.55   (beating coin-flip by meaningful margin)
  simulated_pnl > 0    (positive expected value across all bets)

If NOT met:
  - Review edge_calculator.py signal weights
  - Raise MIN_CAUSAL_WEIGHT (e.g. 0.60 → 0.65)
  - Raise MIN_EDGE_THRESHOLD (e.g. 0.08 → 0.10)
  - Run another 30-day cycle
"""


def main() -> None:
    # First run resolution check to catch any newly resolved markets
    print("Checking for newly resolved markets...")
    checker = ResolutionChecker()
    stats = checker.check_all()
    checker.close()

    # Print full summary
    store = PredictionStore()
    records = store.load_all()
    resolved = [r for r in records if r.get("outcome") is not None]
    unresolved = [r for r in records if r.get("outcome") is None]
    store.close()

    if not resolved:
        print(f"\nNo resolved predictions yet ({len(unresolved)} still pending).")
        print("Come back when markets have resolved.")
        return

    correct = [r for r in resolved if r["outcome"] == "correct"]
    accuracy = len(correct) / len(resolved)
    total_pnl = sum(r.get("simulated_pnl") or 0.0 for r in resolved)
    avg_edge = sum(abs(r.get("edge") or 0.0) for r in resolved) / len(resolved)

    go_live = accuracy > 0.55 and total_pnl > 0

    print(f"""
╔══════════════════════════════════════════════════════╗
║          POLYMARKET BOT — DAY 30 VALIDATION          ║
╠══════════════════════════════════════════════════════╣
║ Total predictions:    {len(records):>6}                      ║
║ Resolved:             {len(resolved):>6}                      ║
║ Pending:              {len(unresolved):>6}                      ║
║                                                      ║
║ Accuracy:             {accuracy:>6.1%}    (need > 55.0%)  ║
║ Simulated P&L:        ${total_pnl:>+9.2f}               ║
║ Avg edge taken:       {avg_edge:>6.1%}                      ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  {'✅  GO LIVE — flip DRY_RUN=false, start with $200-500 USDC' if go_live else '❌  NOT YET — tune signal weights, run another cycle':<52} ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
""")

    if go_live:
        print("Next step:")
        print("  1. Set DRY_RUN=false in your env")
        print("  2. Fund Polygon wallet with $200-500 USDC")
        print("  3. python polymarket/bot.py")
    else:
        print("Tuning suggestions:")
        if accuracy <= 0.55:
            print("  - Raise MIN_CAUSAL_WEIGHT (currently env MIN_CAUSAL_WEIGHT)")
            print("  - Raise MIN_EDGE_THRESHOLD")
        if total_pnl <= 0:
            print("  - Check if high-fee markets are dominating incorrect predictions")
            print("  - Prioritise IS_FEE_FREE=true markets only (set MIN_LIQUIDITY_USD higher)")

    print("\nFalkorDB Cypher query (run directly against graph if needed):")
    print(CYPHER_QUERY)
    print(GO_LIVE_RULES)


if __name__ == "__main__":
    main()
