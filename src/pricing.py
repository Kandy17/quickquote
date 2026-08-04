"""QuickQuote pricing v1: polygon area -> service quote.

What this IS: a deterministic rate-card pricer. It takes the polygons that
infer.py extracts, converts pixel area to real-world area using the imagery's
ground sample distance (GSD), and applies a per-square-foot rate for the
selected service.

What this IS NOT (yet): the comp-based pricing engine. The product vision is
to adjust quotes using what neighboring properties paid for similar work.
That requires transaction data we do not have, so comp_adjustment() below is
an explicit no-op stub, not a hidden fake. When comp data exists, it becomes
a learned model keyed on location + recent nearby jobs.

RATE_CARD values are configurable example rates for demo purposes — they are
not market data and make no claim to be.

Usage:
  python src/pricing.py outputs/samples/polygons_00.json --service roofing
  python src/pricing.py outputs/samples/polygons_00.json --service mowing --gsd 0.5
"""
import argparse
import json
import sys
from pathlib import Path

SQM_TO_SQFT = 10.7639

# service -> (eligible polygon class, USD per sqft low, USD per sqft high)
# Example rates for demo purposes only.
RATE_CARD = {
    "roofing":     {"class": "roof",       "low": 4.50, "high": 8.00},
    "mowing":      {"class": "vegetation", "low": 0.01, "high": 0.03},
    "landscaping": {"class": "vegetation", "low": 0.50, "high": 2.00},
}

MIN_QUOTE_USD = 50.0  # small-job floor so a tiny polygon doesn't price at $2


def polygon_area_sqft(area_px: float, gsd_m: float) -> float:
    """Pixel area -> square feet. gsd_m is meters per pixel (LandCover.ai
    imagery is 0.25 or 0.5 m/px depending on the source orthophoto)."""
    return area_px * (gsd_m ** 2) * SQM_TO_SQFT


def comp_adjustment(location=None, service=None) -> float:
    """PLANNED, NOT IMPLEMENTED: neighbor-spend comp multiplier.

    Intended v2: given a location and service, look up recent nearby jobs of
    the same type and scale the base quote toward local market reality.
    Requires historical transaction data that does not exist yet. Until then
    this returns 1.0 so the pricing path is honest about what it uses.
    """
    return 1.0


def quote(polygons: list[dict], service: str, gsd_m: float) -> dict:
    if service not in RATE_CARD:
        raise ValueError(f"Unknown service '{service}'. "
                         f"Options: {sorted(RATE_CARD)}")
    card = RATE_CARD[service]
    eligible = [p for p in polygons if p["class"] == card["class"]]
    area_sqft = sum(polygon_area_sqft(p["area_px"], gsd_m) for p in eligible)
    adj = comp_adjustment()
    low = max(area_sqft * card["low"] * adj, MIN_QUOTE_USD if eligible else 0)
    high = max(area_sqft * card["high"] * adj, MIN_QUOTE_USD if eligible else 0)
    return {
        "service": service,
        "region_class": card["class"],
        "n_regions": len(eligible),
        "area_sqft": round(area_sqft, 1),
        "quote_low_usd": round(low, 2),
        "quote_high_usd": round(high, 2),
        "comp_adjustment": adj,
        "comp_based_pricing": "planned — no transaction data yet",
        "gsd_m_per_px": gsd_m,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("polygons_json", help="polygons_XX.json from src/infer.py")
    ap.add_argument("--service", required=True, choices=sorted(RATE_CARD))
    ap.add_argument("--gsd", type=float, default=0.25,
                    help="meters per pixel of the source imagery (default 0.25)")
    args = ap.parse_args()

    data = json.loads(Path(args.polygons_json).read_text())
    result = quote(data["polygons"], args.service, args.gsd)
    result["tile"] = data.get("tile")
    print(json.dumps(result, indent=2))
    if result["n_regions"] == 0:
        print(f"\nNote: no '{RATE_CARD[args.service]['class']}' regions "
              "found in this tile — nothing to quote.", file=sys.stderr)


if __name__ == "__main__":
    main()
