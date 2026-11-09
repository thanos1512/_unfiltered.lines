"""Build the Pinterest bulk-create CSV for the first batch of Unfiltered Lines pins."""
import csv
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pins_data import BOARDS, LINK, PINS  # noqa: E402
from render_pins import pin_filename  # noqa: E402

RAW = "https://raw.githubusercontent.com/thanos1512/_unfiltered.lines/main/pinterest/"
OUT = sys.argv[1]

groups = {"ma": [], "funny": [], "feelings": []}
for p in PINS:
    groups["ma" if p["board"] in ("motivation", "attitude") else p["board"]].append(p)
order = []
for i in range(max(len(g) for g in groups.values())):
    for k in ("ma", "funny", "feelings"):
        if i < len(groups[k]):
            order.append(groups[k][i])

# 3 pins a day (UTC 14:00, 20:00, 02:00 = 7pm, 1am, 7am Pakistan time) starting 10 Nov 2026
slots = []
t = datetime(2026, 11, 10, 14, 0, 0)
cycle = [timedelta(hours=6), timedelta(hours=6), timedelta(hours=12)]
for i in range(len(order)):
    slots.append(t)
    t += cycle[i % 3]

problems = []
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Title", "Media URL", "Pinterest board", "Thumbnail", "Description", "Link", "Publish date", "Keywords"])
    for p, when in zip(order, slots):
        if len(p["title"]) > 100:
            problems.append(f"title too long ({len(p['title'])}): {p['title']}")
        if len(p["desc"]) > 500:
            problems.append(f"description too long ({len(p['desc'])}): {p['title']}")
        w.writerow([
            p["title"],
            RAW + pin_filename(p),
            BOARDS[p["board"]],
            "",
            p["desc"],
            LINK,
            when.strftime("%Y-%m-%dT%H:%M:%S"),
            p["keywords"],
        ])
print(f"rows: {len(order)}  first: {slots[0]}  last: {slots[-1]}")
print("problems:", problems or "none")
