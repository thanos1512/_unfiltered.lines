# Pinterest pins

Pinterest versions (1000 × 1500, 2:3) of the Unfiltered Lines quotes.

- `*.jpg` — pin images. Pinterest loads them from their public `raw.githubusercontent.com` links.
- `unfiltered-lines-pins-batch-01.csv` — upload file for Pinterest's **Settings → Bulk create Pins** (33 pins, 3 a day from 10 Nov 2026).
- `tools/` — the generator: `pins_data.py` (quotes, boards, titles, descriptions), `render_pins.py` (images) and `build_csv.py` (upload file). Fonts are OFL-licensed (DM Serif Display, Bebas Neue, Inter); licenses are in `tools/fonts/`.

Regenerate:

```
cd pinterest/tools
python3 render_pins.py ..
python3 build_csv.py ../unfiltered-lines-pins-batch-01.csv
```

Boards: Motivational Quotes, Attitude Quotes, Deep Feelings Quotes, Funny Relatable Quotes. Pinterest creates a board automatically if it doesn't exist yet.
