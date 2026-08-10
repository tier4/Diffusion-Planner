import pandas as pd

from human_match_prototype.render_report import render_html_report


def _fake_review_set(n: int = 6) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "npz_path": f"/path/frame_{i:05d}.npz",
                "R_overall": 95.0 - i,
                "R_lateral": 90.0 - i * 2,
                "selection_reason": "top_overall" if i < 3 else "top_lateral",
                "es_2s": 10.0 + i,
                "es_4s": 15.0 + i,
                "es_8s": 20.0 + i,
            }
        )
    return pd.DataFrame(rows)


class TestRenderHtmlReport:
    def test_produces_html(self, tmp_path):
        review = _fake_review_set()
        ranked = _fake_review_set(20)
        # Create dummy overlay PNGs
        overlay_pngs = []
        for i in range(len(review)):
            png = tmp_path / f"overlay_{i}.png"
            png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            overlay_pngs.append(png)
        dist_png = tmp_path / "distributions.png"
        dist_png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        out = tmp_path / "report.html"
        render_html_report(
            review,
            ranked,
            overlay_pngs,
            dist_png,
            out,
            {
                "temperature": 1.0,
                "seed": 0,
                "num_samples": 64,
                "n_scenes": 500,
            },
        )
        assert out.exists()
        html = out.read_text()
        assert "review candidate" in html.lower() or "Review" in html
        assert "base64" in html  # images embedded
