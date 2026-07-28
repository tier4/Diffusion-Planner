# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for sampling/visualize_cluster_report.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

SAMPLING_DIR = Path(__file__).resolve().parent.parent / "sampling"
sys.path.insert(0, str(SAMPLING_DIR))

from unittest.mock import MagicMock, patch

from visualize_cluster_report import (
    compute_cluster_stats,
    generate_html_report,
    get_args,
    load_cluster_json,
    render_bar_chart,
    render_cluster_videos,
    subsample_cluster_paths,
)


class TestLoadClusterJson:
    def test_loads_valid_json(self, tmp_path):
        data = {
            "cluster_id0": ["/a/0.npz", "/a/1.npz"],
            "cluster_id1": ["/a/2.npz"],
        }
        p = tmp_path / "clusters.json"
        p.write_text(json.dumps(data))
        result = load_cluster_json(str(p))
        assert result == data

    def test_raises_on_missing_file(self):

        with pytest.raises(FileNotFoundError):
            load_cluster_json("/nonexistent/path.json")


class TestComputeClusterStats:
    def test_two_clusters(self):
        clusters = {
            "cluster_id0": [f"/a/{i}.npz" for i in range(90)],
            "cluster_id1": [f"/b/{i}.npz" for i in range(10)],
        }
        stats = compute_cluster_stats(clusters)

        assert len(stats) == 2
        s0 = next(s for s in stats if s["cluster_id"] == "cluster_id0")
        s1 = next(s for s in stats if s["cluster_id"] == "cluster_id1")

        assert s0["count"] == 90
        assert s1["count"] == 10
        assert abs(s0["pct"] - 90.0) < 0.01
        assert abs(s1["pct"] - 10.0) < 0.01

        # Rare cluster should have higher weight
        assert s1["weight"] > s0["weight"]

        # Weights normalized to mean 1.0
        mean_w = (s0["weight"] * 90 + s1["weight"] * 10) / 100
        assert abs(mean_w - 1.0) < 0.01

        # Expected draws: rare cluster gets more than natural 10%
        assert s1["draws_per_epoch"] > 10
        # Expected repeats: rare cluster repeats more per sample
        assert s1["repeats_per_sample"] > s0["repeats_per_sample"]

    def test_sorted_by_cluster_id(self):
        clusters = {
            "cluster_id2": ["/a.npz"],
            "cluster_id0": ["/b.npz"],
            "cluster_id1": ["/c.npz"],
        }
        stats = compute_cluster_stats(clusters)
        ids = [s["cluster_id"] for s in stats]
        assert ids == ["cluster_id0", "cluster_id1", "cluster_id2"]

    def test_alpha_zero_gives_uniform_weights(self):
        clusters = {
            "cluster_id0": [f"/a/{i}.npz" for i in range(90)],
            "cluster_id1": [f"/b/{i}.npz" for i in range(10)],
        }
        stats = compute_cluster_stats(clusters, alpha=0.0)
        for s in stats:
            assert abs(s["weight"] - 1.0) < 1e-6

    def test_alpha_softens_weight_ratio(self):
        clusters = {
            "cluster_id0": [f"/a/{i}.npz" for i in range(90)],
            "cluster_id1": [f"/b/{i}.npz" for i in range(10)],
        }
        full = compute_cluster_stats(clusters, alpha=1.0)
        half = compute_cluster_stats(clusters, alpha=0.5)

        def ratio(stats):
            s0 = next(s for s in stats if s["cluster_id"] == "cluster_id0")
            s1 = next(s for s in stats if s["cluster_id"] == "cluster_id1")
            return s1["weight"] / s0["weight"]

        assert abs(ratio(half) - ratio(full) ** 0.5) < 1e-4

    def test_default_alpha_is_one(self):
        clusters = {
            "cluster_id0": [f"/a/{i}.npz" for i in range(90)],
            "cluster_id1": [f"/b/{i}.npz" for i in range(10)],
        }
        assert compute_cluster_stats(clusters) == compute_cluster_stats(clusters, alpha=1.0)

    def test_negative_alpha_raises(self):

        clusters = {"cluster_id0": ["/a.npz"]}
        with pytest.raises(ValueError, match="alpha"):
            compute_cluster_stats(clusters, alpha=-1.0)

    def test_nan_alpha_raises(self):
        """NaN must be rejected, matching the sampler: it would yield nan weights."""

        clusters = {"cluster_id0": ["/a.npz"]}
        with pytest.raises(ValueError, match="alpha"):
            compute_cluster_stats(clusters, alpha=float("nan"))

    def test_inf_alpha_raises(self):
        """inf must be rejected, matching the sampler: it would render an all-nan table."""

        clusters = {"cluster_id0": ["/a.npz"]}
        with pytest.raises(ValueError, match="alpha"):
            compute_cluster_stats(clusters, alpha=float("inf"))

    def test_report_weight_matches_sampler_weight(self):
        """The report must reproduce the sampler's multiplier exactly."""
        from diffusion_planner.utils.weighted_sampler import (
            ClusterWeightedDistributedSampler,
        )

        data_list = [f"/data/sample_{i}.npz" for i in range(100)]
        clusters = {"cluster_id0": data_list[:10], "cluster_id1": data_list[10:]}
        with tempfile.TemporaryDirectory() as tmp:
            cluster_path = str(Path(tmp) / "clusters.json")
            with open(cluster_path, "w") as f:
                json.dump(clusters, f)

            for alpha in (1.0, 0.5, 0.0):
                sampler = ClusterWeightedDistributedSampler(
                    data_list, cluster_path, num_replicas=1, rank=0, seed=42, alpha=alpha
                )
                stats = compute_cluster_stats(clusters, alpha=alpha)
                for s in stats:
                    assert abs(s["weight"] - sampler.cluster_multipliers[s["cluster_id"]]) < 1e-6, (
                        f"alpha={alpha} cluster={s['cluster_id']}"
                    )


class TestRenderBarChart:
    def test_returns_base64_data_uri(self):
        stats = [
            {
                "cluster_id": "cluster_id0",
                "count": 90,
                "pct": 90.0,
                "weight": 0.5,
                "sampling_rate": 0.5,
                "draws_per_epoch": 50,
                "repeats_per_sample": 0.56,
            },
            {
                "cluster_id": "cluster_id1",
                "count": 10,
                "pct": 10.0,
                "weight": 4.5,
                "sampling_rate": 0.5,
                "draws_per_epoch": 50,
                "repeats_per_sample": 5.0,
            },
        ]
        result = render_bar_chart(stats)
        assert result.startswith("data:image/png;base64,")
        # Should be a reasonable-length base64 string
        assert len(result) > 100


class TestSubsampleClusterPaths:
    def test_caps_at_max_videos(self):
        clusters = {
            "cluster_id0": [f"/a/{i}.npz" for i in range(100)],
        }
        result = subsample_cluster_paths(clusters, max_videos=3, seed=42)
        assert len(result["cluster_id0"]) == 3

    def test_keeps_all_if_under_max(self):
        clusters = {
            "cluster_id0": ["/a/0.npz", "/a/1.npz"],
        }
        result = subsample_cluster_paths(clusters, max_videos=5, seed=42)
        assert len(result["cluster_id0"]) == 2

    def test_deterministic_with_seed(self):
        clusters = {"cluster_id0": [f"/a/{i}.npz" for i in range(50)]}
        r1 = subsample_cluster_paths(clusters, max_videos=3, seed=42)
        r2 = subsample_cluster_paths(clusters, max_videos=3, seed=42)
        assert r1 == r2

    def test_different_seed_different_result(self):
        clusters = {"cluster_id0": [f"/a/{i}.npz" for i in range(50)]}
        r1 = subsample_cluster_paths(clusters, max_videos=3, seed=42)
        r2 = subsample_cluster_paths(clusters, max_videos=3, seed=99)
        assert r1 != r2


class TestRenderClusterVideos:
    def test_calls_render_video_txt_per_cluster(self, tmp_path):
        subsampled = {
            "cluster_id0": ["/a/0.npz", "/a/1.npz"],
            "cluster_id1": ["/b/0.npz"],
        }
        with (
            patch(
                "visualize_cluster_report.shutil.which", return_value="/usr/bin/render-video-txt"
            ),
            patch("visualize_cluster_report.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            render_cluster_videos(subsampled, str(tmp_path), workers=1)
            assert mock_run.call_count == 2

    def test_creates_cluster_subdirectories(self, tmp_path):
        subsampled = {
            "cluster_id0": ["/a/0.npz"],
        }
        with (
            patch(
                "visualize_cluster_report.shutil.which", return_value="/usr/bin/render-video-txt"
            ),
            patch("visualize_cluster_report.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            render_cluster_videos(subsampled, str(tmp_path), workers=1)
            assert (tmp_path / "videos" / "cluster_id0").is_dir()

    def test_checks_render_video_txt_available(self):
        with patch("visualize_cluster_report.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="render-video-txt not found"):
                render_cluster_videos({"cluster_id0": ["/a.npz"]}, "/tmp/out", workers=1)

    def test_collects_errors_from_render_log(self, tmp_path):
        subsampled = {"cluster_id0": ["/a/0.npz"]}
        log_content = '{"file": "a__0", "status": "error", "reason": "corrupt file"}\n'

        def fake_run(cmd, **kwargs):
            log_path = tmp_path / "videos" / "cluster_id0" / "render_log.jsonl"
            log_path.write_text(log_content)
            return MagicMock(returncode=0)

        with (
            patch(
                "visualize_cluster_report.shutil.which", return_value="/usr/bin/render-video-txt"
            ),
            patch("visualize_cluster_report.subprocess.run", side_effect=fake_run),
        ):
            rendered, errors = render_cluster_videos(subsampled, str(tmp_path), workers=1)
            assert len(errors) == 1
            assert errors[0]["cluster_id"] == "cluster_id0"
            assert errors[0]["reason"] == "corrupt file"


class TestGenerateHtmlReport:
    def test_creates_report_html(self, tmp_path):
        stats = [
            {
                "cluster_id": "cluster_id0",
                "count": 90,
                "pct": 90.0,
                "weight": 0.56,
                "sampling_rate": 0.5,
                "draws_per_epoch": 50,
                "repeats_per_sample": 0.56,
            },
            {
                "cluster_id": "cluster_id1",
                "count": 10,
                "pct": 10.0,
                "weight": 4.5,
                "sampling_rate": 0.5,
                "draws_per_epoch": 50,
                "repeats_per_sample": 5.0,
            },
        ]
        chart_uri = "data:image/png;base64,AAAA"
        rendered = {
            "cluster_id0": [str(tmp_path / "videos/cluster_id0/a.mp4")],
            "cluster_id1": [],
        }
        errors = [{"cluster_id": "cluster_id1", "file": "b__0", "reason": "corrupt"}]
        path = generate_html_report(
            stats, chart_uri, rendered, errors, "/fake/cluster.json", str(tmp_path)
        )
        assert Path(path).exists()
        html = Path(path).read_text()
        assert "cluster_id0" in html
        assert "cluster_id1" in html
        assert "90.0" in html
        assert 'preload="none"' in html
        assert "Sampling Behavior" in html
        assert "corrupt" in html

    def test_video_tags_use_relative_paths(self, tmp_path):
        stats = [
            {
                "cluster_id": "cluster_id0",
                "count": 1,
                "pct": 100.0,
                "weight": 1.0,
                "sampling_rate": 1.0,
                "draws_per_epoch": 1,
                "repeats_per_sample": 1.0,
            },
        ]
        vid_path = tmp_path / "videos" / "cluster_id0" / "sample.mp4"
        vid_path.parent.mkdir(parents=True)
        vid_path.touch()
        rendered = {"cluster_id0": [str(vid_path)]}
        path = generate_html_report(
            stats, "data:image/png;base64,AAAA", rendered, [], "/fake/cluster.json", str(tmp_path)
        )
        html = Path(path).read_text()
        assert "videos/cluster_id0/sample.mp4" in html
        # Should NOT contain absolute path
        assert str(tmp_path) not in html.split('src="')[1].split('"')[0]


class TestGetArgs:
    def test_required_args(self):

        with patch("sys.argv", ["prog", "--cluster_json", "/a.json", "--output_dir", "/out"]):
            args = get_args()
            assert args.cluster_json == "/a.json"
            assert args.output_dir == "/out"
            assert args.max_videos == 3
            assert args.workers == 1
            assert args.seed == 42

    def test_optional_args(self):

        with patch(
            "sys.argv",
            [
                "prog",
                "--cluster_json",
                "/a.json",
                "--output_dir",
                "/out",
                "--max_videos",
                "5",
                "--workers",
                "4",
                "--seed",
                "99",
            ],
        ):
            args = get_args()
            assert args.max_videos == 5
            assert args.workers == 4
            assert args.seed == 99

    def test_cluster_weight_alpha_default(self):
        argv = [
            "visualize_cluster_report.py",
            "--cluster_json",
            "/tmp/c.json",
            "--output_dir",
            "/tmp/out",
        ]
        with patch.object(sys, "argv", argv):
            args = get_args()
        assert args.cluster_weight_alpha == 1.0

    def test_cluster_weight_alpha_override(self):
        argv = [
            "visualize_cluster_report.py",
            "--cluster_json",
            "/tmp/c.json",
            "--output_dir",
            "/tmp/out",
            "--cluster_weight_alpha",
            "0.25",
        ]
        with patch.object(sys, "argv", argv):
            args = get_args()
        assert args.cluster_weight_alpha == 0.25


class TestEmptyClusterRows:
    def test_zero_count_clusters_are_dropped(self):
        """An empty cluster would otherwise render weight ~1e8 -- a number training
        never applies, because the sampler builds its multipliers from live matches
        and omits empty clusters entirely. This report is read to pick alpha.
        """
        stats = compute_cluster_stats(
            {"cluster_id0": [f"/a/{i}.npz" for i in range(100)], "cluster_id1": []}
        )
        assert [s["cluster_id"] for s in stats] == ["cluster_id0"]
        assert stats[0]["weight"] == pytest.approx(1.0)

    def test_all_empty_clusters_raise_like_the_sampler(self):
        """ClusterWeightedDistributedSampler raises on an all-empty cluster JSON.
        The report must not exit 0 with "Total samples: 0" -- it exists to catch
        exactly this before a training launch."""
        with pytest.raises(ValueError, match="no paths"):
            compute_cluster_stats({"cluster_id0": [], "cluster_id1": []})
        with pytest.raises(ValueError, match="no paths"):
            compute_cluster_stats({})

    def test_non_numeric_cluster_keys_are_tolerated(self):
        """train.py's log sort deliberately never crashes on odd keys; the report
        must not disagree with a bare ``invalid literal for int()``."""
        stats = compute_cluster_stats(
            {"cluster_id1": ["/a.npz"] * 10, "noise": ["/b.npz"] * 90, "cluster_id0": ["/c.npz"]}
        )
        # cluster_id<N> first in numeric order, non-conforming keys after.
        assert [s["cluster_id"] for s in stats] == ["cluster_id0", "cluster_id1", "noise"]

    def test_subsample_tolerates_non_numeric_keys(self):
        result = subsample_cluster_paths(
            {"noise": [f"/a/{i}.npz" for i in range(10)]}, max_videos=3, seed=42
        )
        assert len(result["noise"]) == 3
        # crc32-based offset, so the pick is reproducible across processes.
        again = subsample_cluster_paths(
            {"noise": [f"/a/{i}.npz" for i in range(10)]}, max_videos=3, seed=42
        )
        assert result == again


class TestRenderLogParsing:
    def _run(self, tmp_path, log_lines):
        subsampled = {"cluster_id0": ["/a/0.npz"]}

        def fake_run(cmd, **kwargs):
            log = Path(cmd[2]) / "render_log.jsonl"
            log.write_text(log_lines)
            return MagicMock(returncode=0)

        with (
            patch(
                "visualize_cluster_report.shutil.which", return_value="/usr/bin/render-video-txt"
            ),
            patch("visualize_cluster_report.subprocess.run", side_effect=fake_run),
        ):
            return render_cluster_videos(subsampled, str(tmp_path), workers=1)

    def test_blank_lines_do_not_discard_a_completed_render(self, tmp_path):
        """render-video-txt is external; a trailing blank line must not raise
        JSONDecodeError after every video has already been rendered."""
        _, errors = self._run(
            tmp_path, '{"status": "error", "file": "/a/0.npz", "reason": "corrupt"}\n\n'
        )
        assert len(errors) == 1
        assert errors[0]["reason"] == "corrupt"

    def test_malformed_line_warns_and_continues(self, tmp_path):
        with pytest.warns(UserWarning, match="malformed render_log"):
            _, errors = self._run(
                tmp_path,
                'not json\n{"status": "error", "file": "/a/0.npz", "reason": "corrupt"}\n',
            )
        assert len(errors) == 1
