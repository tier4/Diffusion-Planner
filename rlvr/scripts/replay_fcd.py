#!/usr/bin/env python3
"""
Offline replay of a TeraSim ghost replay using SUMO FCD output.

Reads the fcd_all.xml trajectory file produced by validate_ghost_replay.py
(when --fcd DIR is used) and visualises it with SumoNetVis overlaid on the
Shinagawa-Odaiba SUMO network.

Install SumoNetVis once:
    pip install SumoNetVis

Usage:
    # Replay the most recent FCD file in a directory:
    python3 rlvr/scripts/replay_fcd.py --fcd_dir /tmp/terasim_fcd

    # Replay a specific FCD file:
    python3 rlvr/scripts/replay_fcd.py \\
        --fcd_file /tmp/terasim_fcd/ghost_replay/raw_data/0/<sim_id>/fcd_all.xml

    # Save as an MP4 animation instead of displaying interactively:
    python3 rlvr/scripts/replay_fcd.py --fcd_dir /tmp/terasim_fcd --save replay.mp4
"""

import argparse
import glob
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[2]
_NET_XML = _REPO_ROOT / "rlvr" / "sim_config" / "maps" / "shinagawa_odaiba.net.xml"


def _find_latest_fcd(fcd_dir: str) -> Path:
    """Return the most recently modified fcd_all.xml under fcd_dir."""
    pattern = str(Path(fcd_dir) / "**" / "fcd_all.xml")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"No fcd_all.xml found under {fcd_dir}.\n"
            "Make sure you ran validate_ghost_replay.py with --fcd DIR."
        )
    return Path(max(matches, key=lambda p: Path(p).stat().st_mtime))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline FCD replay viewer using SumoNetVis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--fcd_dir",
        metavar="DIR",
        help="Directory passed to --fcd during validate_ghost_replay.py. "
             "The most recently written fcd_all.xml under this directory is used.",
    )
    group.add_argument(
        "--fcd_file",
        metavar="FILE",
        help="Explicit path to an fcd_all.xml file.",
    )
    parser.add_argument(
        "--net",
        default=str(_NET_XML),
        help=f"Path to SUMO .net.xml file (default: {_NET_XML})",
    )
    parser.add_argument(
        "--save",
        metavar="FILE",
        default=None,
        help="Save animation to this file (e.g. replay.mp4) instead of "
             "showing an interactive window.  Requires ffmpeg.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=100,
        help="Animation frame interval in milliseconds (default: 100 = 10 fps).",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=80.0,
        help="View half-width in metres centred on the AV (default: 80).",
    )
    args = parser.parse_args()

    # Resolve FCD file
    fcd_file = (
        Path(args.fcd_file) if args.fcd_file
        else _find_latest_fcd(args.fcd_dir)
    )
    if not fcd_file.exists():
        sys.exit(f"FCD file not found: {fcd_file}")

    net_file = Path(args.net)
    if not net_file.exists():
        sys.exit(
            f"SUMO net.xml not found: {net_file}\n"
            "Run convert_lanelet2_to_sumo.py first."
        )

    print(f"Network:  {net_file}")
    print(f"FCD file: {fcd_file}")

    # Import SumoNetVis (user must pip install SumoNetVis)
    try:
        import SumoNetVis
    except ImportError:
        sys.exit(
            "SumoNetVis is not installed.\n"
            "Install it with:  pip install SumoNetVis"
        )

    import matplotlib
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    # -----------------------------------------------------------------------
    # Load network and trajectories
    # -----------------------------------------------------------------------
    print("Loading network…")
    net = SumoNetVis.Net(str(net_file))

    print("Loading trajectories…")
    trajectories = SumoNetVis.Trajectories(str(fcd_file))

    vehicle_ids = list(trajectories.keys())
    print(f"  Vehicles in FCD: {vehicle_ids}")

    av_id = next((v for v in vehicle_ids if "AV" in v), None)
    if av_id is None and vehicle_ids:
        av_id = vehicle_ids[0]
        print(f"  No 'AV' found; using first vehicle: {av_id}")

    # Colour trajectories: AV red, others blue
    for vid in vehicle_ids:
        if "AV" in vid:
            trajectories[vid].assign_colors_constant(color="red")
        else:
            trajectories[vid].assign_colors_constant(color="steelblue")

    # -----------------------------------------------------------------------
    # Build figure
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10))
    net.plot(ax=ax)

    # Set initial view centred on first AV position
    if av_id:
        ts = trajectories.timestep_range()
        first_ts = ts[0] if hasattr(ts, '__iter__') else 0
        try:
            pos = trajectories[av_id].position_at(first_ts)
            ax.set_xlim(pos[0] - args.zoom, pos[0] + args.zoom)
            ax.set_ylim(pos[1] - args.zoom, pos[1] + args.zoom)
        except Exception:
            pass

    ax.set_aspect("equal")
    ax.set_title("TeraSim Ghost Replay — FCD offline replay")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # -----------------------------------------------------------------------
    # Animate
    # -----------------------------------------------------------------------
    timesteps = list(trajectories.timestep_range())

    def update(timestep):
        trajectories.plot_points(timestep, ax)
        ax.set_title(f"TeraSim Ghost Replay   t = {timestep:.1f} s")
        # Keep view centred on AV
        if av_id:
            try:
                pos = trajectories[av_id].position_at(timestep)
                ax.set_xlim(pos[0] - args.zoom, pos[0] + args.zoom)
                ax.set_ylim(pos[1] - args.zoom, pos[1] + args.zoom)
            except Exception:
                pass

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=timesteps,
        interval=args.interval,
        repeat=False,
        blit=False,
    )

    if args.save:
        print(f"Saving animation to {args.save}…")
        writer = animation.FFMpegWriter(fps=1000 // args.interval)
        anim.save(args.save, writer=writer)
        print("Done.")
    else:
        print("Showing interactive window (close to exit)…")
        plt.show()


if __name__ == "__main__":
    main()
