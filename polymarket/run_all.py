"""
Cloud launcher — runs bot.py and resolution_checker.py as parallel processes.
Used as the Docker CMD on Railway.
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from threading import Thread

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("launcher")


def run_resolution_checker():
    """Run resolution_checker daily loop in a subprocess thread."""
    while True:
        try:
            logger.info("Starting resolution_checker...")
            result = subprocess.run(
                [sys.executable, "-m", "polymarket.resolution_checker", "--loop"],
                cwd="/app",
            )
            if result.returncode != 0:
                logger.warning("resolution_checker exited with code %d — restarting in 60s", result.returncode)
        except Exception as exc:
            logger.error("resolution_checker error: %s", exc)
        time.sleep(60)


async def run_bot():
    """Run the main bot loop."""
    from polymarket.bot import main
    await main()


if __name__ == "__main__":
    logger.info("=== Polymarket Intelligence Bot — Cloud Launcher ===")
    logger.info("DRY_RUN: %s", os.environ.get("DRY_RUN", "true"))

    # Start resolution checker in background thread
    checker_thread = Thread(target=run_resolution_checker, daemon=True)
    checker_thread.start()
    logger.info("Resolution checker started in background")

    # Run bot in main async loop (restarts on crash)
    while True:
        try:
            asyncio.run(run_bot())
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            break
        except Exception as exc:
            logger.error("Bot crashed: %s — restarting in 30s", exc)
            time.sleep(30)
