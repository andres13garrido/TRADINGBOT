from pathlib import Path
import logging
import sys
from datetime import datetime

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def setup_logging(log_dir, name="bot"):
    ensure_dirs(log_dir)
    logfile = Path(log_dir) / f"{name}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.log"
    
    # Clear existing handlers
    logger = logging.getLogger()
    logger.handlers.clear()
    
    logging.basicConfig(
        filename=str(logfile),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # also print to stdout
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    return logging