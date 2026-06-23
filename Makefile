# CAFE — convenience targets. The reproduction entrypoint (bench/repro.py) discovers
# every experiment generator, prints the experiment -> paper-artifact manifest, and
# (with repro-run) regenerates all tables/figures from a live model run.

.PHONY: repro repro-run repro-json paper test

# Dry / discovery: print the manifest, run nothing (safe, fast, side-effect free).
repro:
	python3 bench/repro.py

# Execute every discovered generator, then verify each declared output now exists.
repro-run:
	python3 bench/repro.py --run

# Machine-readable manifest (JSON) for CI / reproducibility checks.
# @-prefixed so `make repro-json | jq` receives clean JSON (no echoed recipe line).
repro-json:
	@python3 bench/repro.py --json

# Regenerate all data-derived paper artifacts (parallel) the canonical way.
paper:
	python3 bench/make_paper.py

test:
	pytest -q
