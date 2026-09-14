.PHONY: test backtest stress robustness run demo doctor clean

test:            ## run the full test suite (no dependencies needed)
	python3 -m unittest discover -s tests -v

backtest:        ## single backtest with a full statistical report
	python3 -m kanenas backtest --bars 5000 --seed 2024

stress:          ## try to break the strategy with hostile markets
	python3 -m kanenas stress --runs 8 --bars 3000

robustness:      ## same strategy across many independent random markets
	python3 -m kanenas robustness --runs 24 --bars 4000

run:             ## live paper session, terminal + web dashboards
	python3 -m kanenas run --speed 8 --open

demo:            ## quick bounded paper session (no browser)
	python3 -m kanenas run --bars 1200 --speed 0 --no-web

doctor:          ## check environment and venue reachability
	python3 -m kanenas doctor

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
