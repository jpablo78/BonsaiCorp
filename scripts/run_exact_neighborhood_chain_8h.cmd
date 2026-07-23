@echo off
cd /d C:\AlixPartners
if not exist output_exact_neighborhood_chain_8h_v3 mkdir output_exact_neighborhood_chain_8h_v3
.\.venv\Scripts\python.exe scripts\run_exact_neighborhood_chain.py --data-dir . --warm-start output_lp_pool_after_ba_15m\asignacion_optima.csv --output-dir output_exact_neighborhood_chain_8h_v3 --total-time-seconds 28800 --num-threads 6 --memory-limit-mb 12000 --max-extra-pallets 5000 --random-seed 20260721 --stop-at-target > output_exact_neighborhood_chain_8h_v3\run.log 2>&1
