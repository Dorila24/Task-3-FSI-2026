# Multimodal Inference Simulation Project

## Requirements
- Python 3.10+.
- No external dependencies (stdlib only).

Version check:
```bash
python3 --version
```

## Full Run (1,000,000 requests)
```bash
cd <project-root>
python3 simulate_server.py --verbose --max-n 4096 --output-json simulation_results_full.json
```

The command performs:
- full trace loading,
- minimal feasible accelerator search (`N`),
- TTFT/T statistics calculation,
- telemetry export,
- reproducibility metadata export.

## Quick Smoke Test
```bash
cd <project-root >
python3 simulate_server.py --max-requests 50000 --verbose --max-n 512 --output-json simulation_results_50k.json
```

## Key CLI Parameters
- Cost and hardware: `--A --B --C --K --M`
- Memory footprint: `--X --Y --Z`
- SLA limits: `--P --D`
- Batch-cost shape: `--cost-shape` (`sqrt`, `cuberoot`, `power:<p>`)
- Assignment policy: `--assignment` (`balanced` or `round-robin`)
- Stage scheduler: `--scheduler` (`edf` or `slack`)
- Batch limits: `--stage1-max-requests --stage2-max-requests --stage3-max-requests --stage1-max-units --stage2-max-units --stage3-max-units`

Example alternative configuration:
```bash
cd <project-root >
python3 simulate_server.py \
  --P 4 --D 10 --K 100 \
  --cost-shape cuberoot \
  --scheduler slack \
  --output-json alt_config_result.json
```
