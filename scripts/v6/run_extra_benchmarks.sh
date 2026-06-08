#!/bin/bash
# Wait for the current latest_runner to finish, then run gaia2 + swebench only
echo "Waiting for current experiment (latest_runner) to finish..."
while pgrep -f "latest_runner.py" > /dev/null 2>&1; do
    sleep 30
done
echo "Current experiment finished. Starting gaia2 + swebench..."
cd /root/workspace/SkillForge
/root/.conda/envs/skillforge/bin/python -c "
import asyncio, sys, os, json
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
os.environ['LLM_PROVIDER'] = 'codebuddy'
os.environ['CODEBUDDY_MODEL'] = 'deepseek-v4-pro'
os.environ.setdefault('CODEBUDDY_INTERNET_ENVIRONMENT', 'ioa')
from dotenv import load_dotenv; load_dotenv('.env')

from scripts.v6.latest_runner import run_benchmark, RESULTS_DIR
from benchmarks.loader import BenchmarkLoader

async def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    reports = {}
    for name, config in [
        ('gaia2', {'name': 'gaia2', 'num_samples': 50, 'scenario_dir': '/tmp/harbor-datasets/datasets/gaia2-cli'}),
        ('swebench_dynamic', {'name': 'swebench_dynamic', 'num_samples': 30}),
    ]:
        loader = BenchmarkLoader(config)
        tasks = loader.load()[:config['num_samples']]
        print(f'  {name}: {len(tasks)} tasks')
        if tasks:
            try:
                reports[name] = await run_benchmark(name, tasks)
            except Exception as e:
                import traceback; traceback.print_exc()
                reports[name] = {'error': str(e)}

    # Merge with existing summary
    summary_path = f'{RESULTS_DIR}/final_summary.json'
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            existing = json.load(f)
        existing.update(reports)
        reports = existing
    with open(summary_path, 'w') as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    print(f'Saved merged summary to {summary_path}')

asyncio.run(main())
" >> experiments_results/latest/run_extra.log 2>&1
echo "Extra benchmarks done."
