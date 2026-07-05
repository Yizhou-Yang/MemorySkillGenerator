#!/bin/bash
cd /root/workspace/SkillForge
LOGFILE="experiments_results/latest/run_gaia2_term_$(date +%Y%m%d_%H%M%S).log"
/root/.conda/envs/skillforge/bin/python -c "
import asyncio, sys, os
sys.path.insert(0, '.')
sys.path.insert(0, 'src')
os.environ['LLM_PROVIDER'] = 'codebuddy'
os.environ['CODEBUDDY_MODEL'] = 'deepseek-v4-pro'
os.environ.setdefault('CODEBUDDY_INTERNET_ENVIRONMENT', 'ioa')
from dotenv import load_dotenv
load_dotenv('.env')
from scripts.latest.latest_runner import main
asyncio.run(main())
" > "$LOGFILE" 2>&1 &
echo "PID=$! LOG=$LOGFILE"