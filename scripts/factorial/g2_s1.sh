#!/bin/bash
source /tmp/factorial_common.sh
export MUTATIONS_PATH=/data/workspace/MemorySkillGenerator/experiments_results/fact2/mutations.json
run_arm fact2/B          B 3 100 gaia2
run_arm fact2/C_ranksim  C 3 100 gaia2 C_RANK=sim
