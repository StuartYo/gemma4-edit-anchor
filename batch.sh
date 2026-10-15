#!/bin/bash
# Full experiment queue. Holds E:/_tmp_claude/render.lock for the whole run.
cd E:/projects/gemma4-edit-anchor
LL=E:/tools/llama.cpp/b11215
start_server () {  # $1 = gguf path
  powershell -NoProfile -Command "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force" ; sleep 3
  powershell -NoProfile -Command "\$p = Start-Process -FilePath '$LL/llama-server.exe' -ArgumentList '-m','$1','--port','8091','-c','49152','-ngl','99','-t','4','--jinja','--no-webui','-np','3' -RedirectStandardError 'E:\_tmp_claude\gemma4paper\server.log' -RedirectStandardOutput 'E:\_tmp_claude\gemma4paper\server.out' -PassThru -WindowStyle Hidden; \$p.PriorityClass='BelowNormal'"
  for i in $(seq 1 60); do curl -s localhost:8091/health | grep -q ok && break; sleep 3; done
}
run () { python run_exp.py --tasks data/$2.jsonl --model $1 --cond $3 --mode $4 --seed ${5:-1} --out runs/$1_$2_$4_$3_s${5:-1}.jsonl > /dev/null 2>>runs/batch_err.log; echo "$(date +%H:%M) done $1 $2 $4 $3 s${5:-1}"; }
mkdir -p runs
start_server E:/models/gemma4/gemma-4-E2B_q4_0-it.gguf
run E2B dev100 S target & run E2B dev100 RN target & run E2B dev100 N target & wait
run E2B dev100 R target & run E2B holdout60 S target & run E2B holdout60 RN target & wait
run E2B dev100 S intent
start_server E:/models/gemma4/gemma-4-E4B_q4_0-it.gguf
run E4B dev100 S target & run E4B dev100 RN target & run E4B holdout60 S target & wait
run E4B holdout60 RN target & run E4B dev100 S intent & run E4B dev100 N target & wait
run E4B dev100 R target
powershell -NoProfile -Command "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force"
rm -f E:/_tmp_claude/render.lock
echo ALLDONE
