#!/bin/bash
# Self-restarting wrapper for judge batch generation.
# Uses checkpoint/resume to survive process kills.
# Usage: nohup bash scripts/run_judge_batch_loop.sh > /tmp/judge_batch_4.log 2>&1 &

cd /home/robert_li/Desktop/projects/wp-finetune

TARGET=3000
LOGFILE=/tmp/judge_batch_4.log

while true; do
    # Check current count
    COUNT=$(python -c "
import json
try:
    d=json.load(open('data/phase2_synthetic/output/judge_training/judge_training_calibrated_high_4.json'))
    print(len(d))
except:
    print(0)
" 2>/dev/null)

    echo "[$(date)] Current count: $COUNT / $TARGET"

    if [ "$COUNT" -ge "$TARGET" ]; then
        echo "[$(date)] TARGET REACHED: $COUNT examples"
        break
    fi

    # Run the batch script (will resume from checkpoint)
    echo "[$(date)] Starting batch run..."
    PYTHONUNBUFFERED=1 timeout 540 python -u -m scripts.generate_judge_batch --source passed --batch 4 --count $TARGET 2>&1
    EXIT_CODE=$?

    echo "[$(date)] Batch exited with code $EXIT_CODE"

    # Brief pause before restart
    sleep 5
done

# Final summary
echo ""
echo "=========================================="
echo "GENERATION COMPLETE"
echo "=========================================="
python -c "
import json
d=json.load(open('data/phase2_synthetic/output/judge_training/judge_training_calibrated_high_4.json'))
scores=[x['metadata']['overall_score'] for x in d]
print(f'Total examples: {len(d)}')
print(f'Score range: {min(scores):.0f}-{max(scores):.0f}')
print(f'Average score: {sum(scores)/len(scores):.1f}')

# Distribution
buckets = {}
for s in scores:
    bucket = int(s // 10) * 10
    buckets[bucket] = buckets.get(bucket, 0) + 1
for k in sorted(buckets):
    print(f'  {k}-{k+9}: {buckets[k]} ({buckets[k]*100/len(scores):.1f}%)')
"
