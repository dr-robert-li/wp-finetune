set -u
B=http://127.0.0.1:8765; H="Authorization: Bearer s3cret"
start(){ WPJUDGE_API_KEY=s3cret WPJUDGE_MAX_UPLOAD_MB=1 WPJUDGE_JOBS_DIR=jobs venv/bin/uvicorn api:app --port 8765 >>uv.log 2>&1 & echo $! > uv.pid; sleep 2; }
stop(){ kill $(cat uv.pid); sleep 1; }
venv/bin/python hook.py 8766 hooks.log & HP=$!
echo PK > small.zip; head -c 2000000 /dev/urandom > big.zip
start
echo "== 401 no key";      curl -s -o /dev/null -w "%{http_code}\n" $B/jobs/x
echo "== 401 wrong key";   curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer nope" -F file=@small.zip $B/jobs
echo "== 413 oversize";    curl -s -w " %{http_code} %{size_upload}B sent\n" -H "$H" -F file=@big.zip $B/jobs
echo "== 413 chunked";     curl -s -w " %{http_code}\n" -H "$H" -H "Transfer-Encoding: chunked" -F file=@big.zip $B/jobs; ls jobs | wc -l
echo "== 415 bad type";    curl -s -w " %{http_code}\n" -H "$H" -F file=@hook.py $B/jobs
echo "== 404 traversal";   curl -s -w " %{http_code}\n" -H "$H" "$B/jobs/..%2F..%2Fetc"
J=$(curl -s -H "$H" -F file=@small.zip -F webhook_url=http://127.0.0.1:8766/hook $B/jobs | venv/bin/python -c 'import json,sys;print(json.load(sys.stdin)["job_id"])')
J2=$(curl -s -H "$H" -F file=@small.zip $B/jobs | venv/bin/python -c 'import json,sys;print(json.load(sys.stdin)["job_id"])')
echo "== job $J queued; 2nd $J2"; sleep 1.1
echo "== poll mid-run";    curl -s -H "$H" $B/jobs/$J; echo
echo "== 2nd job waits";   curl -s -H "$H" $B/jobs/$J2; echo
echo "== partial";         curl -s -H "$H" $B/jobs/$J/functions; echo
sleep 1.5
echo "== done";            curl -s -H "$H" $B/jobs/$J; echo
echo "== functions count"; curl -s -H "$H" $B/jobs/$J/functions | venv/bin/python -c 'import json,sys;print(len(json.load(sys.stdin)["functions"]))'
sleep 0.5; echo "== webhook got:"; cat hooks.log
echo "== restart while job running"; J3=$(curl -s -H "$H" -F file=@small.zip $B/jobs | venv/bin/python -c 'import json,sys;print(json.load(sys.stdin)["job_id"])')
J4=$(curl -s -H "$H" -F file=@small.zip $B/jobs | venv/bin/python -c 'import json,sys;print(json.load(sys.stdin)["job_id"])')
sleep 2.6; curl -s -H "$H" $B/jobs/$J3; echo; curl -s -H "$H" $B/jobs/$J4; echo
mkdir -p jobs/oldjob && touch -d '8 days ago' jobs/oldjob; touch -d '6 days ago' jobs/$J
stop; start
echo "== after restart";   curl -s -H "$H" $B/jobs/$J3; echo; curl -s -H "$H" $B/jobs/$J4; echo
echo "== retention: oldjob present? $(ls jobs | grep -c oldjob)  6-day job present? $(ls jobs | grep -c $J)"
sleep 3; echo "== J4 after"; curl -s -H "$H" $B/jobs/$J4 | head -c 200; echo
echo "== event loop free during 40MB upload"; head -c 40000000 /dev/urandom > mid.zip
( WPJUDGE_MAX_UPLOAD_MB=50 true ); stop
WPJUDGE_MAX_UPLOAD_MB=50 WPJUDGE_JOBS_DIR=jobs venv/bin/uvicorn api:app --port 8765 >>uv.log 2>&1 & echo $! > uv.pid; sleep 2
curl -s -o /dev/null --limit-rate 20M -F file=@mid.zip $B/jobs & UP=$!; sleep 0.5
for i in 1 2 3; do curl -s -o /dev/null -w "GET during upload: %{http_code} in %{time_total}s\n" $B/jobs/0123456789abcdef0123456789abcdef; sleep 0.3; done
wait $UP; echo "== no-auth mode upload ok"; ls jobs | wc -l
stop; kill $HP
