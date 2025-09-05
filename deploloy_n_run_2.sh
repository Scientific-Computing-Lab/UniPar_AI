#!/bin/bash
source ~/.bashrc
source /home/tomerbitan/miniconda3/etc/profile.d/conda.sh

conda activate unipar
cd /home/tomerbitan/unipar/UniPar/
IS_DONE=0
CRASH_COUNTER=0
MAX_CRASHES=20

echo "schedualed, starting run:--------------------------------" > ./nohup.out
while [ $IS_DONE -eq 0 ]; do
echo "**looping:**"
# sleep 20

# cd llm_evaluation
./llm_evaluation/deploy_2.sh &
SERVER_PID=$!
# cd ..
# wait for the server to start with looking for: INFO:     Uvicorn running on http://
while ! grep -q "INFO:     Uvicorn running on http://" ./nohup.out; do
    sleep 10
done
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)

# run the llama_inference.py script
# ./llm_evaluation/run_script.sh 0.6 0.9 10000 & > ./llm_evaluation/llama_inference.out
# ./runwhen.sh &
./runwhen.sh & > ./llama_inference2.out
# ./run_few_shot_script.sh & > ./llm_evaluation/llama_inference2.out
RUN_PID=$!
# find name of log file from llama_inference.py
# LOG_FILE=$(grep "logfile_name:" ./llm_evaluation/llama_inference.out | awk '{print $NF}')
while ps -p $SERVER_PID > /dev/null && ps -p $RUN_PID > /dev/null; do
    sleep 300
done 
echo "One of the processes has died or ended"

if grep -q "Too many errors. Exiting now." ./llama_inference2.out; then
    echo "Too many errors. encountered"
    kill $SERVER_PID
    if ps -p $RUN_PID > /dev/null; then
        echo "for some reason run skrip is still running. killing run script"
        kill $RUN_PID
    fi
fi

if grep -q "Finished processing all kernels." ./llama_inference2.out; then
    echo "Finished processing all kernels"
    ###################add loop break here
    IS_DONE=1
fi

if grep -q "crashed:" ./vllm_server_status.txt; then
    echo "server crashed"
    CRASH_COUNTER=$((CRASH_COUNTER + 1))
    kill $RUN_PID
    if ps -p $SERVER_PID > /dev/null; then
        echo "for some reason server is still running. killing run script"
        kill $SERVER_PID
    fi
fi

if grep -q "stopped:" ./vllm_server_status.txt; then
    echo "server stopped for some reason"
    CRASH_COUNTER=$((CRASH_COUNTER + 5))
    kill $RUN_PID
    if ps -p $SERVER_PID > /dev/null; then
        echo "for some reason server is still running. killing run script"
        kill $SERVER_PID
    fi
fi
if [ $CRASH_COUNTER -ge $MAX_CRASHES ]; then
    echo "Reached maximum crash count of $MAX_CRASHES. Terminating script."
    IS_DONE=1
fi

done
echo "done"