import json

def main():
    file_path = "/home/tomerbitan/unipar/UniPar/data/Datasets/HeCBench/prompt_Bioinformatics.jsonl"

    with open(file_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get("parallel_api") == "omp":
                    print(record.get("code").get("main.cpp", ""))
            except json.JSONDecodeError:
                continue

if __name__ == "__main__":
    main()