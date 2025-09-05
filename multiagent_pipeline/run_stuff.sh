#!/bin/bash
# python datasets_after_eval_compile_main.py --dataset_dir ~/unipar/Datasets/eval/gpt_shot_0_m_15000_t_0.2_p_0.9_new --from_api all --to_api omp >& run_log_gpt_cuda.out
# echo "done 1"
# python datasets_after_eval_compile_main.py --dataset_dir ~/unipar/Datasets/eval/eval_shot_0_m_15000_t_0.2_p_0.9_new --from_api all --to_api cuda >& run_log_gpt_cuda.out
# echo "done 2"
# python datasets_after_eval_compile_main.py --dataset_dir ~/unipar/Datasets/eval/eval_shot_0_m_15000_t_0.2_p_0.9_new --from_api all --to_api omp >& run_log_llama_omp.out
# echo "done 3"
# python datasets_after_eval_compile_main.py --dataset_dir ~/unipar/Datasets/eval/eval_shot_0_m_15000_t_0.2_p_0.9_new --from_api all --to_api cuda >& run_log_llama_cuda.out
# echo "done 4"

#this is in 18 till end
# python datasets_after_eval_compile_main.py --dataset_dir ~/unipar/Datasets/eval/FT_llama70_gen_shot1_new/ --from_api all --to_api cuda >& run_log_FT1_cuda.out
# echo "done 5"
# python datasets_after_eval_compile_main.py --dataset_dir ~/unipar/Datasets/eval/FT_llama70_gen_shot1_new/ --from_api all --to_api omp >& run_log_FT1_omp.out
# echo "done 6"

#this is in 13
# python datasets_after_eval_compile_main.py --dataset_dir ~/unipar/Datasets/eval/FT_llama70_gen_new/ --from_api all --to_api cuda >& run_log_FT_cuda.out
# echo "done 7"
# python datasets_after_eval_compile_main.py --dataset_dir ~/unipar/Datasets/eval/FT_llama70_gen_new/ --from_api all --to_api omp >& run_log_FT_omp.out
# echo "done all"
#python datasets_after_eval_compile_main.py --dataset_dir /home/tomerbitan/unipar/Datasets/eval/eval_shot_0_m_15000_t_0.2_p_0.9_to_replace --from_api all --to_api omp >& run_log_llama_replace_2.out                                            
#python datasets_after_eval_compile_main.py --dataset_dir /home/tomerbitan/unipar/Datasets/eval/gpt_shot_0_m_15000_t_0.2_p_0.9_to_replace --from_api all --to_api omp >& run_log_gpt_replace.out                                            
#python datasets_after_eval_compile_main.py --dataset_dir /home/tomerbitan/unipar/Datasets/eval/gpt_shot_0_m_15000_t_0.2_p_0.9_to_replace --from_api all --to_api cuda >& run_log_gpt_replace_2.out                                            

# python datasets_after_eval_compile_main.py --dataset_dir /home/tomerbitan/unipar/Datasets/eval/FT_llama70_gen_omp_cuda_only_replace_agent --from_api all --to_api omp >& run_log_ft_omp_cuda_only_replace_omp.out
# echo "done sleep"
# sleep 18000
# python datasets_after_eval_compile_main.py --dataset_dir /home/tomerbitan/unipar/Datasets/eval/gpt_shot_0_m_15000_t_0.2_p_0.9_new_new_agent --from_api all --to_api omp >& run_log_gpt_agent.out
python datasets_after_eval_compile_main.py --dataset_dir /home/tomerbitan/unipar/Datasets/eval/eval_shot_0_m_15000_t_0.2_p_0.9_new_new_agent --from_api all --to_api cuda >& run_log_33eval_agent_cuda.out

#eval cuda!!!!!!!!!!