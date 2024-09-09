python run_eval_rag.py \
    --model_name_or_url='http://127.0.0.1:5000/v1/' \
    --filepath='data/test1_sample.csv' \
    --cohere_n=5 \
    --suffix='_mixtral8x7B_sample'
    
python run_eval_rag.py \
    --model_name_or_url='http://127.0.0.1:5000/v1/' \
    --filepath='data/test1.csv' \
    --cohere_n=5 \
    --suffix='_mixtral8x7B_n_5'

python run_eval_rag.py \
    --model_name_or_url='gpt-4-1106-preview' \
    --filepath='data/test1.csv' \
    --cohere_n=5 \
    --suffix='_gpt4_n_5'
    
python run_eval_rag.py \
    --model_name_or_url='gpt-4-1106-preview' \
    --filepath='data/test1_sample.csv' \
    --cohere_n=5 \
    --suffix='_take2'


# Ablation
python run_eval_rag.py \
    --model_name_or_url='gpt-4-1106-preview' \
    --filepath='data/bcsc_question_bank_sampled_v1.csv' \
    --cohere_n=None \
    --chromadb_k=5 \
    --suffix='_gpt4_without_cohere_chromak_5'

python run_eval_rag.py \
    --model_name_or_url='gpt-4-1106-preview' \
    --filepath='data/bcsc_question_bank_sampled_v1.csv' \
    --cohere_n=None \
    --chromadb_k=30 \
    --suffix='_gpt4_without_cohere'

python run_eval_rag.py \
    --model_name_or_url='http://127.0.0.1:5000/v1/' \
    --filepath='data/bcsc_question_bank_sampled_v1.csv' \
    --cohere_n=None \
    --chromadb_k=5 \
    --suffix='_mixtral8x7B_n_5_without_cohere_chromak_5'

python run_eval_rag.py \
    --model_name_or_url='http://127.0.0.1:5000/v1/' \
    --filepath='data/bcsc_question_bank_sampled_v1.csv' \
    --cohere_n=None \
    --chromadb_k=30 \
    --suffix='_mixtral8x7B_n_5_without_cohere'


python run_eval_rag.py \
    --model_name_or_url='http://127.0.0.1:5000/v1/' \
    --filepath='data/ophthoquestions/oq_sampled.csv' \
    --cohere_n=5 \
    --chromadb_k=30 \
    --suffix='_mixtral8x7B_n_5' \
    --save_dir='results/sampled'



python run_eval_rag.py \
    --model_name_or_url='http://127.0.0.1:5000/v1/' \
    --filepath='data/ophthoquestions/oq_sampled.csv' \
    --cohere_n=5 \
    --chromadb_k=30 \
    --suffix='_mixtral8x7B_n_5' \
    --save_dir='results/sampled'



python run_eval_rag.py \
    --model_name_or_url='gpt-4-turbo' \
    --filepath='data/ophthoquestions/oq_sampled.csv' \
    --cohere_n=5 \
    --chromadb_k=30 \
    --suffix='_gpt4_turbo_n_5' \
    --save_dir='results/sampled'

python run_eval_rag.py \
    --model_name_or_url='http://127.0.0.1:5000/v1/' \
    --filepath='data/bcsc/bcsc_sampled.csv' \
    --cohere_n=5 \
    --chromadb_k=30 \
    --suffix='_mixtral8x7B_n_5' \
    --save_dir='results/sampled'

python run_eval_rag.py \
    --model_name_or_url='gpt-4-turbo' \
    --filepath='data/bcsc/bcsc_sampled.csv' \
    --cohere_n=5 \
    --chromadb_k=30 \
    --suffix='_gpt4_turbo_n_5' \
    --save_dir='results/sampled'


python run_eval_rag.py \
    --model_name_or_url='http://127.0.0.1:5000/v1/' \
    --filepath='data/ophthoquestions/oq_sampled.csv' \
    --cohere_n=5 \
    --chromadb_k=30 \
    --suffix='_mixtral8x7B_n_cot' \
    --save_dir='results/sampled'

MODEL_NAME="Llama-3-70B-Q8"
python run_eval_rag.py \
    --model_name_or_url='http://127.0.0.1:5000/v1/' \
    --filepath='data/bcsc/bcsc_sampled.csv' \
    --cohere_n=5 \
    --chromadb_k=30 \
    --suffix="_${MODEL_NAME}_n5-cot" \
    --save_dir='results/sampled'

python run_eval_rag.py \
    --model_name_or_url='http://127.0.0.1:5000/v1/' \
    --filepath='data/ophthoquestions/oq_sampled.csv' \
    --cohere_n=5 \
    --chromadb_k=30 \
    --suffix="_${MODEL_NAME}_n5-cot" \
    --save_dir='results/sampled'