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