python main.py --model ../../models/opt-125M \
                --sparsity_ratio 0.5 \
                --sparsity_type unstructured \
                --prune_method wanda \
                --save ../../models/wanda-opt-125M-pr50\
                --save_model ../../models/wanda-opt-125M-pr50

python main.py --model ../../models/opt-125M \
                --sparsity_ratio 0.6 \
                --sparsity_type unstructured \
                --prune_method wanda \
                --save ../../models/wanda-opt-125M-pr50\
                --save_model ../../models/wanda-opt-125M-pr60

python main.py --model ../../models/opt-125M \
                --sparsity_ratio 0.7 \
                --sparsity_type unstructured \
                --prune_method wanda \
                --save ../../models/wanda-opt-125M-pr50\
                --save_model ../../models/wanda-opt-125M-pr70

python main.py --model ../../models/opt-125M \
                --sparsity_ratio 0.75 \
                --sparsity_type unstructured \
                --prune_method wanda \
                --save ../../models/wanda-opt-125M-pr50\
                --save_model ../../models/wanda-opt-125M-pr75
