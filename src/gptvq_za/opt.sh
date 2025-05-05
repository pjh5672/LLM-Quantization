python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 10 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 2 \
                --vq-dim 2 \
                --groupsize 4096 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --vq-scaling-blocksize 64 \
                ../../models/opt-125M wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 10 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 3 \
                --vq-dim 2 \
                --groupsize 16384 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --vq-scaling-blocksize 64 \
                --zero-aware \
                ../../models/opt-125M-pr50 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 10 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 3 \
                --vq-dim 2 \
                --groupsize 32768 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --vq-scaling-blocksize 64 \
                --zero-aware \
                ../../models/opt-125M-pr50 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 10 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 3 \
                --vq-dim 2 \
                --groupsize 65536 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --vq-scaling-blocksize 64 \
                --zero-aware \
                ../../models/opt-125M-pr50 wikitext2