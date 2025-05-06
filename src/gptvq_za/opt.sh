python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 10 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 2 \
                --vq-dim 2 \
                --groupsize 2048 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --vq-scaling-blocksize 64 \
                --output-dir ../../models/opt-125M-vq \
                ../../models/opt-125M wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 10 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 2 \
                --vq-dim 2 \
                --groupsize 8192 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --vq-scaling-blocksize 64 \
                --zero-aware \
                --output-dir ../../models/opt-125M-pr50-spp-merged-vq \
                ../../models/opt-125M-pr50-spp-merged wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 10 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 3 \
                --vq-dim 1 \
                --groupsize 65536 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --vq-scaling-blocksize 64 \
                --zero-aware \
                --output-dir ../../models/opt-125M-pr50-spp-merged-vq31 \
                ../../models/opt-125M-pr50-spp-merged wikitext2

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
                --output-dir ../../models/opt-125M-pr50-spp-merged-vq32 \
                ../../models/opt-125M-pr50-spp-merged wikitext2
