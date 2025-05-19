python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 2 \
                --vq-dim 1 \
                --groupsize 256 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --output-dir ../../models/opt-125M-w2d1 \
                ../../models/opt-125M wikitext2

##########################################################################################################

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 2 \
                --vq-dim 1 \
                --groupsize 256 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --output-dir ../../models/wanda-opt-125M-pr50-w2d1 \
                ../../models/wanda-opt-125M-pr50 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 2 \
                --vq-dim 1 \
                --groupsize 256 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --output-dir ../../models/wanda-opt-125M-pr60-w2d1 \
                ../../models/wanda-opt-125M-pr60 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 2 \
                --vq-dim 1 \
                --groupsize 256 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --output-dir ../../models/wanda-opt-125M-pr70-w2d1 \
                ../../models/wanda-opt-125M-pr70 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 2 \
                --vq-dim 1 \
                --groupsize 512 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/wanda-opt-125M-pr50-sp-w2d1 \
                ../../models/wanda-opt-125M-pr50 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 2 \
                --vq-dim 1 \
                --groupsize 512 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/wanda-opt-125M-pr60-sp-w2d1 \
                ../../models/wanda-opt-125M-pr60 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 2 \
                --vq-dim 1 \
                --groupsize 512 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/wanda-opt-125M-pr70-sp-w2d1 \
                ../../models/wanda-opt-125M-pr70 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 2 \
                --vq-dim 1 \
                --groupsize 512 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/wanda-opt-125M-pr50-spp-sp-w2d1 \
                ../../models/wanda-opt-125M-pr50-spp wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 2 \
                --vq-dim 1 \
                --groupsize 512 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/wanda-opt-125M-pr60-spp-sp-w2d1 \
                ../../models/wanda-opt-125M-pr60-spp wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 2 \
                --vq-dim 1 \
                --groupsize 512 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/wanda-opt-125M-pr70-spp-sp-w2d1 \
                ../../models/wanda-opt-125M-pr70-spp wikitext2

# ##########################################################################################################

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 2 \
                --vq-dim 1 \
                --groupsize 256 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --output-dir ../../models/RIA-opt-125M-pr50-w2d1 \
                ../../models/RIA-opt-125M-pr50 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 2 \
                --vq-dim 1 \
                --groupsize 256 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --output-dir ../../models/RIA-opt-125M-pr60-w2d1 \
                ../../models/RIA-opt-125M-pr60 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 2 \
                --vq-dim 1 \
                --groupsize 256 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --output-dir ../../models/RIA-opt-125M-pr70-w2d1 \
                ../../models/RIA-opt-125M-pr70 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 2 \
                --vq-dim 1 \
                --groupsize 512 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/RIA-opt-125M-pr50-sp-w2d1 \
                ../../models/RIA-opt-125M-pr50 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 2 \
                --vq-dim 1 \
                --groupsize 512 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/RIA-opt-125M-pr60-sp-w2d1 \
                ../../models/RIA-opt-125M-pr60 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 2 \
                --vq-dim 1 \
                --groupsize 512 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/RIA-opt-125M-pr70-sp-w2d1 \
                ../../models/RIA-opt-125M-pr70 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 2 \
                --vq-dim 1 \
                --groupsize 512 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/RIA-opt-125M-pr50-spp-sp-w2d1 \
                ../../models/RIA-opt-125M-pr50-spp wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 2 \
                --vq-dim 1 \
                --groupsize 512 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/RIA-opt-125M-pr60-spp-sp-w2d1 \
                ../../models/RIA-opt-125M-pr60-spp wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 2 \
                --vq-dim 1 \
                --groupsize 512 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/RIA-opt-125M-pr70-spp-sp-w2d1 \
                ../../models/RIA-opt-125M-pr70-spp wikitext2

#########################################################################################################
#########################################################################################################
#########################################################################################################

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 3 \
                --vq-dim 1 \
                --groupsize 512 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --output-dir ../../models/opt-125M-w3d1 \
                ../../models/opt-125M wikitext2

#########################################################################################################

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 3 \
                --vq-dim 1 \
                --groupsize 512 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --output-dir ../../models/wanda-opt-125M-pr50-w3d1 \
                ../../models/wanda-opt-125M-pr50 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 3 \
                --vq-dim 1 \
                --groupsize 512 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --output-dir ../../models/wanda-opt-125M-pr60-w3d1 \
                ../../models/wanda-opt-125M-pr60 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 3 \
                --vq-dim 1 \
                --groupsize 512 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --output-dir ../../models/wanda-opt-125M-pr70-w3d1 \
                ../../models/wanda-opt-125M-pr70 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 3 \
                --vq-dim 1 \
                --groupsize 1024 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/wanda-opt-125M-pr50-sp-w3d1 \
                ../../models/wanda-opt-125M-pr50 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 3 \
                --vq-dim 1 \
                --groupsize 1024 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/wanda-opt-125M-pr60-sp-w3d1 \
                ../../models/wanda-opt-125M-pr60 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 3 \
                --vq-dim 1 \
                --groupsize 1024 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/wanda-opt-125M-pr70-sp-w3d1 \
                ../../models/wanda-opt-125M-pr70 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 3 \
                --vq-dim 1 \
                --groupsize 1024 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/wanda-opt-125M-pr50-spp-sp-w3d1 \
                ../../models/wanda-opt-125M-pr50-spp wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 3 \
                --vq-dim 1 \
                --groupsize 1024 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/wanda-opt-125M-pr60-spp-sp-w3d1 \
                ../../models/wanda-opt-125M-pr60-spp wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 3 \
                --vq-dim 1 \
                --groupsize 1024 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/wanda-opt-125M-pr70-spp-sp-w3d1 \
                ../../models/wanda-opt-125M-pr70-spp wikitext2

##########################################################################################################

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 3 \
                --vq-dim 1 \
                --groupsize 512 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --output-dir ../../models/RIA-opt-125M-pr50-w3d1 \
                ../../models/RIA-opt-125M-pr50 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 3 \
                --vq-dim 1 \
                --groupsize 512 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --output-dir ../../models/RIA-opt-125M-pr60-w3d1 \
                ../../models/RIA-opt-125M-pr60 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 3 \
                --vq-dim 1 \
                --groupsize 512 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --output-dir ../../models/RIA-opt-125M-pr70-w3d1 \
                ../../models/RIA-opt-125M-pr70 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 3 \
                --vq-dim 1 \
                --groupsize 1024 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/RIA-opt-125M-pr50-sp-w3d1 \
                ../../models/RIA-opt-125M-pr50 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 3 \
                --vq-dim 1 \
                --groupsize 1024 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/RIA-opt-125M-pr60-sp-w3d1 \
                ../../models/RIA-opt-125M-pr60 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 3 \
                --vq-dim 1 \
                --groupsize 1024 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/RIA-opt-125M-pr70-sp-w3d1 \
                ../../models/RIA-opt-125M-pr70 wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 3 \
                --vq-dim 1 \
                --groupsize 1024 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/RIA-opt-125M-pr50-spp-sp-w3d1 \
                ../../models/RIA-opt-125M-pr50-spp wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 3 \
                --vq-dim 1 \
                --groupsize 1024 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/RIA-opt-125M-pr60-spp-sp-w3d1 \
                ../../models/RIA-opt-125M-pr60-spp wikitext2

python llama.py --columns-per-group 256 \
                --use-vq \
                --kmeans-iters 100 \
                --kmeans-init-method mahalanobis \
                --hessian-weighted-lookups \
                --include-m-step \
                --wbits 3 \
                --vq-dim 1 \
                --groupsize 1024 \
                --codebook-bitwidth 8 \
                --quantize-per-codebook \
                --zero-aware \
                --output-dir ../../models/RIA-opt-125M-pr70-spp-sp-w3d1 \
                ../../models/RIA-opt-125M-pr70-spp wikitext2
