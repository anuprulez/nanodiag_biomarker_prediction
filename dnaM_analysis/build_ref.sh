methylpy build-reference \
	--input-files ref/Homo_sapiens_assembly38.fasta \
	--output-prefix ref/hg38 \
    --path-to-aligner /home/anup/anaconda3/envs/dnaM/bin/bowtie2 \
    --num-procs 12 \
    --aligner bowtie2