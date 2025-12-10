methylpy paired-end-pipeline \
	--read1-files Tspan8_largeEV/Tsapn8_largeEV_1.fastqsanger.gz \
	--read2-files Tspan8_largeEV/Tsapn8_largeEV_2.fastqsanger.gz \
	--sample Tsapn8_largeEV \
	--forward-ref ref/hg38_f \
	--reverse-ref ref/hg38_r \
	--ref-fasta ref/Homo_sapiens_assembly38.fasta \
	--num-procs 12 \
	--remove-clonal False \
	--path-to-picard="/home/anup/anaconda3/envs/dnaM/bin/picard/"