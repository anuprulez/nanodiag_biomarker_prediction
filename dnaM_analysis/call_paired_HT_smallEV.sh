methylpy paired-end-pipeline \
	--read1-files HT_small_EV/HT_smallEV_N_1.fastqsanger.gz \
	--read2-files HT_small_EV/HT_smallEV_N_2.fastqsanger.gz \
	--sample HT_smallEV_new \
	--forward-ref ref/hg38_f \
	--reverse-ref ref/hg38_r \
	--ref-fasta ref/Homo_sapiens_assembly38.fasta \
	--num-procs 12 \
	--remove-clonal True \
	--path-to-picard="/home/anup/anaconda3/envs/dnaM/bin/picard/"