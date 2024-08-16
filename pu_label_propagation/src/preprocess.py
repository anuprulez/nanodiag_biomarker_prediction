import sys

if len(sys.argv) < 5:
    sys.exit('Sintassi: adaptBiogridWithScoredSeedGene Biogrid seedGene outLink outGene')

biogrid = sys.argv[1]
seed_gene = sys.argv[2]
out_link = sys.argv[3]
out_gene = sys.argv[4]

gene = {}
ngene = 0
mat = {}

with open(biogrid, 'r') as flink, open(out_link, 'w') as fout_link:
    for line in flink:
        node1, node2 = line.strip().split()
        if node1 != node2:
            if node1 not in gene:
                gene[node1] = ngene
                ngene += 1
            if node2 not in gene:
                gene[node2] = ngene
                ngene += 1
            
            id1 = gene[node1]
            id2 = gene[node2]
            
            if (id1, id2) not in mat and (id2, id1) not in mat:
                mat[(id1, id2)] = mat[(id2, id1)] = 1
                fout_link.write(f"{id1} {id2}\n")

score_seed_gene = {}
max_score = 0
nseedgene = 0
notfoundseedgene = 0

with open(seed_gene, 'r') as fin:
    for line in fin:
        name_gene, score = line.strip().split()
        score = float(score)
        if name_gene in gene:
            score_seed_gene[name_gene] = score
            nseedgene += 1
        else:
            print(f"Error, not found seed gene {name_gene}")
            notfoundseedgene += 1
        if score > max_score:
            max_score = score

print(f"{notfoundseedgene} seed genes not found")
print(f"{nseedgene} seed genes present")
print(f"Maximum score {max_score}")

with open(out_gene, 'w') as fout_gene:
    for name_gene, gene_id in gene.items():
        if name_gene in score_seed_gene:
            adapt_score = max_score - score_seed_gene[name_gene]
            fout_gene.write(f"{gene_id} {name_gene} {score_seed_gene[name_gene]}\n")
        else:
            fout_gene.write(f"{gene_id} {name_gene} 0.0\n")