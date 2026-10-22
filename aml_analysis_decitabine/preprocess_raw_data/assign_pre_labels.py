import argparse
import sys
import subprocess
import pandas as pd

# if len(sys.argv) < 5:
#    sys.exit('Sintassi: adaptBiogridWithScoredSeedGene Biogrid seedGene outLink outGene')

# biogrid = sys.argv[1]
# seed_gene = sys.argv[2]
# out_link = sys.argv[3]
# out_gene = sys.argv[4]


def create_network_gene_ids(ppi_path, links_path):
    gene = {}
    ngene = 0
    mat = {}
    with open(ppi_path, "r") as flink, open(links_path, "w") as fout_link:
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
    return gene


def mark_seed_genes(seed_genes_path, genes_path, gene):
    score_seed_gene = {}
    max_score = 0
    nseedgene = 0
    notfoundseedgene = 0

    with open(seed_genes_path, "r") as fin:
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

    # out_gene = "data/output/out_gene"
    with open(genes_path, "w") as fout_gene:
        for name_gene, gene_id in gene.items():
            if name_gene in score_seed_gene:
                adapt_score = max_score - score_seed_gene[name_gene]
                fout_gene.write(f"{gene_id} {name_gene} {score_seed_gene[name_gene]}\n")
            else:
                fout_gene.write(f"{gene_id} {name_gene} 0.0\n")


def calculate_features(links_data_path, genes_data_path, nedbit_path):
    # nedbit_features_calculator out_links out_genes nedbit_features
    # nedbit-features-calculator

    print("calculating nedbit features ...")
    print(links_data_path, genes_data_path, nedbit_path)
    result = subprocess.run(
        ["nedbit-features-calculator", links_data_path, genes_data_path, nedbit_path],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print("Error:", result.stderr)


def assign_initial_labels(
    nedbit_path, header, output_gene_ranking_path, q1=0.05, q2=0.2
):
    # apu_label_propagation nedbit_features HEADER_PRESENCE output_gene_ranking 0.05 0.2
    # apu-label-propagation
    print("propagating labels ...")
    result = subprocess.run(
        [
            "apu-label-propagation",
            nedbit_path,
            header,
            output_gene_ranking_path,
            q1,
            q2,
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print("Error:", result.stderr)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "-ppi", "--probe_gene_network", required=True, help="path probe_gene_network"
    )
    arg_parser.add_argument(
        "-sg",
        "--seed_gene_scores",
        required=True,
        help="path list of seed genes and association scores",
    )
    arg_parser.add_argument(
        "-ol", "--output_links", required=True, help="path to list of gene-gene links"
    )
    arg_parser.add_argument(
        "-og", "--output_genes", required=True, help="path to list of genes"
    )
    arg_parser.add_argument(
        "-nf", "--nedbit_features", required=True, help="path to NeDBIT features"
    )
    arg_parser.add_argument(
        "-nh", "--nedbit_header", required=True, help="data header 0/1"
    )
    arg_parser.add_argument(
        "-gr",
        "--output_gene_ranking",
        required=True,
        help="path to output gene ranking",
    )
    arg_parser.add_argument(
        "-qt1", "--quantile_1", required=True, help="quantile threshold 1"
    )
    arg_parser.add_argument(
        "-qt2", "--quantile_2", required=True, help="quantile threshold 2"
    )

    args = vars(arg_parser.parse_args())

    genes = create_network_gene_ids(args["probe_gene_network"], args["output_links"])
    mark_seed_genes(args["seed_gene_scores"], args["output_genes"], genes)

    calculate_features(
        args["output_links"], args["output_genes"], args["nedbit_features"]
    )

    assign_initial_labels(
        args["nedbit_features"],
        args["nedbit_header"],
        args["output_gene_ranking"],
        args["quantile_1"],
        args["quantile_2"],
    )
