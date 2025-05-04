## Cristiano Nicolau - 108536

import pandas as pd
import ast
import argparse
import json
from collections import defaultdict, Counter

def parse_genres(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

def load_clusters_from_bfr_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    clusters = []
    for cluster_id_str, track_ids in data["DS"].items():
        cluster_id = int(cluster_id_str)
        for track_id in track_ids:
            clusters.append({"track_id": track_id, "cluster_id": cluster_id})
    
    return pd.DataFrame(clusters)

def main(cluster_assignments_path, tracks_csv_path, genres_csv_path=None):
    clusters_df = load_clusters_from_bfr_json(cluster_assignments_path)

    tracks = pd.read_csv(tracks_csv_path, header=[0, 1], index_col=0)

    if ('track', 'genres') not in tracks.columns:
        raise KeyError("Coluna ('track', 'genres') não encontrada no tracks.csv. Verifique o cabeçalho.")

    tracks.columns = ['_'.join(col).strip() for col in tracks.columns.values]  # Remove MultiIndex
    tracks = tracks.reset_index()  # Garante que 'track_id' é coluna

    if 'track_genres' not in tracks.columns:
        raise KeyError("Coluna 'track_genres' não encontrada. Verifique se a coluna correta foi formada após o flatten.")

    tracks['genres'] = tracks['track_genres'].apply(parse_genres)
    tracks_subset = tracks[['track_id', 'genres']]

    merged = clusters_df.merge(tracks_subset, on='track_id', how='left')

    genre_counts_per_cluster = defaultdict(list)
    for _, row in merged.iterrows():
        cid = row['cluster_id']
        for gid in row['genres']:
            genre_counts_per_cluster[cid].append(gid)

    genre_id_to_name = {}
    if genres_csv_path:
        genres_df = pd.read_csv(genres_csv_path)
        genre_id_to_name = dict(zip(genres_df['genre_id'], genres_df['title']))

    for cid in sorted(genre_counts_per_cluster):
        print(f"\nTop genres by cluster {cid}:")
        counter = Counter(genre_counts_per_cluster[cid]).most_common(10)
        for gid, count in counter:
            genre_name = genre_id_to_name.get(gid, f"ID {gid}")
            print(f"  {genre_name}: {count} musics")

    with open("_output/cluster_genres.txt", "w") as f:
        for cid in sorted(genre_counts_per_cluster):
            f.write(f"\nTop genres by cluster {cid}:\n")
            counter = Counter(genre_counts_per_cluster[cid]).most_common(10)
            for gid, count in counter:
                genre_name = genre_id_to_name.get(gid, f"ID {gid}")
                f.write(f"  {genre_name}: {count} musics\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genres by cluster")
    parser.add_argument("-c", "--cluster_assignments", default="_output/bfr_result.json", help="CSV com track_id e cluster_id")
    parser.add_argument("-t", "--tracks_csv", default="_input/fma_metadata/tracks.csv"  ,help="tracks.csv (metadados)")
    parser.add_argument("-g", "--genres_csv", default="_input/fma_metadata/genres.csv" ,help="genres.csv (opcional, para nomes dos genres)")
    args = parser.parse_args()

    main(args.cluster_assignments, args.tracks_csv, args.genres_csv)
