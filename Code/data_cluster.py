import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist

def select_cluster_heads(distractors, n_clusters=3, model_name='all-MiniLM-L6-v2', distance_threshold=1.2):
    """
    Select representative items from clusters of distractors using sentence transformers
    and agglomerative clustering with a distance threshold.
    """
    # Initialize the sentence transformer model
    model = SentenceTransformer(model_name)
    
    # Generate embeddings
    embeddings = model.encode(distractors, convert_to_tensor=True)
    embeddings = embeddings.cpu().numpy()  # Convert to numpy array
    
    # Perform clustering with distance threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,  # Set to None when using distance_threshold
        distance_threshold=distance_threshold,
        linkage='ward',
        compute_distances=True  # Enable distance computation
    )
    cluster_labels = clustering.fit_predict(embeddings)
    
    # Get number of clusters formed
    n_clusters_formed = len(np.unique(cluster_labels))
    #print(f"Number of clusters formed with threshold {distance_threshold}: {n_clusters_formed}")
    
    # Group distractors by cluster
    clusters = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        clusters[label].append((distractors[idx], idx))
    
    # Select cluster heads
    selected_distractors = []
    cluster_info = {}
    
    for cluster_id, cluster_items in clusters.items():
        cluster_indices = [idx for _, idx in cluster_items]
        cluster_embeddings = embeddings[cluster_indices]
        
        # Calculate centroid
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Calculate distances to centroid
        distances = cdist([centroid], cluster_embeddings, metric='euclidean')[0]
        
        # Select item closest to centroid
        closest_idx = np.argmin(distances)
        cluster_head = cluster_items[closest_idx][0]
        
        selected_distractors.append(cluster_head)
        
        cluster_info[cluster_id] = {
            'head': cluster_head,
            'members': [item for item, _ in cluster_items],
            'size': len(cluster_items)
        }
    
    return {
        'selected_heads': selected_distractors[:3],  # Limit to top 3 clusters
        'clusters': cluster_info,
        'embeddings': embeddings,
        'labels': cluster_labels,
        'n_clusters_formed': n_clusters_formed
    }

def process_json_file(input_path, output_path):
    """
    Process JSON file containing questions and their distractors.
    Apply clustering to select representative distractors.
    """
    try:
        # Read input JSON file
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process each item in the data
        for item in data:
            if 'generated_distractors' in item and len(item['generated_distractors']) > 0:
                # Apply clustering to select distractors
                cluster_result = select_cluster_heads(
                    item['generated_distractors'],
                    distance_threshold=1.2  # Set threshold to 1.2
                )
                # Update the generated_distractors with selected heads
                item['generated_distractors'] = cluster_result['selected_heads']
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save processed data to new JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"Successfully processed file and saved to: {output_path}")
            
    except FileNotFoundError:
        print(f"Error: Could not find the input file at: {input_path}")
        print("Please check if the file path is correct and the file exists.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Get the current working directory
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

# Use absolute paths
input_file = os.path.join(current_dir, "sciq_candidates", "t5_base_ft_cg_004_10.json")
output_file = os.path.join(current_dir, "sciq_candidates", "t5_base_ft_cg_004_10_clustered.json")

print(f"Input file path: {input_file}")
print(f"Output file path: {output_file}")

process_json_file(input_file, output_file)