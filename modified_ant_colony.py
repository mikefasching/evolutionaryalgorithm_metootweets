import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score, silhouette_score
import progressbar
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from collections import Counter
import time
import pandas as pd
from scipy import stats
from scipy import stats as scipy_stats
from concurrent.futures import ThreadPoolExecutor
import os

class AntColonyClustering:
    def __init__(self, n_clusters, n_ants, max_iterations, alpha=1.0, beta=2.0, rho=0.01,
                 pheromone_init=1.0, Q=1.0, learning_rate=0.1, exploration_phase=0.1, 
                 noise_gamma=1, run_id=0):
        """
        Ant Colony Optimization for clustering with performance tracking.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        n_ants : int
            Number of ant agents
        max_iterations : int
            Maximum number of iterations
        alpha : float
            Pheromone importance
        beta : float
            Heuristic importance
        rho : float
            Pheromone evaporation rate
        pheromone_init : float
            Initial pheromone level
        Q : float
            Pheromone deposit amount
        learning_rate : float
            Learning rate for updating centroids
        exploration_phase : float
            Fraction of iterations for exploration
        noise_gamma : float
            Random noise factor
        run_id : int
            Run identifier for multiple runs
        """
        self.n_clusters = n_clusters
        self.n_ants = n_ants
        self.max_iterations = max_iterations
        self.exploration_phase = exploration_phase
        self.alpha = alpha
        self.beta = beta
        self.noise_gamma = noise_gamma
        self.rho = rho
        self.pheromone_bounds = None 
        self.pheromones = None
        self.temp_pheromone_updates = None
        self.pheromone_init = pheromone_init
        self.Q = Q
        self.learning_rate = learning_rate
        self.run_id = run_id
        
        # Algorithm state
        self.best_centroids = None
        self.global_best_centroids = None
        self.best_fitness = -np.inf
        self.best_assignments = None
        self.global_best_fitness = -np.inf
        self.global_best_fitness_history = []
        
        # Performance tracking
        self.iteration_times = []
        self.fitness_evaluations = 0
        self.fitness_evaluations_per_iteration = []
        self.fitness_history = []  # Average fitness per iteration
        self.best_fitness_history = []  # Best fitness per iteration
        self.silhouette_history = []  # (iteration, silhouette) pairs
        self.all_fitness_values = []  # All fitness values calculated
        self.all_fitness_timestamps = []  # Timestamp for each fitness evaluation
        self.all_silhouette_values = []  # All silhouette scores
        self.centroid_movement = []  # Movement of centroids between iterations
        self.cluster_sizes = []  # Track cluster sizes over iterations
        self.pheromone_stats = []  # Track pheromone matrix statistics
        self.ant_fitness_per_iteration = []  # Track fitness for each ant per iteration

    def normalize_vector(self, v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm

    def initialize_pheromones(self, data, initial_assignments):
        """Initialize pheromone matrix for cluster-data point influence."""
        self.pheromone_init = 1.0  # Set a uniform initial value
        self.pheromones = np.full((self.n_clusters, data.shape[0]), self.pheromone_init)
        self.pheromone_bounds = (0.1, 10.0)
        self.temp_pheromone_updates = np.full((self.n_clusters, data.shape[0]), self.pheromone_init)
        
        # Record initial pheromone stats
        self.pheromone_stats.append({
            'iteration': 0,
            'mean': np.mean(self.pheromones),
            'min': np.min(self.pheromones),
            'max': np.max(self.pheromones),
            'std': np.std(self.pheromones)
        })

    def initialize_centroids(self, data):
        """Initialize centroids using k-means++ within the data space."""
        # Check if we should load saved centroids for fair comparison with ABC
        centroids_filename = f'initial_centroids_{self.n_clusters}.npy'
        if os.path.exists(centroids_filename):
            # Load the centroids from the file
            centroids = np.load(centroids_filename)
            print("Centroids loaded from file - same initialization as ABC.-----")
            return centroids
        
        n_samples = data.shape[0]
        centroids = [data[np.random.randint(n_samples)]]
        
        for _ in range(1, self.n_clusters):
            # For each point, compute the cosine distance to the closest centroid.
            # For normalized data: cosine_distance = 1 - cosine_similarity.
            distances = np.min([
                1 - np.dot(data, c)  # since both data and c are normalized
                for c in centroids
            ], axis=0)
            
            # Avoid division by zero.
            total_distance = distances.sum() + 1e-10
            probabilities = distances / total_distance
            
            next_idx = np.random.choice(n_samples, p=probabilities)
            centroids.append(data[next_idx])
        
        # Re-normalize centroids (they should already be normalized if data is)
        centroids = np.array(centroids)
        #centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
        
        # # Save centroids for future comparison runs
        # np.save(centroids_filename, centroids)
        # print(f"Saved initial centroids to {centroids_filename}")
        
        return centroids

    def calculate_probabilities(self, data, centroids, iteration):
        """Calculate probabilities for influencing centroids with normalized pheromones and heuristics."""
        # Calculate cosine similarity between data points and centroids
        cosine_sim = cosine_similarity(data, centroids)
        cosine_sim = np.clip(cosine_sim, 0, None) 

        # Dynamic adjustment of alpha and beta
        alpha_dynamic = self.alpha * (iteration / self.max_iterations)  # Increase pheromone influence over time
        beta_dynamic = self.beta * (1 - iteration / self.max_iterations)  # Decrease heuristic influence over time
        noise_gamma_dynamic = self.noise_gamma * (1 - iteration / self.max_iterations)  # Decrease heuristic influence over time

        #Normalization
        pheromone_levels = self.pheromones.T
        pheromone_levels = np.clip(pheromone_levels, 1e-10, None)
        pheromone_levels = pheromone_levels / (pheromone_levels.sum(axis=1, keepdims=True) + 1e-10)
        cosine_sim = cosine_sim / (cosine_sim.max(axis=1, keepdims=True) + 1e-10)
        
        noise = np.random.uniform(0.9, 1.1, size=pheromone_levels.shape)

        # Combined influence with normalization and probability scaling
        combined_influence = (pheromone_levels ** alpha_dynamic) * (cosine_sim ** beta_dynamic) * (noise ** noise_gamma_dynamic)
        combined_influence /= combined_influence.sum(axis=1, keepdims=True)  # Rescale to sum to 1

        return combined_influence

    def move_centroids(self, data, probabilities, centroids, iteration):
        """Move centroids with angular changes influenced by both current centroids and pheromone levels."""
        new_centroids = np.zeros_like(centroids)
        
        # Maximum possible angular change (in degrees, converted to radians)
        max_angle_degrees = 45.0  # Similar to your EFWA's π/4 radians
        max_angle_radians = np.radians(max_angle_degrees)
        
        # Decrease max angle over time for convergence
        current_max_angle = max_angle_radians * (1 - iteration / self.max_iterations * 0.7)
        
        for cluster in range(self.n_clusters):
            # Extract weights for current cluster (derived from pheromone levels)
            cluster_weights = probabilities[:, cluster]
            weight_sum = cluster_weights.sum()
            
            if weight_sum > 1e-10:
                # Calculate weighted direction using pheromone-influenced probabilities
                weighted_direction = np.dot(cluster_weights, data) / weight_sum
                weighted_direction = self.normalize_vector(weighted_direction)
                
                # Calculate angle between current centroid and weighted direction
                cos_angle = np.clip(np.dot(centroids[cluster], weighted_direction), -1.0, 1.0)
                angle_between = np.arccos(cos_angle)
                
                # Limit the angular change based on our maximum
                move_angle = min(angle_between, current_max_angle) * self.learning_rate
                
                if move_angle > 1e-6:  # Only rotate if there's a meaningful angle
                    # Compute axis of rotation (perpendicular to both vectors)
                    ortho_component = weighted_direction - cos_angle * centroids[cluster]
                    ortho_norm = np.linalg.norm(ortho_component)
                    
                    if ortho_norm > 1e-10:
                        rotation_axis = ortho_component / ortho_norm
                        
                        # Apply rotation using the angle and axis
                        cos_move = np.cos(move_angle)
                        sin_move = np.sin(move_angle)
                        new_centroid = centroids[cluster] * cos_move + rotation_axis * sin_move
                    else:
                        # If vectors are nearly parallel or opposite, small random move
                        new_centroid = centroids[cluster]
                else:
                    new_centroid = centroids[cluster]
            else:
                new_centroid = centroids[cluster]
            
            # Normalize to stay on hypersphere
            new_centroids[cluster] = self.normalize_vector(new_centroid)
        
        return new_centroids
        
        # """Move centroids with focus on angular changes rather than just position."""
        # new_centroids = np.zeros_like(centroids)
        # min_bounds = np.min(data, axis=0)
        # max_bounds = np.max(data, axis=0)
        
        # # How much to balance angle vs position (increases focus on angle over time)
        # angle_focus = min(0.8, 0.3 + (iteration / self.max_iterations) * 0.5)
        
        # for cluster in range(self.n_clusters):
        #     # Extract weights for current cluster
        #     cluster_weights = probabilities[:, cluster]
        #     weight_sum = cluster_weights.sum()
            
        #     # Compute the weighted average centroid direction
        #     if weight_sum > 1e-10:
        #         # Standard position-based update (weighted average)
        #         position_update = np.dot(cluster_weights, data) / weight_sum
                
        #         # Get the current centroid direction (unit vector)
        #         current_norm = np.linalg.norm(centroids[cluster])
        #         if current_norm > 1e-10:
        #             current_direction = centroids[cluster] / current_norm
        #         else:
        #             current_direction = centroids[cluster]
                    
        #         # Calculate direction update - efficiently using numpy operations
        #         direction_update = np.dot(cluster_weights, data)
        #         direction_norm = np.linalg.norm(direction_update)
                
        #         if direction_norm > 1e-10:
        #             direction_update = direction_update / direction_norm
        #         else:
        #             direction_update = current_direction
                
        #         # Create orthogonal component (perpendicular to current direction)
        #         dot_product = np.dot(direction_update, current_direction)
        #         ortho_component = direction_update - dot_product * current_direction
        #         ortho_norm = np.linalg.norm(ortho_component)
                
        #         if ortho_norm > 1e-10:
        #             ortho_component = ortho_component / ortho_norm
                
        #         # Combine position and direction updates with angle_focus balance
        #         combined_update = (1 - angle_focus) * position_update + \
        #                          angle_focus * (current_direction + self.learning_rate * ortho_component * current_norm)
        #     else:
        #         combined_update = centroids[cluster]
            
        #     # Optional: Random jump for exploration (with reduced probability as iterations increase)
        #     if np.random.random() < 0.1 * (1 - iteration / self.max_iterations):
        #         combined_update = np.random.uniform(min_bounds, max_bounds)
            
        #     # Gradual movement with learning rate
        #     updated_centroid = centroids[cluster] + self.learning_rate * (combined_update - centroids[cluster])
            
        #     # Normalize for cosine similarity focus
        #     norm = np.linalg.norm(updated_centroid)
        #     if norm > 1e-10:
        #         updated_centroid = updated_centroid / norm
        #     else:
        #         updated_centroid = centroids[cluster]
            
        #     new_centroids[cluster] = updated_centroid
        
        # return new_centroids

    
    def track_centroid_angles(self, old_centroids, new_centroids):
        """Track the angular changes in centroids between iterations."""
        angles = []
        for i in range(len(old_centroids)):
            angle = self.centroid_angle(old_centroids[i], new_centroids[i])
            angles.append(angle)
        return np.mean(angles)
    # def move_centroids(self, data, probabilities, centroids, iteration):
    #     """Move centroids based on weighted sampling, ensuring movement accounts for cluster size."""
    #     new_centroids = np.zeros_like(centroids)
    #     min_bounds = np.min(data, axis=0)  # Feature-wise minimum bounds
    #     max_bounds = np.max(data, axis=0)  # Feature-wise maximum bounds
        
    #     for cluster in range(self.n_clusters):
    #         # Extract the probability (weight) for each data point for the current cluster.
    #         cluster_weights = probabilities[:, cluster]
            
    #         # Compute the weighted average: the ideal new centroid
    #         if cluster_weights.sum() > 0:
    #             temp_centroid = np.sum(data * cluster_weights[:, np.newaxis], axis=0) / cluster_weights.sum()
    #         else:
    #             temp_centroid = centroids[cluster]
            
    #         # Optionally, include a small chance for a random jump to improve exploration.
    #         if np.random.random() < 0.1:
    #             temp_centroid = np.random.uniform(min_bounds, max_bounds)
            
    #         # Update gradually: move a fraction (learning_rate) toward the weighted average.
    #         updated_centroid = centroids[cluster] + self.learning_rate * (temp_centroid - centroids[cluster])
            
    #         # If using cosine similarity, re-normalize the updated centroid to a unit vector.
    #         norm = np.linalg.norm(updated_centroid)
    #         if norm > 0:
    #             updated_centroid = updated_centroid / norm
    #         else:
    #             updated_centroid = centroids[cluster]
            
    #         new_centroids[cluster] = updated_centroid
        
    #     return new_centroids

    def evaluate_fitness(self, data, assignments):
        """Evaluate clustering fitness using the Calinski-Harabasz Score."""
        # Increment the fitness evaluation counter
        
        # Track when this fitness was calculated
        timestamp = time.time()
        
        # Calculate actual fitness score
        fitness = calinski_harabasz_score(data, assignments) if len(np.unique(assignments)) > 1 else -1
        
        # Store the fitness value and timestamp
        self.all_fitness_values.append(fitness)
        self.all_fitness_timestamps.append(timestamp)
        
        return fitness

    def assign_clusters_cosine(self, data, centroids):
        """Assigns each point to the most similar centroid using cosine similarity."""        
        similarities = cosine_similarity(data, centroids)  # Compute similarity between points and centroids
        assignments = np.argmax(similarities, axis=1)  # Assign each point to the highest similarity centroid
        return assignments
        
    def compute_silhouette(self, data, assignments):
        """Compute silhouette score for clustering evaluation"""
        try:
            score = silhouette_score(data, assignments, sample_size=min(1000, len(data)))
            return score
        except:
            return -1  # Return -1 for invalid clusterings

    def compute_centroid_cosine_similarity(self, centroids):
        """
        Computes the pairwise cosine similarity between centroids and returns statistics.
        """
        n_clusters = centroids.shape[0]
        similarities = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                # Compute cosine similarity as the dot product divided by the product of norms.
                norm_i = np.linalg.norm(centroids[i])
                norm_j = np.linalg.norm(centroids[j])
                # Add a small epsilon to avoid division by zero.
                epsilon = 1e-10
                cosine_sim = np.dot(centroids[i], centroids[j]) / ((norm_i * norm_j) + epsilon)
                similarities.append(cosine_sim)
        
        if len(similarities) == 0:
            return 0, 0, 0
            
        similarities = np.array(similarities)
        return similarities.min(), similarities.max(), similarities.mean()
        
    def calculate_centroid_movement(self, old_centroids, new_centroids):
        """Calculate angles in degrees between old and new centroid vectors."""
        # Normalize the centroids
        old_norms = np.linalg.norm(old_centroids, axis=1, keepdims=True)
        new_norms = np.linalg.norm(new_centroids, axis=1, keepdims=True)
    
        # Avoid division by zero
        old_norms[old_norms == 0] = 1
        new_norms[new_norms == 0] = 1
    
        # Compute dot product
        dot_products = np.dot(old_centroids, new_centroids.T)
    
        # Calculate cosine of the angles
        cos_theta = np.clip(dot_products / (old_norms * new_norms.T), -1.0, 1.0)
    
        # Calculate angles in degrees
        angles_rad = np.arccos(cos_theta)
        angles_deg = np.degrees(angles_rad)
    
        return np.mean(angles_deg)
        
        # """Calculate the average movement of centroids between iterations"""
        # distances = np.linalg.norm(old_centroids - new_centroids, axis=1)
        # return np.mean(distances)

    def update_pheromones(self, fitness_scores, data, iteration, matrix):
        # Sort solutions by fitness in descending order.
        fitness_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Determine how many ants are in the top and bottom groups.
        num_best = max(1, int(0.2 * self.n_ants))
        num_worst = max(1, int(0.2 * self.n_ants))
        
        # --- Process the Top 20% (Reward) ---
        # Use a dictionary to record unique candidate solutions.
        best_solutions = {}
        for rank, (fitness, assignments, _) in enumerate(fitness_scores[:num_best]):
            # Convert the assignments to a tuple to serve as a unique key.
            assignments_key = tuple(assignments)
            # If this solution hasn't been recorded yet, add it.
            if assignments_key not in best_solutions:
                # Use the rank-based weight from this ant (or choose one strategy)
                best_solutions[assignments_key] = 1 / (rank + 1)
        
        # Update pheromones once per unique candidate in the top 20%
        for assignments_key, rank_weight in best_solutions.items():
            assignments = np.array(assignments_key)
            for cluster_idx in range(self.n_clusters):
                # Find indices where this candidate solution assigns data points to the current cluster.
                cluster_points = np.where(assignments == cluster_idx)[0]
                if len(cluster_points) > 0:
                    matrix[cluster_idx, cluster_points] += self.Q * rank_weight * 1.0
        
        # --- Process the Bottom 20% (Punishment) ---
        worst_solutions = {}
        for rank, (fitness, assignments, _) in enumerate(fitness_scores[-num_worst:]):
            assignments_key = tuple(assignments)
            if assignments_key not in worst_solutions:
                worst_solutions[assignments_key] = 1 / (rank + 1)
        
        # Update pheromones once per unique candidate in the bottom 20%
        for assignments_key, rank_weight in worst_solutions.items():
            assignments = np.array(assignments_key)
            for cluster_idx in range(self.n_clusters):
                cluster_points = np.where(assignments == cluster_idx)[0]
                if len(cluster_points) > 0:
                    matrix[cluster_idx, cluster_points] -= self.Q * rank_weight * 0.5
                    matrix[cluster_idx, cluster_points] = np.maximum(matrix[cluster_idx, cluster_points], self.pheromone_bounds[0])
        
        # --- Evaporation ---
        evaporation_rate = 0.01 * (1 - iteration / self.max_iterations)
        matrix *= (1 - evaporation_rate)
        
        # --- Clipping ---
        matrix = np.clip(matrix, *self.pheromone_bounds)
        
        # --- Track pheromone statistics ---
        self.pheromone_stats.append({
            'iteration': iteration + 1,
            'mean': np.mean(matrix),
            'min': np.min(matrix),
            'max': np.max(matrix),
            'std': np.std(matrix)
        })

    def run(self, data):
        """Run the Ant Colony Optimization clustering algorithm with performance tracking."""
        # Initialize tracking variables
        overall_start_time = time.time()
        iteration = -1
        
        # Initialize centroids and assignments
        centroids = self.initialize_centroids(data)
        initial_assignments = self.assign_clusters_cosine(data, centroids)
        
        # Initialize pheromones based on initial assignments
        self.initialize_pheromones(data, initial_assignments)
        
        # Evaluate initial fitness
        initial_fitness = self.evaluate_fitness(data, initial_assignments)
        self.best_fitness = initial_fitness  
        self.global_best_fitness = initial_fitness
        self.best_fitness_history.append(initial_fitness)
        self.fitness_evaluations_per_iteration.append(1)
        self.global_best_fitness_history.append(self.global_best_fitness)
        
        # Initial silhouette score
        initial_silhouette = self.compute_silhouette(data, initial_assignments)
        self.silhouette_history.append((0, initial_silhouette))
        self.all_silhouette_values.append(initial_silhouette)
        
        # Save initial cluster sizes
        cluster_counts = np.bincount(initial_assignments, minlength=self.n_clusters)
        self.cluster_sizes.append(dict(zip(range(self.n_clusters), cluster_counts)))
        
        # Print initial diagnostic information
        print(f"ACO Initialization - Fitness: {initial_fitness:.2f}, Silhouette: {initial_silhouette:.4f}")
        
        # Number of iterations for exploration phase
        exploration_iterations = int(self.exploration_phase * self.max_iterations)

        
        
        # Define ant processing function for parallel execution
        def process_ant(ant_idx):
            # Calculate probabilities based on pheromones and heuristics
            probabilities = self.calculate_probabilities(data, centroids, iteration)
            
            # Move centroids based on probabilities
            moved_centroids = self.move_centroids(data, probabilities, centroids, iteration)
            
            # Assign data points to clusters based on new centroids
            assignments = self.assign_clusters_cosine(data, moved_centroids)
            
            # Evaluate the clustering quality
            fitness = self.evaluate_fitness(data, assignments)
            
            return fitness, assignments, moved_centroids
        
        # Progress bar for iterations
        progress_bar = progressbar.ProgressBar(max_value=self.max_iterations,
                                              widgets=['ACO Progress: ', progressbar.Percentage(), ' ',
                                                       progressbar.Bar(marker='█'), ' ',
                                                       progressbar.ETA(), ' ',
                                                       progressbar.DynamicMessage('best_fitness')])
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            iteration_start_time = time.time()

            evals_before_iteration = self.fitness_evaluations
            
            # Store previous best centroids for movement tracking
            prev_centroids = centroids.copy() if iteration > 0 else None
            
            # Collect fitness scores for all ants
            fitness_scores = []
            
            # Process all ants in parallel
            with ThreadPoolExecutor(max_workers=min(self.n_ants, 8)) as executor:
                ant_results = list(executor.map(process_ant, range(self.n_ants)))
            
            # Process results
            fitness_scores = ant_results
            iteration_fitness_values = [result[0] for result in ant_results]
            
            # Store all fitness values for this iteration
            self.ant_fitness_per_iteration.append(iteration_fitness_values)
            
            # Sort ants by fitness
            fitness_scores.sort(reverse=True, key=lambda x: x[0])
            
            # Extract best solution for this iteration
            best_fitness, best_assignments, best_centroids = fitness_scores[0]
            
            # Track cluster sizes for best solution
            unique_clusters, cluster_counts = np.unique(best_assignments, return_counts=True)
            self.cluster_sizes.append(dict(zip(unique_clusters, cluster_counts)))
            
            # Update pheromones based on fitness scores
            if iteration < exploration_iterations:
                self.update_pheromones(fitness_scores, data, iteration, self.temp_pheromone_updates)
            else:
                # Transition from exploration to exploitation
                if iteration == exploration_iterations:
                    self.pheromones = self.temp_pheromone_updates.copy()
                self.update_pheromones(fitness_scores, data, iteration, self.pheromones)
            
            # Update global best if improvement found
            if best_fitness > self.global_best_fitness:
                self.global_best_fitness = best_fitness
                self.best_centroids = best_centroids.copy()
                self.best_assignments = best_assignments.copy()
            
            # Calculate centroid movement if we have previous centroids
            if prev_centroids is not None:
                angle_change = self.calculate_centroid_movement(prev_centroids, best_centroids)
                # You could add a list to store these values
                self.centroid_movement.append(angle_change)
            
            # if prev_centroids is not None:
            #     movement = self.calculate_centroid_movement(prev_centroids, best_centroids)
            #     self.centroid_movement.append(movement)
            
            # Update fitness history
            self.fitness_history.append(np.mean(iteration_fitness_values))
            self.best_fitness_history.append(best_fitness)
            self.global_best_fitness_history.append(self.global_best_fitness)
            
            # Calculate silhouette score periodically to reduce computational overhead
            if iteration % 5 == 0 or iteration == self.max_iterations - 1:
                silhouette = self.compute_silhouette(data, best_assignments)
                self.silhouette_history.append((iteration, silhouette))
                self.all_silhouette_values.append(silhouette)
            
            # Set centroids for next iteration
            centroids = self.best_centroids.copy()
            
            # Record iteration time
            iteration_end_time = time.time()
            iteration_time = iteration_end_time - iteration_start_time
            self.iteration_times.append(iteration_time)
            
            # Store evaluations for this iteration
            self.fitness_evaluations += self.n_ants
            evals_this_iteration = self.fitness_evaluations - evals_before_iteration
            self.fitness_evaluations_per_iteration.append(evals_this_iteration)
            
            # Update progress bar
            progress_bar.update(iteration, best_fitness=self.global_best_fitness)
            
        # Assign data points using the final best centroids
        final_assignments = self.assign_clusters_cosine(data, self.best_centroids)
        
        # Final silhouette score
        final_silhouette = self.compute_silhouette(data, final_assignments)
        print(f"\nFinal Results - Best Fitness: {self.global_best_fitness:.2f}, Final Silhouette: {final_silhouette:.4f}")
        print(f"Total fitness evaluations: {self.fitness_evaluations}")
        
        return final_assignments, self.best_centroids

    def visualize_performance(self, show=True, save_path=None):
        """Visualize algorithm performance metrics in a comprehensive way"""
        plt.figure(figsize=(18, 12))
        
        # Plot 1: Fitness History
        plt.subplot(2, 3, 1)
        plt.plot(self.global_best_fitness_history,'g-',label='Global Best Fitness')
        plt.plot(self.best_fitness_history, 'r--', label='Best Fitness per Iteration')
        plt.plot(self.fitness_history, 'b--', label='Average Fitness')
        plt.title('Fitness History')
        plt.xlabel('Iteration')
        plt.ylabel('Calinski-Harabasz Score')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Iteration Times
        plt.subplot(2, 3, 2)
        plt.plot(self.iteration_times, 'g-')
        plt.title('Iteration Execution Times')
        plt.xlabel('Iteration')
        plt.ylabel('Time (seconds)')
        plt.grid(True)
        
        # Plot 3: Pheromone Statistics
        plt.subplot(2, 3, 3)
        if len(self.pheromone_stats) > 0:
            iterations = [stat['iteration'] for stat in self.pheromone_stats]
            means = [stat['mean'] for stat in self.pheromone_stats]
            mins = [stat['min'] for stat in self.pheromone_stats]
            maxs = [stat['max'] for stat in self.pheromone_stats]
            plt.plot(iterations, means, 'b-', label='Mean')
            plt.plot(iterations, mins, 'g--', label='Min')
            plt.plot(iterations, maxs, 'r--', label='Max')
            plt.title('Pheromone Statistics')
            plt.xlabel('Iteration')
            plt.ylabel('Pheromone Values')
            plt.legend()
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'No pheromone data', 
                    horizontalalignment='center',
                    verticalalignment='center')
        
        # Plot 4: Centroid Movement
        plt.subplot(2, 3, 4)
        if len(self.centroid_movement) > 0:
            plt.plot(self.centroid_movement, 'm-')
            plt.title('Average Centroid Movement')
            plt.xlabel('Iteration')
            plt.ylabel('Average Movement')
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'No centroid movement data', 
                    horizontalalignment='center',
                    verticalalignment='center')
        
        # Plot 5: Silhouette Scores
        plt.subplot(2, 3, 5)
        if len(self.silhouette_history) > 0:
            iterations, scores = zip(*self.silhouette_history)
            plt.plot(iterations, scores, 'c-o')
            plt.title('Silhouette Scores')
            plt.xlabel('Iteration')
            plt.ylabel('Silhouette Score')
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'No silhouette data', 
                    horizontalalignment='center',
                    verticalalignment='center')
        
        # Plot 6: Fitness Evaluations Distribution
        plt.subplot(2, 3, 6)
        if len(self.all_fitness_values) > 0:
            plt.hist(self.all_fitness_values, bins=20, color='orange', alpha=0.7)
            plt.title('Distribution of Fitness Values')
            plt.xlabel('Fitness Score')
            plt.ylabel('Frequency')
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'No fitness distribution data', 
                    horizontalalignment='center',
                    verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return self

    def plot_pheromone_distribution(self, iteration):
        """Visualize the pheromone distribution."""
        plt.figure(figsize=(12, 8))
        plt.imshow(self.pheromones, aspect='auto', cmap='viridis')
        plt.colorbar(label='Pheromone Intensity')
        plt.title(f'Pheromone Distribution (Iteration {iteration + 1})')
        plt.xlabel('Data Points')
        plt.ylabel('Clusters')
        plt.show()

    def centroid_angle(self, old_centroid, new_centroid):
        """
        Computes the angle (in degrees) between two centroid vectors.
        """
        # Safely compute dot product / norms to avoid numerical issues
        norm_old = np.linalg.norm(old_centroid) + 1e-10
        norm_new = np.linalg.norm(new_centroid) + 1e-10
        dot_val = np.dot(old_centroid, new_centroid)
        
        # Cosine of angle
        cos_theta = dot_val / (norm_old * norm_new)
        # Clip to [-1, 1] to avoid floating-point rounding errors
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        # Convert radians -> degrees
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)
        return angle_deg
        
    def get_performance_data(self):
        """Return all performance data for statistical analysis"""
        return {
            'algorithm': 'ACO',
            'run_id': self.run_id,
            'iteration_times': self.iteration_times,
            'fitness_evaluations': self.fitness_evaluations,
            'fitness_history': self.fitness_history,
            'best_fitness_history': self.best_fitness_history,
            'global_best_fitness_history': self.global_best_fitness_history, 
            'all_fitness_values': self.all_fitness_values,
            'all_fitness_timestamps': self.all_fitness_timestamps,
            'centroid_movement': self.centroid_movement,
            'silhouette_scores': self.silhouette_history,
            'all_silhouette_values': self.all_silhouette_values,
            'final_fitness': self.global_best_fitness,
            'pheromone_stats': self.pheromone_stats,
            'fitness_evaluations_per_iteration': self.fitness_evaluations_per_iteration, 
            'ant_fitness_per_iteration': self.ant_fitness_per_iteration
        }

def run_aco_with_stats(data, n_clusters=15, n_ants=15, max_iterations=50, 
                       alpha=1.0, beta=2.0, rho=0.01, Q=1.0, learning_rate=0.1,
                       exploration_phase=0.1, noise_gamma=1, num_runs=5):
    """
    Run ACO multiple times to collect statistics
    
    Parameters:
    -----------
    data : array-like
        The data to cluster
    n_clusters : int
        Number of clusters
    n_ants : int
        Number of ant agents
    max_iterations : int
        Maximum number of iterations
    alpha : float
        Pheromone importance
    beta : float
        Heuristic importance
    rho : float
        Pheromone evaporation rate
    Q : float
        Pheromone deposit amount
    learning_rate : float
        Learning rate for updating centroids
    exploration_phase : float
        Fraction of iterations for exploration
    noise_gamma : float
        Random noise factor
    num_runs : int
        Number of runs for statistical analysis
    
    Returns:
    --------
    DataFrame with statistics and performance metrics
    """
    results = []
    all_run_data = []
    
    for run in range(num_runs):
        print(f"ACO Run {run+1}/{num_runs}")
        aco = AntColonyClustering(
            n_clusters=n_clusters,
            n_ants=n_ants,
            max_iterations=max_iterations,
            alpha=alpha,
            beta=beta,
            rho=rho,
            Q=Q,
            learning_rate=learning_rate,
            exploration_phase=exploration_phase,
            noise_gamma=noise_gamma,
            run_id=run
        )
        
        final_assignments, best_centroids = aco.run(data)
        
        # Calculate final silhouette score
        final_silhouette = aco.compute_silhouette(data, final_assignments)
        
        # Store results for this run
        results.append({
            'run_id': run,
            'best_fitness': aco.global_best_fitness,
            'total_time': sum(aco.iteration_times),
            'avg_iteration_time': np.mean(aco.iteration_times),
            'fitness_evaluations': aco.fitness_evaluations,
            'final_silhouette': final_silhouette
        })
        
        # Store detailed performance data
        all_run_data.append(aco.get_performance_data())
        
        # Visualize this run
        aco.visualize_performance(show=False, save_path=f"aco_performance_run_{run}.png")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate statistics
    stats = {
        'algorithm': 'ACO',
        'mean_fitness': results_df['best_fitness'].mean(),
        'std_fitness': results_df['best_fitness'].std(),
        'min_fitness': results_df['best_fitness'].min(),
        'max_fitness': results_df['best_fitness'].max(),
        'mean_time': results_df['total_time'].mean(),
        'mean_evaluations': results_df['fitness_evaluations'].mean(),
        'mean_silhouette': results_df['final_silhouette'].mean()
    }
    
    # Confidence intervals (95%)
    t_value = scipy_stats.t.ppf(0.975, num_runs-1)  # Two-tailed 95% CI
    stats['fitness_ci'] = t_value * (results_df['best_fitness'].std() / np.sqrt(num_runs))
    stats['time_ci'] = t_value * (results_df['total_time'].std() / np.sqrt(num_runs))
    
    return pd.Series(stats), results_df, all_run_data

def scale_data(data, feature_range=(0.1, 1.0)):
    """Scale data to specified range"""
    scaler = MinMaxScaler(feature_range=feature_range)
    return scaler.fit_transform(data)

