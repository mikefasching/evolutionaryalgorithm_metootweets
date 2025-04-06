import numpy as np
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import time
import uuid
from joblib import Parallel, delayed
import multiprocessing
import os

class EnhancedFireworksClustering:
    def __init__(self, n_clusters=5, max_iter=50, n_sparks=30, n_guassian=5,
                 explosion_amp=0.8, a_min=0.05, a_max=0.8, gaussian_amp=1.0, 
                 s_min=2, s_max=40, run_id=None, n_jobs=-1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_sparks = n_sparks
        self.n_guassian = n_guassian
        self.explosion_amp = explosion_amp
        self.a_min = a_min
        self.a_max = a_max
        self.gaussian_amp = gaussian_amp
        self.s_min = s_min  # Minimum sparks per centroid
        self.s_max = s_max
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self.run_id = str(run_id) if run_id is not None else str(uuid.uuid4())[:8]
        self.normalized_data = 0
        self.all_angular_changes = []
        
        # Performance tracking metrics
        self.best_fitness_history = []
        self.fitness_history = []
        self.iteration_times = []
        self.fitness_evaluations = 0
        self.fitness_evaluations_per_iteration = []
        self.global_best_fitness_history = []
        self.centroid_movement = []
        self.cluster_sizes = []
        self.all_fitness_values = []
        self.all_fitness_timestamps = []
        self.firework_fitness_per_iteration = []
        self.silhouette_history = []
        
        self.best_fitness = float('-inf')
        self.global_best_fitness = float('-inf')
        self.best_centroids = None
        self.best_assignments = None
        
    def normalize_vector(self, v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm
        
    def initialize_centroids(self, data):
        centroids_filename = f'initial_centroids_{self.n_clusters}.npy'
        if os.path.exists(centroids_filename):
            # Load the centroids from the file
            centroids = np.load(centroids_filename)
            print("Centroids loaded from file - same initialization as ABC.-----")
            return np.apply_along_axis(self.normalize_vector, 1, centroids)
        
        indices = np.random.choice(data.shape[0], self.n_clusters, replace=False)
        centroids = data[indices].copy()
        return np.apply_along_axis(self.normalize_vector, 1, centroids)
            
    def assign_clusters(self, data, centroids):
        similarity = cosine_similarity(data, centroids)
        labels = np.argmax(similarity, axis=1)
        return labels
    
    def calculate_fitness(self, data, labels):
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return -1
            
        cluster_counts = np.bincount(labels, minlength=self.n_clusters)
        if np.any(cluster_counts == 0):
            return -1
                
        try:
            score = calinski_harabasz_score(data, labels)
            return score
        except Exception as e:
            return -1
    
    def compute_silhouette(self, data, labels):
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return -1
        
        try:
            sil_score = silhouette_score(data, labels, metric='cosine')
            return sil_score
        except Exception as e:
            return -1
    
    def calculate_explosion_params(self, centroid_fitness_scores):
        # 1. Calculate amplitudes using your existing method
        if len(centroid_fitness_scores) <= 1:
            amplitudes = np.full(len(centroid_fitness_scores), self.a_max)
            # For a single centroid, use the default number of sparks
            return amplitudes, np.full(len(centroid_fitness_scores), self.n_sparks, dtype=int)
        
        min_quality = min(centroid_fitness_scores)
        max_quality = max(centroid_fitness_scores)
        
        if max_quality == min_quality:
            amplitudes = np.full(len(centroid_fitness_scores), (self.a_max + self.a_min) / 2)
            # Equal quality means equal spark counts too
            return amplitudes, np.full(len(centroid_fitness_scores), self.n_sparks, dtype=int)
        
        # Normalize and invert quality scores (lower quality gets higher amplitude)
        normalized_quality = (np.array(centroid_fitness_scores) - min_quality) / (max_quality - min_quality)
        inverted_quality = 1.0 - normalized_quality
        
        # Scale between min and max amplitude
        amplitudes = self.a_min + inverted_quality * (self.a_max - self.a_min)
        
        # 2. Calculate spark counts based on the original FWA formula (equation 2.1)
        # For FWA, spark count is inversely proportional to fitness
        # (better solutions get fewer sparks to focus exploration on worse solutions)
        
        # Total spark budget (parameter from constructor)
        total_sparks = self.n_sparks * len(centroid_fitness_scores)
        
        # Calculate spark counts
        spark_counts = []
        for i, fitness in enumerate(centroid_fitness_scores):
            # Use the inverted quality directly (already normalized and inverted)
            relative_quality = inverted_quality[i]
            
            # Calculate spark count (high inverted_quality = more sparks)
            s_i = total_sparks * (relative_quality / np.sum(inverted_quality))
            spark_counts.append(int(round(s_i)))
        
        # Apply bounds as per equation 2.2
        s_min = 2  # Minimum sparks per firework
        s_max = self.n_sparks * 2  # Maximum sparks per firework
        
        for i in range(len(spark_counts)):
            if spark_counts[i] < s_min:
                spark_counts[i] = s_min
            elif spark_counts[i] > s_max:
                spark_counts[i] = s_max
        
        print(f'Quality of Centroids: {normalized_quality} - Amplitudes: {amplitudes} - Spark counts: {spark_counts}')
        
        return amplitudes, spark_counts
    
    def generate_sparks(self, centroid, amplitude, num_sparks, data_dim):
        # Create random direction vectors
        random_directions = np.random.rand(num_sparks, data_dim) - 0.5
        # Normalize to create unit vectors perpendicular to the centroid
        random_directions = np.apply_along_axis(self.normalize_vector, 1, random_directions)
        
        # Make the directions perpendicular to centroid by removing the projection onto centroid
        # This ensures movements are tangential to the hypersphere surface
        for i in range(num_sparks):
            # Calculate dot product
            proj = np.dot(random_directions[i], centroid)
            # Subtract the projection to make it orthogonal to centroid
            random_directions[i] = random_directions[i] - proj * centroid
            # Normalize again to ensure unit length
            random_directions[i] = self.normalize_vector(random_directions[i])
        
        # Convert amplitude to an angular displacement (in radians)
        # Scale amplitude to control maximum angular displacement (e.g., π/2 radians = 90 degrees)
        max_angle = np.pi/2  # Maximum angle is 90 degrees
        angular_amplitude = amplitude * max_angle * self.explosion_amp
        
        # Create rotated vectors by applying Rodrigues' rotation formula
        # v_rot = v * cos(θ) + (k × v) * sin(θ) + k * (k·v) * (1 - cos(θ))
        # where k is the rotation axis, v is the vector to rotate, and θ is the angle
        sparks = np.zeros((num_sparks, data_dim))
        for i in range(num_sparks):
            # Rotation axis is the cross product of centroid and random direction
            # In high dimensions, we can use the random direction itself as rotation axis
            axis = random_directions[i]
            
            # Apply rotation formula
            cos_theta = np.cos(angular_amplitude)
            sin_theta = np.sin(angular_amplitude)
            sparks[i] = centroid * cos_theta + axis * sin_theta
        
        # Normalize the resulting vectors
        sparks = np.apply_along_axis(self.normalize_vector, 1, sparks)
        
        # No need for mapping rule since we're directly generating points on the hypersphere
        return sparks
        
    def generate_gaussian_sparks(self, centroid, amplitude, num_sparks, data_dim):
        # Create random direction vectors
        random_directions = np.random.normal(0, 1, (num_sparks, data_dim))
        # Normalize to create unit vectors
        random_directions = np.apply_along_axis(self.normalize_vector, 1, random_directions)
        
        # Make the directions perpendicular to centroid
        for i in range(num_sparks):
            # Calculate dot product
            proj = np.dot(random_directions[i], centroid)
            # Subtract the projection to make it orthogonal to centroid
            random_directions[i] = random_directions[i] - proj * centroid
            # Normalize again to ensure unit length
            random_directions[i] = self.normalize_vector(random_directions[i])
        
        # Convert amplitude to a base angular displacement
        max_angle = np.pi/2  # Maximum angle is 90 degrees
        base_angular_amplitude = amplitude * max_angle
        
        # Generate Gaussian distributed angles around the base amplitude
        # Using amplitude/3 as standard deviation to keep most angles reasonable
        gaussian_angles = np.random.normal(0, base_angular_amplitude/3, (num_sparks, 1))
        
        # Apply self.gaussian_amp to scale the angles
        angular_amplitudes = np.abs(gaussian_angles) * self.gaussian_amp
        
        # Create rotated vectors using rotation formula
        sparks = np.zeros((num_sparks, data_dim))
        for i in range(num_sparks):
            # Use perpendicular direction as rotation axis
            axis = random_directions[i]
            angle = angular_amplitudes[i][0]  # Extract the scalar angle value
            
            # Apply rotation formula: v_rot = v * cos(θ) + axis * sin(θ)
            sparks[i] = centroid * np.cos(angle) + axis * np.sin(angle)
        
        # Normalize the resulting vectors (should already be unit length but for numerical stability)
        sparks = np.apply_along_axis(self.normalize_vector, 1, sparks)
        
        return sparks
    
    def evaluate_spark(self, data, spark, current_centroids, centroid_idx):
        candidate_centroids = current_centroids.copy()
        candidate_centroids[centroid_idx] = spark
        
        temp_labels = self.assign_clusters(data, candidate_centroids)
        fitness = self.calculate_fitness(data, temp_labels)
        
        return fitness, candidate_centroids

    def mapping_rule(self, position):
        """Apply mapping rule to ensure position is within bounds"""
        min_bounds = np.min(self.normalized_data, axis=0)
        max_bounds = np.max(self.normalized_data, axis=0)
        
        # Apply periodic boundary conditions
        mapped_position = min_bounds + (position - min_bounds) % (max_bounds - min_bounds)
        
        return mapped_position

    def calculate_centroid_fitness(self, cluster_idx, labels):
        mask = labels == cluster_idx
        if np.sum(mask) > 0:
            cluster_data = self.normalized_data[mask]
            if len(cluster_data) > 1:
                # Calculate cluster centroid
                cluster_centroid = np.mean(cluster_data, axis=0)
                cluster_centroid = self.normalize_vector(cluster_centroid)
                
                # Calculate similarity to centroid
                similarities = np.dot(cluster_data, cluster_centroid)
                return np.mean(similarities)
            else:
                print("Single point cluster detected")
                return 0.5
        else:
            print("Zero point cluster detected!")
            return 0.0

    def track_centroid_angular_changes(self, current_centroids, previous_centroids):
        angular_changes = []
        
        for i in range(len(current_centroids)):
            # Calculate the cosine of the angle between the vectors
            # Using the dot product formula: cos(θ) = (a·b)/(|a|*|b|)
            # Since vectors are normalized, the denominator is 1
            cos_angle = np.clip(np.dot(current_centroids[i], previous_centroids[i]), -1.0, 1.0)
            
            # Convert to angle in degrees
            angle_degrees = np.arccos(cos_angle) * (180.0 / np.pi)
            
            angular_changes.append(angle_degrees)
        
        return angular_changes        
    
    def optimize(self, data):
        self.normalized_data = np.apply_along_axis(self.normalize_vector, 1, np.copy(data))
        data_dim = data.shape[1]
        self.all_angular_changes = []
        
        # Initialize a single firework (set of centroids)
        centroids = self.initialize_centroids(self.normalized_data)
        
        # Calculate initial fitness
        labels = self.assign_clusters(self.normalized_data, centroids)
        print(f'Unique labels: {np.unique(labels)}')
        current_fitness = self.calculate_fitness(self.normalized_data, labels)
        print(f'Current Fitness: {current_fitness}')
        
        # Initialize best solution tracking
        self.best_fitness = current_fitness
        self.global_best_fitness = current_fitness
        self.best_centroids = centroids.copy()
        self.best_assignments = labels.copy()
        
        # Initial cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        self.cluster_sizes.append(dict(zip(unique, counts)))
        
        # # Calculate initial silhouette
        # initial_silhouette = self.compute_silhouette(normalized_data, labels)
        # self.silhouette_history.append((0, initial_silhouette))
        
        # Main algorithm loop
        pbar = tqdm(range(self.max_iter), desc="Firework Algorithm")
        
        for iteration in pbar:
            iteration_start_time = time.time()
            iteration_fitness_evals = 0
            
            # Store previous centroids for movement tracking
            previous_centroids = centroids.copy()
            
            cluster_counts = np.bincount(labels, minlength=self.n_clusters)
            print(f"Cluster counts: {cluster_counts.tolist()}")
                        
            centroid_fitness_scores = []
            centroid_fitness_scores = Parallel(n_jobs=self.n_jobs)(
                delayed(self.calculate_centroid_fitness)(i, labels) 
                for i in range(self.n_clusters)
            )
            
            
            # Calculate explosion amplitude for each centroid based on its quality
            amplitudes, spark_counts = self.calculate_explosion_params(centroid_fitness_scores)       
            
            # All candidate centroid sets and their fitness scores
            all_candidate_centroids = []
            all_fitness_scores = []
            
            # For each centroid in the firework
            for i in range(self.n_clusters):
                current_centroid = centroids[i]
                current_amplitude = amplitudes[i]
                
                # Generate uniform sparks
                total_sparks_for_centroid = spark_counts[i]
                n_regular = int(total_sparks_for_centroid * 0.8)
                n_gaussian = total_sparks_for_centroid - n_regular
                
                if total_sparks_for_centroid > 1:
                    n_regular = max(1, n_regular)
                    n_gaussian = max(1, total_sparks_for_centroid - n_regular)
                else:
                    # If only one spark, make it a regular one
                    n_regular = 1
                    n_gaussian = 0
                
                if n_regular > 0:
                    uniform_sparks = self.generate_sparks(
                        current_centroid, current_amplitude, n_regular, data_dim
                    )
                    
                    # Evaluate regular sparks
                    regular_results = Parallel(n_jobs=self.n_jobs)(
                        delayed(self.evaluate_spark)(
                            self.normalized_data, spark, centroids, i
                        ) for spark in uniform_sparks
                    )
                    
                    # Extract results
                    regular_fitness = [res[0] for res in regular_results]
                    regular_centroids = [res[1] for res in regular_results]
                    
                    # Add to candidates
                    all_candidate_centroids.extend(regular_centroids)
                    all_fitness_scores.extend(regular_fitness)
                    
                    # Count fitness evaluations
                    iteration_fitness_evals += len(uniform_sparks)
                    self.fitness_evaluations += len(uniform_sparks)
                    
                    # Track all fitness values
                    self.all_fitness_values.extend(regular_fitness)
                    self.all_fitness_timestamps.extend([time.time()] * len(regular_fitness))
                
                # Generate Gaussian sparks if needed
                if n_gaussian > 0:
                    gaussian_sparks = self.generate_gaussian_sparks(
                        current_centroid, current_amplitude, n_gaussian, data_dim
                    )
                    
                    # Evaluate Gaussian sparks
                    gaussian_results = Parallel(n_jobs=self.n_jobs)(
                        delayed(self.evaluate_spark)(
                            self.normalized_data, spark, centroids, i
                        ) for spark in gaussian_sparks
                    )
                    
                    # Extract results
                    gaussian_fitness = [res[0] for res in gaussian_results]
                    gaussian_centroids = [res[1] for res in gaussian_results]
                    
                    # Add to candidates
                    all_candidate_centroids.extend(gaussian_centroids)
                    all_fitness_scores.extend(gaussian_fitness)
                    
                    # Count fitness evaluations
                    iteration_fitness_evals += len(gaussian_sparks)
                    self.fitness_evaluations += len(gaussian_sparks)
                    
                    # Track all fitness values
                    self.all_fitness_values.extend(gaussian_fitness)
                    self.all_fitness_timestamps.extend([time.time()] * len(gaussian_fitness))
                
            # Find the best candidate solution
            best_idx = np.argmax(all_fitness_scores)
            iteration_best_fitness = all_fitness_scores[best_idx]
            iteration_best_centroids = all_candidate_centroids[best_idx]
            
            # Update current solution
            centroids = iteration_best_centroids.copy()
            
            # Calculate angular changes for each centroid
            if iteration > 0:  # Skip first iteration since we don't have previous centroids
                angular_changes = self.track_centroid_angular_changes(centroids, previous_centroids)
                self.all_angular_changes.append(angular_changes)
                
                # Optional: print angular changes
                print(f"Iteration {iteration} angular changes (degrees): {angular_changes}")
                
            labels = self.assign_clusters(self.normalized_data, centroids)
            current_fitness = iteration_best_fitness
            
            # Update best solution if improvement found
            if current_fitness > self.global_best_fitness:
                self.global_best_fitness = current_fitness
                self.best_centroids = centroids.copy()
                self.best_assignments = labels.copy()
            
            # Calculate centroid movement (angular change in degrees)
            centroid_shifts = np.arccos(np.clip(
                np.sum(centroids * previous_centroids, axis=1), -1.0, 1.0
            )) * (180.0 / np.pi)
            self.centroid_movement.append(np.mean(centroid_shifts))
            
            # Update tracking metrics
            self.best_fitness_history.append(current_fitness)
            self.global_best_fitness_history.append(self.global_best_fitness)
            # self.fitness_history.append(current_fitness)
            
            # Track cluster sizes
            unique, counts = np.unique(labels, return_counts=True)
            self.cluster_sizes.append(dict(zip(unique, counts)))
            
            # # Calculate silhouette periodically
            # if iteration % 5 == 0 or iteration == self.max_iter - 1:
            #     silhouette = self.compute_silhouette(normalized_data, labels)
            #     self.silhouette_history.append((iteration, silhouette))
            
            # Track evaluations and time
            self.fitness_evaluations_per_iteration.append(iteration_fitness_evals)
            iteration_end_time = time.time()
            iteration_time = iteration_end_time - iteration_start_time
            self.iteration_times.append(iteration_time)
            
            # Update progress bar
            pbar.set_postfix({
                'Fitness': f'{current_fitness:.2f}',
                'Global Best': f'{self.global_best_fitness:.2f}',
                'Evals': f'{iteration_fitness_evals}'
            })
        
        # # Calculate final silhouette score
        # final_silhouette = self.compute_silhouette(normalized_data, self.best_assignments)
        
        # Compile performance metrics
        perf_metrics = {
            'best_fitness_history': self.global_best_fitness_history,
            'fitness_evaluations': self.fitness_evaluations,
            'fitness_evaluations_per_iteration': self.fitness_evaluations_per_iteration,
            'iteration_times': self.iteration_times
            # 'final_silhouette': final_silhouette
        }
        
        return self.best_centroids, self.global_best_fitness, perf_metrics
    
    def visualize_performance(self, show=True, save_path=None):
        plt.figure(figsize=(18, 12))
        
        plt.subplot(2, 3, 1)
        plt.plot(self.global_best_fitness_history,'g-',label='Global Best Fitness')
        plt.plot(self.best_fitness_history, 'r--', label='Current Fitness')
        plt.title('Fitness History')
        plt.xlabel('Iteration')
        plt.ylabel('Calinski-Harabasz Score')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 2)
        plt.plot(self.iteration_times, 'g-')
        plt.title('Iteration Execution Times')
        plt.xlabel('Iteration')
        plt.ylabel('Time (seconds)')
        plt.grid(True)
        
        plt.subplot(2, 3, 4)
        if len(self.centroid_movement) > 0:
            plt.plot(self.centroid_movement, 'm-')
            plt.title('Average Centroid Movement')
            plt.xlabel('Iteration')
            plt.ylabel('Average Angular Change (degrees)')
            plt.grid(True)
        
        plt.subplot(2, 3, 5)
        plt.plot(self.fitness_evaluations_per_iteration, 'r-')
        plt.title('Fitness Evaluations per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Number of Evaluations')
        plt.grid(True)
        
        plt.subplot(2, 3, 6)
        if len(self.all_fitness_values) > 0:
            plt.hist(self.all_fitness_values, bins=20, color='orange', alpha=0.7)
            plt.title('Distribution of Fitness Values')
            plt.xlabel('Fitness Score')
            plt.ylabel('Frequency')
            plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return self
    
    def get_performance_data(self):
        return {
            'algorithm': 'EFWA',
            'run_id': self.run_id,
            'iteration_times': self.iteration_times,
            'fitness_evaluations': self.fitness_evaluations,
            # 'fitness_history': self.fitness_history,
            'best_fitness_history': self.best_fitness_history,
            'global_best_fitness_history': self.global_best_fitness_history, 
            'all_fitness_values': self.all_fitness_values,
            'all_fitness_timestamps': self.all_fitness_timestamps,
            'centroid_movement': self.centroid_movement,
            'final_fitness': self.global_best_fitness,
            'fitness_evaluations_per_iteration': self.fitness_evaluations_per_iteration
        }