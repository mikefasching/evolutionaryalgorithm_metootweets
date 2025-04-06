import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score, silhouette_score
import progressbar
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import time
from joblib import Parallel, delayed
import pandas as pd
from scipy import stats
from scipy import stats as scipy_stats
import os

class BeeClustering:
    def __init__(self, tweets_vectors, n_clusters=5, num_bees=20, max_iter=100, scout_limit=10, n_jobs=-1, emp_per=0.2, onlook_per=0.1, run_id=0):
        """
        Bee Optimization Algorithm with performance tracking.
        
        Parameters:
        -----------
        tweets_vectors : array-like
            The data to cluster
        n_clusters : int
            Number of clusters
        num_bees : int
            Number of bee agents
        max_iter : int
            Maximum number of iterations
        scout_limit : int
            Limit for scout bee phase
        n_jobs : int
            Number of parallel jobs
        emp_per : float
            Employed bee perturbation percentage
        onlook_per : float
            Onlooker bee perturbation percentage
        run_id : int
            Identifier for multiple runs
        """
        self.n_jobs = n_jobs
        self.scaler = MinMaxScaler()
        self.tweets_vectors = self.scaler.fit_transform(tweets_vectors)
        
        self.n_clusters = n_clusters
        self.num_bees = num_bees
        self.max_iter = max_iter
        self.scout_limit = scout_limit
        self.emp_per = emp_per
        self.onlook_per = onlook_per
        self.run_id = run_id
        
        # Performance tracking metrics
        self.iteration_times = []
        self.fitness_evaluations = 0
        self.fitness_evaluations_per_iteration = []
        self.fitness_history = []
        self.best_fitness_history = []
        self.silhouette_history = []
        self.iteration_best_fitness = []
        
        # Detailed metrics for statistical comparison
        self.all_fitness_values = []  # Store all fitness values calculated
        self.all_fitness_timestamps = []  # When each fitness was calculated
        self.all_silhouette_values = [] # Store all silhouette values calculated
        
        # Centroid tracking
        self.centroid_movement = []  # Track centroid movements
        self.cluster_sizes = []  # Track cluster sizes
        
        # Initialize Centroids
        self.centroids = self._initialize_centroids()
        
        # Initialize bee solutions (random centroids)
        self.bees = [self._initialize_centroids() for _ in range(num_bees)]
        
        # Parallel fitness computation for initial bees
        start_time = time.time()
        self.fitness = np.array(Parallel(n_jobs=self.n_jobs)(
            delayed(self._compute_fitness)(centroids) for centroids in self.bees
        ))

        # Find the best initial solution
        self.best_index = np.argmax(self.fitness)
        self.best_initial_centroids = self.bees[self.best_index].copy()

        # Save the best initialization, overwriting any existing file
        centroids_filename = f'initial_centroids_{self.n_clusters}.npy'
        np.save(centroids_filename, self.best_initial_centroids)
        print(f"Saved best initial centroids (score: {self.fitness[self.best_index]:.2f}) to {centroids_filename}")

        # Track fitness evaluations
        for f in self.fitness:
            self.all_fitness_values.append(f)
            self.all_fitness_timestamps.append(time.time() - start_time)
            
        # Store initial best fitness
        self.best_fitness = max(self.fitness)
        self.best_fitness_history.append(self.best_fitness)
        self.fitness_evaluations_per_iteration.append(num_bees)
        
        # Initialize data for statistical significance
        self.best_solution = None
        self.final_clusters = None

    def _initialize_centroids(self):
        """Initialize one set of centroids using cosine similarity approach."""
        n_samples = self.tweets_vectors.shape[0]
    
        # Normalize tweet vectors for cosine similarity
        normalized_tweets = self.tweets_vectors / (np.linalg.norm(self.tweets_vectors, axis=1, keepdims=True) + 1e-10)
        
        # Randomly select the first centroid
        centroids = [normalized_tweets[np.random.randint(n_samples)]]
        
        for _ in range(1, self.n_clusters):
            # Compute cosine similarity between each point and selected centroids
            similarities = np.max(cosine_similarity(normalized_tweets, np.array(centroids)), axis=1)
            
            # Compute cosine dissimilarity
            dissimilarities = np.abs(1 - similarities)
            
            # Create probability distribution
            total_dissimilarity = dissimilarities.sum() + 1e-10
            probabilities = dissimilarities / total_dissimilarity
            
            # Add randomness to selection process
            if np.random.rand() < 0.5:  # 50% chance to select based on dissimilarity
                next_idx = np.random.choice(n_samples, p=probabilities)
            else:  # 50% chance to select a random point
                next_idx = np.random.randint(n_samples)
            
            centroids.append(normalized_tweets[next_idx])
        
        return np.array(centroids)

    def _assign_clusters(self, centroids):
        """Assign tweets to nearest centroids using cosine similarity"""
        similarities = cosine_similarity(self.tweets_vectors, centroids)
        return np.argmax(similarities, axis=1)

    def _update_centroids(self, clusters):
        """Update centroids based solely on the angular direction of assigned tweets."""
        n_features = self.tweets_vectors.shape[1]
        new_centroids = np.zeros((self.n_clusters, n_features))
        counts = np.zeros(self.n_clusters)
    
        # Accumulate vectors for each cluster
        np.add.at(new_centroids, clusters, self.tweets_vectors)
        np.add.at(counts, clusters, 1)
    
        # Normalize to get directions
        for cluster in range(self.n_clusters):
            if counts[cluster] > 0:
                new_centroids[cluster] /= np.linalg.norm(new_centroids[cluster])
            else:
                # Keep the old centroid if no points are assigned
                new_centroids[cluster] = self.centroids[cluster]  # Or any default value
    
        return new_centroids
        
        # """Update centroids as the mean of assigned tweets"""
        # return np.array(Parallel(n_jobs=self.n_jobs)(
        #     delayed(lambda i: np.mean(self.tweets_vectors[clusters == i], axis=0)
        #             if len(self.tweets_vectors[clusters == i]) > 0
        #             else self.tweets_vectors[np.random.randint(len(self.tweets_vectors))])(i)
        #     for i in range(self.n_clusters)
        # ))

    def _compute_fitness(self, centroids):
        """Compute clustering quality using Calinski-Harabasz Index with a penalty for unused clusters"""
        clusters = self._assign_clusters(centroids)
        unique_clusters = len(set(clusters))
        
        # Only calculate score if we have at least 2 clusters
        if unique_clusters > 1:
            base_score = calinski_harabasz_score(self.tweets_vectors, clusters)
            
            # Apply scaled penalty if we have fewer clusters than requested
            if unique_clusters < self.n_clusters:
                # Calculate penalty percentage (e.g., 20% per missing cluster)
                penalty_per_missing = 0.2  # 20% penalty per missing cluster
                missing_clusters = self.n_clusters - unique_clusters
                penalty_factor = penalty_per_missing * missing_clusters
                
                # Cap the penalty at 90% to avoid zero or negative fitness
                penalty_factor = min(penalty_factor, 0.9)
                
                # Apply the penalty to the base score
                return base_score * (1 - penalty_factor)
            else:
                return base_score
        else:
            return 0  # Return 0 if we have fewer than 2 unique clusters

    def _compute_silhouette(self, centroids):
        """Compute silhouette score for clustering"""
        clusters = self._assign_clusters(centroids)
        try:
            return silhouette_score(self.tweets_vectors, clusters, sample_size=min(1000, len(self.tweets_vectors)))
        except:
            return -1  # Handle edge cases

    def _employed_bee_phase(self, i):
        """
        Advanced employed bee phase with multi-dimensional angular changes.
        Modified to return only (new_solution, new_fitness) to work with existing loop.
        """
        current_solution = self.bees[i].copy()
        new_solution = current_solution.copy()
        data_dim = current_solution.shape[1]
        
        # For each centroid in the solution
        for c in range(self.n_clusters):
            # Choose a random neighbor (different from current bee)
            k = i
            while k == i:
                k = np.random.randint(0, self.num_bees)
            
            # Current and neighbor centroids
            current_centroid = current_solution[c]
            neighbor_centroid = self.bees[k][c]
            
            # Calculate angle between centroids
            cos_angle = np.clip(np.dot(current_centroid, neighbor_centroid), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            # If centroids are not perfectly aligned
            if angle > 1e-6:
                # Calculate the orthogonal component (direction of rotation)
                ortho_component = neighbor_centroid - cos_angle * current_centroid
                ortho_norm = np.linalg.norm(ortho_component)
                
                if ortho_norm > 1e-10:
                    # Normalize the orthogonal component to create rotation axis
                    rotation_axis = ortho_component / ortho_norm
                    
                    # Apply phi factor from original ABC formula
                    phi = np.random.uniform(-1, 1) * self.emp_per
                    
                    # Scale the angular change
                    angular_change = phi * angle
                    
                    # Apply rotation using Rodrigues' formula
                    cos_rot = np.cos(angular_change)
                    sin_rot = np.sin(angular_change)
                    
                    # Apply rotation
                    new_centroid = current_centroid * cos_rot + rotation_axis * sin_rot
                    
                    # Normalize the new centroid
                    new_solution[c] = new_centroid / np.linalg.norm(new_centroid)
                    
                    # For multi-dimensional perturbation, add small rotations in other planes
                    if data_dim > 3 and np.random.random() < 0.3:  # 30% chance
                        # Create basis for orthogonal subspace
                        basis_vectors = [current_centroid, rotation_axis]
                        
                        # Generate 1-2 additional orthogonal basis vectors
                        for _ in range(min(2, data_dim - 2)):
                            # Random vector
                            random_vec = np.random.randn(data_dim)
                            
                            # Make it orthogonal to existing basis
                            for basis in basis_vectors:
                                random_vec = random_vec - np.dot(random_vec, basis) * basis
                            
                            # Normalize if possible
                            random_norm = np.linalg.norm(random_vec)
                            if random_norm > 1e-10:
                                new_basis = random_vec / random_norm
                                basis_vectors.append(new_basis)
                                
                                # Apply small rotation in this dimension
                                small_phi = np.random.uniform(-0.5, 0.5) * self.emp_per * 0.5
                                small_angle = small_phi * np.pi/8  # Max ±22.5 degrees
                                
                                cos_small = np.cos(small_angle)
                                sin_small = np.sin(small_angle)
                                
                                # Apply small rotation
                                rotated = new_solution[c] * cos_small + new_basis * sin_small
                                
                                # Renormalize
                                new_solution[c] = rotated / np.linalg.norm(rotated)
        
        # Evaluate fitness of the new solution
        new_fitness = self._compute_fitness(new_solution)
        
        # Only return the new solution and its fitness - let the loop handle trial counters
        return new_solution, new_fitness    
        

    def _onlooker_bee_phase(self, i):
        """
        Onlooker bee phase - reuses employed bee phase with different perturbation parameter.
        """
        # Save original perturbation value
        original_emp_per = self.emp_per
        
        # Set perturbation to onlooker value
        self.emp_per = self.onlook_per
        
        # Call the employed bee phase function
        new_solution, new_fitness = self._employed_bee_phase(i)
        
        # Restore original perturbation value
        self.emp_per = original_emp_per
        
        return new_solution, new_fitness
        
    def _initialize_scout_centroids(self):
        """Reinitialize scout centroids with modified cosine similarity approach."""
        n_samples = self.tweets_vectors.shape[0]
        
        # Normalize tweet vectors for cosine similarity
        normalized_tweets = self.tweets_vectors / (np.linalg.norm(self.tweets_vectors, axis=1, keepdims=True) + 1e-10)
        
        # Randomly select the first centroid
        centroids = [normalized_tweets[np.random.randint(n_samples)]]
        
        for _ in range(1, self.n_clusters):
            # Compute cosine similarity between each point and selected centroids
            similarities = np.max(cosine_similarity(normalized_tweets, np.array(centroids)), axis=1)
            
            # Compute cosine dissimilarity
            dissimilarities = np.abs(1 - similarities)
            
            # Create probability distribution
            total_dissimilarity = dissimilarities.sum() + 1e-10
            probabilities = dissimilarities / total_dissimilarity
            
            # Different randomness balance for scouts (e.g., 70% chance to select based on dissimilarity)
            if np.random.rand() < 0.7:  # Higher probability for dissimilarity-based selection
                next_idx = np.random.choice(n_samples, p=probabilities)
            else:
                next_idx = np.random.randint(n_samples)
            
            centroids.append(normalized_tweets[next_idx])
        
        return np.array(centroids)

    def _compute_probabilities(self):
        """Compute selection probabilities for onlooker bees using softmax"""
        fitness_min, fitness_max = np.min(self.fitness), np.max(self.fitness)
        if fitness_max == fitness_min:
            return np.ones_like(self.fitness) / len(self.fitness)
            
        normalized_fitness = (self.fitness - fitness_min) / (fitness_max - fitness_min)
        exp_fitness = np.exp(normalized_fitness * 10)
        probabilities = exp_fitness / np.sum(exp_fitness)
        return probabilities

    def _calculate_centroid_movement(self, old_centroids, new_centroids):
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
        

    def optimize(self):
        """Main optimization loop with performance tracking"""
        best_solution = self.best_initial_centroids
        best_fitness = self.fitness[self.best_index]
        scout_counters = np.zeros(self.num_bees)
        # At the start of the optimize method
        prev_bee_centroids = [bee.copy() for bee in self.bees]
        silhouette = self._compute_silhouette(best_solution)

        print(f"ABC Initialization - Fitness: {self.fitness[self.best_index]:.2f}, Silhouette: {silhouette:.4f}")
 
        
        overall_start_time = time.time()
        
        # Progress bar setup
        bar = progressbar.ProgressBar(max_value=self.max_iter, 
                                      widgets=['ABC Optimizing: ', progressbar.Percentage(), ' ', 
                                               progressbar.Bar(marker='█'), ' ',
                                               progressbar.ETA(), ' ',
                                               progressbar.DynamicMessage('best_fitness')])

        for iteration in range(self.max_iter):
            iteration_start = time.time()

            evals_before_iteration = self.fitness_evaluations # To calcualte the call of evaluation function per iteration
            
            # Store previous best centroids for movement tracking
            # prev_best_centroids = self.bees[np.argmax(self.fitness)].copy() if iteration > 0 else None
            
            # Employed Bees Phase (Parallelized)
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._employed_bee_phase)(i) for i in range(self.num_bees)
            )
            self.fitness_evaluations += self.num_bees
           
            for i, (new_centroids, new_fitness) in enumerate(results):
                if new_fitness > self.fitness[i]:
                    self.bees[i] = new_centroids
                    self.fitness[i] = new_fitness
                    scout_counters[i] = 0
                else:
                    scout_counters[i] += 1
            
            # Record best fitness after employed phase
            current_best_idx = np.argmax(self.fitness)
            current_best_fitness = self.fitness[current_best_idx]
            self.iteration_best_fitness.append(current_best_fitness)
            
            # Onlooker Bees Phase (Selection)
            probabilities = self._compute_probabilities()
            selected_bees = np.random.choice(self.num_bees, self.num_bees // 2, p=probabilities)
            
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._onlooker_bee_phase)(i) for i in selected_bees
            )
            self.fitness_evaluations += len(selected_bees)
            
            for i, (new_centroids, new_fitness) in zip(selected_bees, results):
                if new_fitness > self.fitness[i]:
                    self.bees[i] = new_centroids
                    self.fitness[i] = new_fitness
                    scout_counters[i] = 0
            
            # Scout Bees Phase (Reinitialize stagnating bees)
            for i in range(self.num_bees):
                if scout_counters[i] > self.scout_limit:
                    # Reinitialize the bee's centroids
                    self.bees[i] = self._initialize_scout_centroids()
                    new_fitness = self._compute_fitness(self.bees[i])
                    
                    # Check if the new fitness is better
                    # if new_fitness > self.fitness[i]:  # Only update if it's better
                    self.fitness[i] = new_fitness
                    
                    self.fitness_evaluations += 1
                    scout_counters[i] = 0
                    
                    # Track scout phase fitness
                    self.all_fitness_values.append(new_fitness)
                    self.all_fitness_timestamps.append(time.time() - overall_start_time)
                    # print(f"Scout for bee {i}")
                    # self.bees[i] = self._initialize_scout_centroids()
                    # new_fitness = self._compute_fitness(self.bees[i])
                    # self.fitness[i] = new_fitness
                    # self.fitness_evaluations += 1
                    # scout_counters[i] = 0
                    
                    # # Track scout phase fitness
                    # self.all_fitness_values.append(new_fitness)
                    # self.all_fitness_timestamps.append(time.time() - overall_start_time)
                    # print(f"Scout for bee {i}")

            
            # print(f'Scout Counter: {scout_counters} - Iteration: {iteration}')
            # Update best solution
            max_fitness_idx = np.argmax(self.fitness)
            if self.fitness[max_fitness_idx] > best_fitness:
                best_fitness = self.fitness[max_fitness_idx]
                best_solution = self.bees[max_fitness_idx]
                
                # Track cluster sizes for best solution
                cluster_assignments = self._assign_clusters(best_solution)
                unique_clusters, cluster_counts = np.unique(cluster_assignments, return_counts=True)
                self.cluster_sizes.append(dict(zip(unique_clusters, cluster_counts)))
            
            # Calculate centroid movement if we have previous centroids
            # Then in each iteration
            bee_movements = []
            for i in range(self.num_bees):
                movement = self._calculate_centroid_movement(prev_bee_centroids[i], self.bees[i])
                bee_movements.append(movement)
            
            # Average movement across all bees
            avg_movement = np.mean(bee_movements)
            self.centroid_movement.append(avg_movement)

            # Update previous centroids for next iteration
            prev_bee_centroids = [bee.copy() for bee in self.bees]
            
            # Calculate additional metrics periodically
            if iteration % 5 == 0 or iteration == self.max_iter - 1:
                # Calculate silhouette score for best solution
                silhouette = self._compute_silhouette(best_solution)
                self.silhouette_history.append((iteration, silhouette))
                self.all_silhouette_values.append(silhouette)
            
            # Track fitness history after each iteration
            self.fitness_history.append(np.mean(self.fitness))
            self.best_fitness_history.append(best_fitness)

            evals_this_iteration = self.fitness_evaluations - evals_before_iteration
            self.fitness_evaluations_per_iteration.append(evals_this_iteration)
            
            # Track time for this iteration
            iteration_end = time.time()
            self.iteration_times.append(iteration_end - iteration_start)
            
            # Update progress bar
            bar.update(iteration, best_fitness=best_fitness)
        
        self.best_solution = best_solution
        self.final_clusters = self._assign_clusters(best_solution)
        
        # Return best solution, its fitness, and a summary of performance metrics
        performance_metrics = {
            'total_time': sum(self.iteration_times),
            'avg_iteration_time': np.mean(self.iteration_times),
            'fitness_evaluations': self.fitness_evaluations,
            'fitness_evaluations_per_iteration': self.fitness_evaluations_per_iteration,
            'best_fitness': best_fitness,
            'final_silhouette': self._compute_silhouette(best_solution),
            'best_fitness_history': self.best_fitness_history
        }
        
        return best_solution, best_fitness, performance_metrics

    def predict(self, centroids=None):
        """Assign tweets to the final optimized centroids"""
        if centroids is None:
            centroids = self.best_solution
        return self._assign_clusters(centroids)
    
    def get_performance_data(self):
        """Return all performance data for statistical analysis"""
        return {
            'algorithm': 'ABC',
            'run_id': self.run_id,
            'iteration_times': self.iteration_times,
            'fitness_evaluations': self.fitness_evaluations,
            'fitness_evaluations_per_iteration': self.fitness_evaluations_per_iteration,
            'fitness_history': self.fitness_history,
            'best_fitness_history': self.best_fitness_history,
            'all_fitness_values': self.all_fitness_values,
            'all_fitness_timestamps': self.all_fitness_timestamps,
            'centroid_movement': self.centroid_movement,
            'silhouette_scores': self.silhouette_history,
            'all_silhouette_values': self.all_silhouette_values,
            'final_fitness': max(self.fitness) if len(self.fitness) > 0 else 0,
            'cluster_sizes': self.cluster_sizes
        }
    
    def visualize_performance(self, show=True, save_path=None):
        """Visualize algorithm performance metrics"""
        plt.figure(figsize=(18, 12))
        
        # Plot 1: Fitness History
        plt.subplot(2, 2, 1)
        plt.plot(self.best_fitness_history, 'r-', label='Best Fitness')
        plt.plot(self.fitness_history, 'b--', label='Average Fitness')
        plt.title('Fitness History')
        plt.xlabel('Iteration')
        plt.ylabel('Calinski-Harabasz Score')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Iteration Times
        plt.subplot(2, 2, 2)
        plt.plot(self.iteration_times, 'g-')
        plt.title('Iteration Execution Times')
        plt.xlabel('Iteration')
        plt.ylabel('Time (seconds)')
        plt.grid(True)
        
        # Plot 3: Centroid Movement
        plt.subplot(2, 2, 3)
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
        
        # Plot 4: Silhouette Scores
        plt.subplot(2, 2, 4)
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
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return self

def run_abc_with_stats(tweets_vectors, n_clusters=10, num_bees=15, max_iter=50, 
                       scout_limit=5, n_jobs=-1, emp_per=0.1, onlook_per=0.05, num_runs=5):
    """
    Run ABC multiple times to collect statistics
    
    Parameters:
    -----------
    tweets_vectors : array-like
        The data to cluster
    n_clusters : int
        Number of clusters
    num_bees : int
        Number of bee agents
    max_iter : int
        Maximum number of iterations
    scout_limit : int
        Limit for scout bee phase
    n_jobs : int
        Number of parallel jobs
    emp_per : float
        Employed bee perturbation percentage
    onlook_per : float
        Onlooker bee perturbation percentage
    num_runs : int
        Number of runs for statistical analysis
    
    Returns:
    --------
    DataFrame with statistics and performance metrics
    """
    results = []
    all_run_data = []
    
    for run in range(num_runs):
        print(f"ABC Run {run+1}/{num_runs}")
        boa = BeeClustering(
            tweets_vectors, 
            n_clusters=n_clusters, 
            num_bees=num_bees, 
            max_iter=max_iter, 
            scout_limit=scout_limit, 
            n_jobs=n_jobs, 
            emp_per=emp_per, 
            onlook_per=onlook_per, 
            run_id=run
        )
        
        best_centroids, best_fitness, perf_metrics = boa.optimize()
        
        # Store results for this run
        results.append({
            'run_id': run,
            'best_fitness': best_fitness,
            'total_time': perf_metrics['total_time'],
            'avg_iteration_time': perf_metrics['avg_iteration_time'],
            'fitness_evaluations': perf_metrics['fitness_evaluations'],
            'final_silhouette': perf_metrics['final_silhouette']
        })
        
        # Store detailed performance data
        all_run_data.append(boa.get_performance_data())
        
        # Visualize this run
        boa.visualize_performance(show=False, save_path=f"abc_performance_run_{run}.png")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate statistics
    stats = {
        'algorithm': 'ABC',
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