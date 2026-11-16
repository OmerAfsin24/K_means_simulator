import numpy as np

class KMeans:

    def __init__(
        self,
        X,
        K=5,
        max_iterations=100,
        tolerance=0.05
    ):

        self.X = np.asarray(X, dtype=float)

        if self.X.ndim != 2 or self.X.shape[1] != 2:
            raise ValueError("X must contain exactly two features.")

        if len(self.X) < K or K < 1:
            raise ValueError("K must be between 1 and number of samples.")

        if not np.isfinite(self.X).all():
            raise ValueError("Dataset contains invalid values.")

        self.K = K
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self.rng = np.random.default_rng()

        self.centroids = None
        self.labels = None
        self.wcss = None

        self.history = []
        self.n_iterations = 0
        self.converged = False
        self.stop_reason = None

    # ----------------------------------
    # 1. Centroid Initialization
    # ----------------------------------

    def initialize_centroids(self):

        indices = self.rng.choice(
            len(self.X),
            self.K,
            replace=False
        )

        return self.X[indices].copy()

    # ----------------------------------
    # 2. Euclidean Distance
    # ----------------------------------

    @staticmethod
    def calculate_distance(a, b):

        return np.linalg.norm(a - b)

    # ----------------------------------
    # 3. Cluster Assignment
    # ----------------------------------

    def cluster_assignment(self, centroids):

        distances = np.linalg.norm(
            self.X[:, np.newaxis, :] -
            centroids[np.newaxis, :, :],
            axis=2
        )

        return np.argmin(distances, axis=1)

    # ----------------------------------
    # 4. Centroid Update
    # ----------------------------------

    def update_centroids(self, labels, centroids):

        new_centroids = centroids.copy()

        for k in range(self.K):

            cluster_points = self.X[labels == k]

            if len(cluster_points) > 0:

                new_centroids[k] = np.mean(
                    cluster_points,
                    axis=0
                )

            else:

                # Reinitialize empty cluster
                distances = np.min(
                    np.linalg.norm(
                        self.X[:, None, :] -
                        centroids[None, :, :],
                        axis=2
                    ),
                    axis=1
                )

                new_centroids[k] = self.X[
                    np.argmax(distances)
                ]

        return new_centroids

    # ----------------------------------
    # 5. WCSS Calculation
    # ----------------------------------

    def calculate_wcss(self, labels, centroids):

        distances = np.linalg.norm(
            self.X - centroids[labels],
            axis=1
        )

        return np.sum(distances ** 2)

    # ----------------------------------
    # 6. Fit Algorithm
    # ----------------------------------

    def fit(self):

        centroids = self.initialize_centroids()

        previous_labels = None

        self.history = []

        for iteration in range(self.max_iterations):

            old_centroids = centroids.copy()

            # Step 1: Assignment
            labels = self.cluster_assignment(centroids)

            # Step 2: Update
            new_centroids = self.update_centroids(
                labels,
                centroids
            )

            # Step 3: Calculate centroid movement
            movement = np.linalg.norm(
                new_centroids - old_centroids,
                axis=1
            )

            total_movement = np.sum(movement)

            # Step 4: Reassign using updated centroids
            new_labels = self.cluster_assignment(
                new_centroids
            )

            # Step 5: Calculate WCSS
            wcss = self.calculate_wcss(
                new_labels,
                new_centroids
            )

            assignment_converged = (
                previous_labels is not None
                and np.array_equal(
                    previous_labels,
                    new_labels
                )
            )

            centroid_converged = (
                total_movement < self.tolerance
            )

            # Save iteration
            self.history.append({

                "iteration": iteration + 1,

                "old_centroids": old_centroids.copy(),

                "centroids": new_centroids.copy(),

                "labels": new_labels.copy(),

                "wcss": wcss,

                "movement": movement.copy(),

                "total_movement": total_movement,

                "assignment_converged": assignment_converged,

                "centroid_converged": centroid_converged

            })

            centroids = new_centroids.copy()

            self.n_iterations = iteration + 1

            # Stop condition
            if assignment_converged or centroid_converged:

                self.converged = True

                self.stop_reason = (
                    "Assignments unchanged"
                    if assignment_converged
                    else "Centroid movement below tolerance"
                )

                break

            previous_labels = new_labels.copy()

        # Final results
        self.centroids = centroids

        self.labels = self.cluster_assignment(
            self.centroids
        )

        self.wcss = self.calculate_wcss(
            self.labels,
            self.centroids
        )

        if not self.converged:
            self.stop_reason = "Maximum iterations reached"

        return self
