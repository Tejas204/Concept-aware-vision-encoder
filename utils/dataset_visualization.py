import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer():
    def __init__(self, data_path, storage_path):
        self.data_path = data_path
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

        # Open data
        with open(self.data_path, "r") as file:
            self.data = json.load(file)

    def _save_figure(self, figure, filename):
        """
        --------------------------------------------------------------------------------------------
        Save a matplotlib figure to the configured storage directory.

        Args:
            figure: Matplotlib figure object to save.
            filename: File name to use when writing the figure.

        Returns:
            None.

        --------------------------------------------------------------------------------------------
        """
        figure_path = os.path.join(self.storage_path, filename)
        figure.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(figure)

    def _plot_frequency_bar(self, ax, frequency_map, title, k=None):
        """
        --------------------------------------------------------------------------------------------
        Draw a seaborn bar plot for a frequency dictionary.

        Args:
            ax: Matplotlib axis to draw on.
            frequency_map: Dictionary of labels and their counts.
            title: Title for the plot.
            k: Optional limit for the number of top entries to plot.

        Returns:
            None.

        --------------------------------------------------------------------------------------------
        """
        if not frequency_map:
            ax.set_title(title)
            ax.set_xlabel("Class")
            ax.set_ylabel("Frequency")
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            return

        sorted_items = sorted(frequency_map.items(), key=lambda item: item[1], reverse=True)
        if k is not None:
            sorted_items = sorted_items[:k]

        labels = [str(item[0]) for item in sorted_items]
        values = [item[1] for item in sorted_items]

        sns.barplot(x=labels, y=values, ax=ax, color="#4C72B0")
        ax.set_title(title)
        ax.set_xlabel("Class")
        ax.set_ylabel("Frequency")
        ax.tick_params(axis="x", rotation=45)

        for patch, value in zip(ax.patches, values):
            ax.annotate(
                f"{value}",
                (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
            )

    def _plot_stacked_frequency_bar(self, ax, overall_map, train_map, test_map, val_map, title, k=None):
        """
        --------------------------------------------------------------------------------------------
        Draw a stacked bar plot for overall, train, test, and validation counts.

        Args:
            ax: Matplotlib axis to draw on.
            overall_map: Dictionary with overall frequencies.
            train_map: Dictionary with train split frequencies.
            test_map: Dictionary with test split frequencies.
            val_map: Dictionary with validation split frequencies.
            title: Title for the plot.
            k: Optional limit for the number of top entries to plot.

        Returns:
            None.

        --------------------------------------------------------------------------------------------
        """
        all_labels = list(overall_map.keys())
        if not all_labels:
            ax.set_title(title)
            ax.set_xlabel("Class")
            ax.set_ylabel("Frequency")
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            return

        sorted_labels = sorted(all_labels, key=lambda label: overall_map.get(label, 0), reverse=True)
        if k is not None:
            sorted_labels = sorted_labels[:k]

        overall_values = [overall_map.get(label, 0) for label in sorted_labels]
        train_values = [train_map.get(label, 0) for label in sorted_labels]
        test_values = [test_map.get(label, 0) for label in sorted_labels]
        val_values = [val_map.get(label, 0) for label in sorted_labels]

        x_positions = np.arange(len(sorted_labels))
        colors = sns.color_palette("Set2", 3)

        ax.bar(x_positions, train_values, color=colors[0], label="Train")
        ax.bar(x_positions, test_values, bottom=train_values, color=colors[1], label="Test")
        stacked_top = np.array(train_values) + np.array(test_values)
        ax.bar(x_positions, val_values, bottom=stacked_top, color=colors[2], label="Val")

        ax.bar(x_positions, overall_values, facecolor="none", edgecolor="#2F2F2F", linewidth=1.5, label="Overall")

        for xpos, total in zip(x_positions, overall_values):
            ax.annotate(
                f"{total}",
                (xpos, total),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
            )

        ax.set_title(title)
        ax.set_xlabel("Class")
        ax.set_ylabel("Frequency")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(sorted_labels, rotation=45, ha="right")
        ax.legend(frameon=False)

    
    def compute_subject_object_frequency(self, k: int = None):
        """
        --------------------------------------------------------------------------------------------
        Build subject and object frequency plots for the full data and by split.

        Args:
            k: Optional limit for the number of top entries to plot.

        Returns:
            None.

        --------------------------------------------------------------------------------------------
        """
        # Initialize dictionaries
        subject_overall_frequency = {}
        object_overall_frequency = {}

        subject_train_frequency = {}
        object_train_frequency = {}

        subject_val_frequency = {}
        object_val_frequency = {}

        subject_test_frequency = {}
        object_test_frequency = {}

        for _, annotations in self.data.items():
            for annotation in annotations["annotations"]:
                # Get subject and object
                subject_name = annotation["subject"]["name"]
                object_name = annotation["object"]["name"]

                # Update overall frequency count
                subject_overall_frequency[subject_name] = subject_overall_frequency.get(subject_name, 0) + 1
                object_overall_frequency[object_name] = object_overall_frequency.get(object_name, 0) + 1

                # Update subject and object frequencies in train, test and val
                if annotations['split'] == 'train':
                    subject_train_frequency[subject_name] = subject_train_frequency.get(subject_name, 0) + 1
                    object_train_frequency[object_name] = object_train_frequency.get(object_name, 0) + 1

                elif annotations['split'] == 'test':
                    subject_test_frequency[subject_name] = subject_test_frequency.get(subject_name, 0) + 1
                    object_test_frequency[object_name] = object_test_frequency.get(object_name, 0) + 1

                else:
                    subject_val_frequency[subject_name] = subject_val_frequency.get(subject_name, 0) + 1
                    object_val_frequency[object_name] = object_val_frequency.get(object_name, 0) + 1

        # Plot top-k frequencies as 4 separate bar plots
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        self._plot_frequency_bar(axes[0, 0], subject_overall_frequency, "Overall Subject Frequency", k)
        self._plot_frequency_bar(axes[0, 1], object_overall_frequency, "Overall Object Frequency", k)
        self._plot_frequency_bar(axes[1, 0], subject_train_frequency, "Train Subject Frequency", k)
        self._plot_frequency_bar(axes[1, 1], object_train_frequency, "Train Object Frequency", k)

        plt.tight_layout()
        self._save_figure(fig, "subject_object_frequency.png")

        # Plot stacked split frequencies using a readable color palette.
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        self._plot_stacked_frequency_bar(
            axes[0],
            subject_overall_frequency,
            subject_train_frequency,
            subject_test_frequency,
            subject_val_frequency,
            "Stacked Subject Frequency by Split",
            k,
        )
        self._plot_stacked_frequency_bar(
            axes[1],
            object_overall_frequency,
            object_train_frequency,
            object_test_frequency,
            object_val_frequency,
            "Stacked Object Frequency by Split",
            k,
        )

        plt.tight_layout()
        self._save_figure(fig, "subject_object_stacked_frequency.png")


    def compute_pairwise_frequency(self, k: int = None):
        """
        --------------------------------------------------------------------------------------------
        Build pairwise subject-object frequency plots for the full data and by split.

        Args:
            k: Optional limit for the number of top entries to plot.

        Returns:
            None.

        --------------------------------------------------------------------------------------------
        """
        # Initialize pairwise frequency
        pairwise_overall_frequency = {}
        pairwise_train_frequency = {}
        pairwise_val_frequency = {}
        pairwise_test_frequency = {}

        for _, annotations in self.data.items():
            for annotation in annotations["annotations"]:
                # Get subject and object
                subject_name = annotation["subject"]["name"]
                object_name = annotation["object"]["name"]
                pair = (subject_name, object_name)

                # Update overall frequency count
                pairwise_overall_frequency[pair] = pairwise_overall_frequency.get(pair, 0) + 1

                # Update subject and object frequencies in train, test and val
                if annotations['split'] == 'train':
                    pairwise_train_frequency[pair] = pairwise_train_frequency.get(pair, 0) + 1

                elif annotations['split'] == 'test':
                    pairwise_test_frequency[pair] = pairwise_test_frequency.get(pair, 0) + 1

                else:
                    pairwise_val_frequency[pair] = pairwise_val_frequency.get(pair, 0) + 1

        # Plot top-k frequencies as 4 bar plots
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        self._plot_frequency_bar(
            axes[0, 0],
            {f"{subject} | {object_name}": value for (subject, object_name), value in pairwise_overall_frequency.items()},
            "Overall Pairwise Frequency",
            k,
        )
        self._plot_frequency_bar(
            axes[0, 1],
            {f"{subject} | {object_name}": value for (subject, object_name), value in pairwise_train_frequency.items()},
            "Train Pairwise Frequency",
            k,
        )
        self._plot_frequency_bar(
            axes[1, 0],
            {f"{subject} | {object_name}": value for (subject, object_name), value in pairwise_val_frequency.items()},
            "Validation Pairwise Frequency",
            k,
        )
        self._plot_frequency_bar(
            axes[1, 1],
            {f"{subject} | {object_name}": value for (subject, object_name), value in pairwise_test_frequency.items()},
            "Test Pairwise Frequency",
            k,
        )

        plt.tight_layout()
        self._save_figure(fig, "pairwise_frequency.png")

    def compute_predicate_frequency(self):
        predicate_overall_frequency = {}
        predicate_train_frequency = {}
        predicate_test_frequency = {}
        predicate_val_frequency = {}

        for _, annotations in self.data.items():
            for annotation in annotations["annotations"]:
                # Get subject and object
                spatial_concept = annotation["predicate"]

                # Update overall frequency count
                predicate_overall_frequency[spatial_concept] = predicate_overall_frequency.get(spatial_concept, 0) + 1

                # Update subject and object frequencies in train, test and val
                if annotations['split'] == 'train':
                    predicate_train_frequency[spatial_concept] = predicate_train_frequency.get(spatial_concept, 0) + 1

                elif annotations['split'] == 'test':
                    predicate_test_frequency[spatial_concept] = predicate_test_frequency.get(spatial_concept, 0) + 1

                else:
                    predicate_val_frequency[spatial_concept] = predicate_val_frequency.get(spatial_concept, 0) + 1

        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        self._plot_frequency_bar(axes[0, 0], predicate_overall_frequency, "Overall Predicate Frequency")
        self._plot_frequency_bar(axes[0, 1], predicate_train_frequency, "Train Predicate Frequency")
        self._plot_frequency_bar(axes[1, 0], predicate_test_frequency, "Test Predicate Frequency")
        self._plot_frequency_bar(axes[1, 1], predicate_val_frequency, "Validation Predicate Frequency")

        plt.tight_layout()
        self._save_figure(fig, "predicate_frequency.png")

        

                    
