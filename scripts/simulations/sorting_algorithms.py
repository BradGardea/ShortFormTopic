import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import random
from tqdm import tqdm


# Sorting algorithms
def bubble_sort(arr):
    n = len(arr)
    frames = []
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
            frames.append((arr.copy(), j, j + 1))  # Store the state after each swap with the indices of the bars swapped
    return frames

def selection_sort(arr):
    n = len(arr)
    frames = []
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
        frames.append((arr.copy(), i, min_idx))  # Store the state after each swap with the indices of the bars swapped
    return frames

# Insertion Sort
def insertion_sort(arr):
    n = len(arr)
    frames = []
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            frames.append((arr.copy(), j, j + 1))  # Store the state with indices of bars being compared
            j -= 1
        arr[j + 1] = key
        frames.append((arr.copy(), j + 1, i))  # After inserting the key
    return frames

# Merge Sort
def merge_sort(arr):
    frames = []

    def merge(left, mid, right):
        # Merging two halves
        n1 = mid - left + 1
        n2 = right - mid
        L = arr[left:mid + 1]
        R = arr[mid + 1:right + 1]

        i, j, k = 0, 0, left
        while i < n1 and j < n2:
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            frames.append((arr.copy(), k, k))  # Highlight current merge index
            k += 1

        while i < n1:
            arr[k] = L[i]
            frames.append((arr.copy(), k, k))
            i += 1
            k += 1

        while j < n2:
            arr[k] = R[j]
            frames.append((arr.copy(), k, k))
            j += 1
            k += 1

    def sort(left, right):
        if left < right:
            mid = left + (right - left) // 2
            sort(left, mid)
            sort(mid + 1, right)
            merge(left, mid, right)

    sort(0, len(arr) - 1)
    return frames

# Quick Sort
def quick_sort(arr):
    frames = []

    def partition(low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] < pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                frames.append((arr.copy(), i, j))  # Highlight swapped elements
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        frames.append((arr.copy(), i + 1, high))  # Highlight pivot swap
        return i + 1

    def sort(low, high):
        if low < high:
            pi = partition(low, high)
            sort(low, pi - 1)
            sort(pi + 1, high)

    sort(0, len(arr) - 1)
    return frames

# Heap Sort
def heap_sort(arr):
    frames = []

    def heapify(n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            frames.append((arr.copy(), i, largest))  # Highlight swapped elements
            heapify(n, largest)

    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(n, i)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        frames.append((arr.copy(), i, 0))  # Highlight root swap
        heapify(i, 0)

    return frames

# Create the figure and axis for the animation
def create_sorting_animation(n, output_path, sorting_algorithm='random', duration=5, ):
    rng = np.random.default_rng()
    arr = rng.choice(n, size=n, replace=False)

    # Map the sorting algorithm string to the actual function
    sorting_algorithms = {
        # 'bubble_sort': bubble_sort,
        'selection_sort': selection_sort,
        'insertion_sort': insertion_sort,
        'merge_sort': merge_sort,
        'quick_sort': quick_sort,
        'heap_sort': heap_sort,
    }
    if sorting_algorithm == "random":
        sorting_algorithm = random.choice(list(sorting_algorithms.keys()))

    print("Using algorithm: ", sorting_algorithm)
    # Check if the sorting algorithm exists in the map
    if sorting_algorithm not in sorting_algorithms:
        raise ValueError(f"Sorting algorithm {sorting_algorithm} is not supported.")

    # Get the sorting algorithm function
    sort_func = sorting_algorithms[sorting_algorithm]

    # Get the frames for the selected sorting algorithm
    frames = sort_func(arr.copy())

    # Set the background color to black and the bar color to white
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')  # Black background for the figure
    ax.set_facecolor('black')  # Black background for the axis
    ax.set_xlim(0, n)
    ax.set_ylim(0, 100)

    # Set the bars closer together (reduce the width)
    bars = ax.bar(range(n), arr, align='edge', color='white', width=0.8)

    # Function to create a gradient between two random colors
    def generate_gradient_color(start_color, end_color, n_bars):
        gradient = []
        for i in range(n_bars):
            # Interpolate between the two colors
            interpolated_color = [
                start_color[j] + (end_color[j] - start_color[j]) * i / (n_bars - 1) 
                for j in range(3)
            ]
            gradient.append(tuple(interpolated_color))
        return gradient

    # Calculate FPS based on the duration of the animation
    fps = len(frames) / duration

    # Generate the gradient colors
    start_color = np.array([random.random(), random.random(), random.random()])
    end_color = np.array([random.random(), random.random(), random.random()])
    gradient = generate_gradient_color(start_color, end_color, n)

        # Update function for the sorting animation
    def update(frame_index):
        if frame_index < len(frames):
            frame_data, current_index_1, current_index_2 = frames[frame_index]
            for rect in bars:
                rect.set_color('white')  # Reset all bars to white
            bars[current_index_1].set_color('red')  # Highlight the current bar
              
            for rect, height in zip(bars, frame_data):
                rect.set_height(height)  # Update heights
        elif frame_index < len(frames) + n:
            bar_index = frame_index -   len(frames)
            bars[bar_index].set_color(gradient[bar_index])  # Apply gradient
        
        return bars  # Return the list of bars to update the animation

    # Wrap the frames with tqdm for progress tracking
    frame_count = len(frames) + n
    with tqdm(total=frame_count, desc="Rendering Animation") as pbar:
        def wrapped_update(frame_index):
            pbar.update(1)  # Update the progress bar
            return update(frame_index)

        # Create the sorting animation
        ani = FuncAnimation(fig, wrapped_update, frames=frame_count, repeat=False, blit=True, interval=1)
        ani.save('sorting_animation.mp4', fps=fps, dpi=100, codec='libx264')

# Example Usage:
create_sorting_animation(n=150, duration=5)
