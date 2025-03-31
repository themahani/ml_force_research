import sys
from pathlib import Path

# Add the parent directory to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "ml-force"))

from typing import Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from ml_force.models import MorrisLecar, MorrisLecarCurrent
from ml_force.models import minmax_transform
from scipy.signal import convolve
from tqdm import tqdm


def smooth(
    signal: np.ndarray,
    window_len: int = 5,
    window: Literal["hanning", "hamming", "bartlett", "blackman"] = "flat",
    mode: Literal["full", "valid", "same"] = "same",
    axis: int = 0,
    method: Literal["auto", "direct", "fft"] = "auto",
) -> np.ndarray:
    """Smooth the data using a window with requested size.

    Parameters
    ----------
    signal : np.ndarray
        The input signal to smooth, must be 2D
    window_len : int, optional
        The dimension of the smoothing window, by default 5
    window : {'flat', 'hanning', 'hamming', 'bartlett', 'blackman'}, optional
        The type of window function to use, by default 'flat'
    mode : {'full', 'valid', 'same'}, optional
        The convolution mode, by default 'same'
    axis : {0, 1}, optional
        The axis along which to perform smoothing, by default 0
    method : {'auto', 'direct', 'fft'}, optional
        The convolution method, by default 'auto'

    Returns
    -------
    np.ndarray
        Smoothed signal

    Raises
    ------
    ValueError
        If signal is not 2D, window_len < 3, signal shorter than window,
        or axis not in {0, 1}
    """
    if signal.ndim != 2:
        raise ValueError("Signal must be 2D")
    if window_len < 3:
        return signal
    if signal.shape[axis] < window_len:
        raise ValueError("Input signal must be longer than window size")
    if window == "flat":
        kernel = np.ones(window_len) / window_len
    else:
        kernel = eval(f"np.{window}(window_len)")
    if axis == 0:
        kernel_nd = kernel[:, np.newaxis]
    elif axis == 1:
        kernel_nd = kernel[np.newaxis, :]
    else:
        raise ValueError("Axis must be 0 or 1")
    return convolve(signal, kernel_nd / kernel_nd.sum(), mode=mode, method=method)


def add_noise_to_image(image: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    """Add noise to an image by flipping pixel values.

    Parameters
    ----------
    image : np.ndarray
        Input image array
    noise_level : float, optional
        Fraction of pixels to flip, by default 0.1

    Returns
    -------
    np.ndarray
        Noisy version of input image
    """
    n_pixels = int(noise_level * image.size)
    noise_pixels = np.random.choice(np.arange(image.size), size=n_pixels, replace=False)
    new_image = image.copy().ravel()
    new_image[noise_pixels] *= -1
    return new_image.reshape(image.shape)


def generate_signals(
    images: np.ndarray,
    corrupted_images: np.ndarray,
    height: int,
    width: int,
    dt: float = 0.05,
    n_exposures: int = 30,
    t_transient: float = 1000,
    nt_min: int = 2000,
    nt_max: int = 3000,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Generate signal and noisy signal sequences for memory task.

    Parameters
    ----------
    images : np.ndarray
        Array of original images
    corrupted_images : np.ndarray
        Array of corrupted/noisy images
    height : int
        Height of the images
    width : int
        Width of the images
    dt : float, optional
        Time step in milliseconds, by default 0.05
    n_exposures : int, optional
        Number of images to show, by default 30
    t_transient : float, optional
        Transient time in milliseconds between images, by default 1000
    nt_min : int, optional
        Minimum number of time steps to show an image, by default 2000
    nt_max : int, optional
        Maximum number of time steps to show an image, by default 3000

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[int], list[int]]
        signal: Array of original signal sequences
        noisy_signal: Array of noisy signal sequences
        exposures: List of image indices shown
        time_stamps: List of time points when images were shown
    """
    nt_min = nt_min // dt
    nt_max = nt_max // dt
    nt_transient = int(t_transient // dt)
    n_images = len(images)

    signal = []
    noisy_signal = []
    exposures = []
    noisy_images = []
    time_stamps = []

    for i in range(n_exposures):
        signal += [np.zeros(height * width) for _ in range(nt_transient)]
        noisy_signal += [np.zeros(height * width) for _ in range(nt_transient)]
        nt = np.random.randint(nt_min, nt_max)

        image_index = np.random.randint(0, n_images)
        exposures.append(image_index)
        image_raveled = images[image_index].ravel()
        noisy_image_raveled = corrupted_images[image_index].ravel()
        noisy_images.append(noisy_image_raveled)

        signal += [image_raveled for _ in range(nt)]
        noisy_signal += [noisy_image_raveled for _ in range(nt)]
        time_stamps.append(len(signal) - int(1000 // dt))

    return np.array(signal), np.array(noisy_signal), exposures, time_stamps


def generate_images(
    n_images: int,
    height: int,
    width: int,
    noise_level: float = 0.1,
    multiplier: float = 1.0,
    n_try: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random binary images and their noisy versions.

    Parameters
    ----------
    n_images : int
        Number of images to generate
    height : int
        Height of the images
    width : int
        Width of the images
    noise_level : float, optional
        Fraction of pixels to be flipped, by default 0.1
    multiplier : float, optional
        Scale factor for the image values, by default 1.0
    n_try : int, optional
        Number of attempts to generate unique images, by default 0

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        images: Array of original images
        noisy_images: Array of noisy versions of the images

    Raises
    ------
    ValueError
        If too many attempts to generate unique images or invalid parameters
    """
    if n_try > 10:
        raise ValueError("Too many attempts to generate unique images.")
    if n_images < 1:
        raise ValueError("Number of images must be at least 1.")
    if height < 1 or width < 1:
        raise ValueError("Height and width must be at least 1.")
    if noise_level < 0 or noise_level > 1:
        raise ValueError("Noise level must be between 0 and 1.")
    if multiplier <= 0:
        raise ValueError("Multiplier must be positive.")
    if n_images > 2 ** (height * width):
        raise ValueError("Number of images exceeds the maximum unique combinations.")

    images = (
        np.array(
            [
                np.random.choice([-1, 1], size=(height, width), replace=True)
                for _ in range(n_images)
            ]
        )
        * multiplier
    )
    noisy_images = np.array(
        [add_noise_to_image(image, noise_level) for image in images]
    )
    if test_repetition(images, noisy_images):
        print("Warning: Generated images contain repetitions, retrying...")
        return generate_images(
            n_images, height, width, noise_level, multiplier, n_try + 1
        )

    return images, noisy_images


def save_reservoir_state(
    ml: MorrisLecarCurrent, save_dir: Union[str, Path], save_prefix: str = None
) -> None:
    """Save the reservoir state variables.

    Parameters
    ----------
    ml : MorrisLecarCurrent
        Morris-Lecar network instance
    save_dir : Union[str, Path]
        Directory to save the states
    save_prefix : str, optional
        Prefix for the save directory name, by default None
    """
    if save_prefix is None:
        save_prefix = ""

    save_dir = Path(save_dir) / f"reservoir_states{save_prefix}"
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / "v.npy", ml.v.cpu().numpy())
    np.save(save_dir / "n.npy", ml.n.cpu().numpy())
    np.save(save_dir / "s.npy", ml.s.cpu().numpy())
    np.save(save_dir / "eta.npy", ml.eta.cpu().numpy())
    np.save(save_dir / "dec.npy", ml.dec.cpu().numpy())


def train_network(
    ml: MorrisLecarCurrent,
    noisy_signal_smoothed: np.ndarray,
    signal_smoothed: np.ndarray,
    I_bias: float = 70,
    dt: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train the Morris-Lecar network.

    Parameters
    ----------
    ml : MorrisLecarCurrent
        Morris-Lecar network instance
    noisy_signal_smoothed : np.ndarray
        Smoothed noisy input signal
    signal_smoothed : np.ndarray
        Smoothed target signal
    I_bias : float, optional
        Bias current, by default 70
    dt : float, optional
        Time step in milliseconds, by default 0.05

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        s_rec: Synaptic gating variable
        n_rec: Recovery variable
        v_rec: Membrane voltage
    """
    nt = signal_smoothed.shape[0]
    device = ml.device

    # Initial transient period
    transient_time = 500
    nt_transient = int(transient_time // dt)
    noisy_sup_tensor = torch.tensor(
        noisy_signal_smoothed, dtype=torch.float32, device=device
    )

    s_rec = torch.zeros((ml._N, 1), dtype=torch.float32, device=device)
    n_rec = torch.zeros((ml._N, 1), dtype=torch.float32, device=device)
    v_rec = torch.zeros((ml._N, 1), dtype=torch.float32, device=device)

    for _ in tqdm(range(nt_transient)):
        ml.euler_step(closed_loop=True)

    for i in tqdm(range(nt)):
        ml._BIAS = I_bias + ml.eta @ noisy_sup_tensor[i].reshape(-1, 1)
        ml.euler_step(closed_loop=True, voltage_bound=None)
        ml.x_hat_rec[i] = ml.x_hat.ravel()

        if i % 20 == 1:
            ml.rls(i)

    return ml.s.clone().detach(), ml.n.clone().detach(), ml.v.clone().detach()


def test_network(
    ml: MorrisLecarCurrent,
    corrupted_test_images: np.ndarray,
    test_signal_smoothed: torch.Tensor,
    I_bias: float = 70,
) -> torch.Tensor:
    """Test the trained network on multiple images.

    Parameters
    ----------
    ml : MorrisLecarCurrent
        Trained Morris-Lecar network
    corrupted_test_images : np.ndarray
        Set of corrupted test images
    test_signal_smoothed : torch.Tensor
        Smoothed test signal sequence
    I_bias : float, optional
        Bias current, by default 70

    Returns
    -------
    torch.Tensor
        Network output for test sequence
    """
    device = ml.device
    nt_test = test_signal_smoothed.shape[0]
    output = torch.zeros((nt_test, test_signal_smoothed.shape[1]), device=device)

    for j in tqdm(range(nt_test)):
        ml._BIAS = I_bias + ml.eta @ test_signal_smoothed[j].reshape(-1, 1)
        ml.euler_step(closed_loop=True)
        output[j] = ml.x_hat.ravel()

    return output


def plot_test_results(
    output,
    corrupted_test_images,
    test_images,
    test_exposures,
    exposures,
    height,
    width,
    save_dir,
    dt=0.05,
):
    """Generate and save visualization plots for test results."""
    nt = int(1000 // dt)
    renders = []

    for i in range(len(test_exposures)):
        model_render = (
            output[(i + 1) * nt - 2000 : (i + 1) * nt].cpu().numpy().mean(axis=0)
        )
        renders.append(model_render)

    losses = [
        rmse(test_images[i], renders[i].reshape(-1, 1)) for i in range(len(renders))
    ]

    vmin = min(corrupted_test_images.min(), test_images.min(), np.min(renders))
    vmax = max(corrupted_test_images.max(), test_images.max(), np.max(renders))

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(renders)):
        fig, ax = plt.subplots(figsize=(15, 5), ncols=3)
        clr = ax[0].imshow(
            unravel_image(corrupted_test_images[i], height, width), vmin=vmin, vmax=vmax
        )
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title("Corrupted Image")
        ax[1].imshow(unravel_image(test_images[i], height, width), vmin=vmin, vmax=vmax)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title("Target Image")
        ax[2].imshow(
            unravel_image(renders[i].reshape(-1, 1), height, width),
            vmin=vmin,
            vmax=vmax,
        )
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        ax[2].set_title("Model Render")
        plt.colorbar(clr, ax=ax[2])
        plt.suptitle(
            f"Test: {i}, Exposure: {test_exposures[i]}, Image {exposures[test_exposures[i]]}, Loss: {losses[i]:.4f}",
            y=0.95,
            fontsize=16,
        )
        plt.savefig(
            save_dir / f"image_comparison_{i}.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

    return np.mean(losses)


def unravel_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Reshape a flattened image back to 2D."""
    return image.ravel().reshape(height, width)


def test_repetition(images: np.ndarray, corrupted_images: np.ndarray) -> bool:
    """Check if the same image appears multiple times in the collection."""
    unique_images = set()
    for img in images:
        img_tuple = tuple(img.ravel())
        if img_tuple in unique_images:
            return True
        unique_images.add(img_tuple)
    for img in corrupted_images:
        img_tuple = tuple(img.ravel())
        if img_tuple in unique_images:
            return True
        unique_images.add(img_tuple)
    # No repetitions found
    return False


def rmse(output: np.ndarray, target: np.ndarray) -> float:
    """Calculate Root Mean Square Error."""
    return np.sqrt(np.mean((output - target) ** 2))


def main() -> None:
    """Execute the memory task training and testing workflow.

    This function performs the following steps:
    1. Generate training and testing images
    2. Create signal sequences
    3. Train Morris-Lecar network
    4. Test network performance
    5. Save results and visualizations
    """
    # Set random seeds
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # Setup parameters
    save_dir = Path(__file__).parent / "memory_task_with_training" / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Image generation parameters
    n_images = 5
    width, height = 5, 5
    noise_level = 0.05
    multiplier = 1.0

    # Network parameters
    N = 500
    dt = 0.05
    T = None  # Will be set after signal generation
    I_bias = 70
    Q = 50
    gbar = 15
    lamda = 0.8
    p_sparsity = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate images
    images, corrupted_images = generate_images(
        n_images, height, width, noise_level, multiplier
    )

    # Generate training signals
    signal, noisy_signal, exposures, time_stamps = generate_signals(
        images, corrupted_images, height, width
    )

    # Set total time T based on signal length
    T = signal.shape[0] * dt

    # Setup smoothing parameters
    window_len = 300 // dt
    smoothing_params = {
        "window_len": window_len,
        "window": "hanning",
        "mode": "same",
        "method": "fft",
        "axis": 0,
    }
    signal_smoothed = smooth(signal, **smoothing_params)
    noisy_signal_smoothed = smooth(noisy_signal, **smoothing_params)

    # Initialize Morris-Lecar network
    ml = MorrisLecarCurrent(
        supervisor=signal_smoothed,
        N=N,
        T=T,
        dt=dt,
        BIAS=I_bias,
        Q=Q,
        gbar=gbar,
        p_sparsity=p_sparsity,
        l=lamda,
        device=device,
    )

    # Train network
    s_rec, n_rec, v_rec = train_network(
        ml, noisy_signal_smoothed, signal_smoothed, I_bias, dt
    )
    save_reservoir_state(ml, save_dir, f"_N{N}_Q{Q}_gbar{gbar}_l{lamda}_p{p_sparsity}")

    # Generate test data
    n_tasks = 20
    duration = 1000
    nt = int(duration // dt)
    test_exposures = np.random.choice(exposures, size=n_tasks, replace=True)
    test_images = np.array(
        [images[img_id].ravel().reshape(-1, 1) for img_id in test_exposures]
    )
    corrupted_test_images = np.array(
        [corrupted_images[img_id].ravel().reshape(-1, 1) for img_id in test_exposures]
    )

    # Generate test signals
    test_signal = []
    corrupted_test_signal = []
    nt_transient = int(1000 // dt)

    for image_id in test_exposures:
        test_signal += [np.zeros(height * width) for _ in range(nt_transient)]
        corrupted_test_signal += [np.zeros(height * width) for _ in range(nt_transient)]
        test_signal += [images[image_id].ravel() for _ in range(nt)]
        corrupted_test_signal += [corrupted_images[image_id].ravel() for _ in range(nt)]

    test_signal = np.array(test_signal)
    corrupted_test_signal = np.array(corrupted_test_signal)

    # Smooth test signals
    test_signal_smoothed = torch.tensor(
        smooth(test_signal, **smoothing_params), dtype=torch.float32, device=device
    )
    corrupted_test_signal_smoothed = torch.tensor(
        smooth(corrupted_test_signal, **smoothing_params),
        dtype=torch.float32,
        device=device,
    )

    # Test network
    output = test_network(
        ml, corrupted_test_images, corrupted_test_signal_smoothed, I_bias
    )

    # Plot and save results
    mean_loss = plot_test_results(
        output,
        corrupted_test_images,
        test_images,
        test_exposures,
        exposures,
        height,
        width,
        save_dir,
        dt,
    )
    print(f"Average Test Loss: {mean_loss:.4f}")


if __name__ == "__main__":
    main()
