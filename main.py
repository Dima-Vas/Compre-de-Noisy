import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0


def power_iteration(A, num_iterations=1000, epsilon=1e-8):
    n, d = A.shape

    v = np.random.rand(d)
    v = v / np.linalg.norm(v)

    for _ in range(num_iterations):
        Av = np.dot(A, v)
        v_new = Av / np.linalg.norm(Av)
        if np.abs(np.dot(v, v_new)) > 1 - epsilon:
            break

        v = v_new

    return np.dot(Av, v), v


def svd_decomposition(img_matrix, num_iterations=50, epsilon=1e-8):
    A = img_matrix
    ATA = np.dot(A.T, A)
    n, m = A.shape
    U = np.zeros((n, m))
    S = np.zeros(m)
    Vt = np.zeros((m, m))

    for i in range(m):
        s, v = power_iteration(ATA, num_iterations, epsilon)
        if s < 0:
            break
        S[i] = np.sqrt(s)
        Vt[i] = v
        ATA = ATA - s * np.outer(v, v)
        if S[i] < epsilon:
            break
        u = np.dot(A, v) / S[i]
        U[:, i] = u
        A = A - np.outer(u, np.dot(S[i], v))

    return U, S, Vt


def compress_image(U, s, Vt, rank):
    s_compressed = s[:rank]
    U_compressed = U[:, :rank]
    Vt_compressed = Vt[:rank, :]
    return U_compressed, s_compressed, Vt_compressed


def denoise_image(U, s, Vt, threshold):
    s_denoised = np.where(s > threshold, s, 0)
    return U, s_denoised, Vt

def iterative_denoise(image, initial_threshold, threshold_step, max_iterations, target_mse):
    U, s, Vt = svd_decomposition(image)

    threshold = initial_threshold
    for iteration in range(max_iterations):
        U_denoised, s_denoised, Vt_denoised = denoise_image(U, s, Vt, threshold)
        denoised_image = reconstruct_image(U_denoised, s_denoised, Vt_denoised)

        mse = mean_squared_error(image.flatten(), denoised_image.flatten())
        print(f"Iteration {iteration + 1}, MSE: {mse}, Threshold: {threshold}")

        if mse <= target_mse:
            print("Target MSE reached. Stopping.")
            break

        threshold += threshold_step

    return denoised_image, threshold

def reconstruct_image(U, s, Vt):
    return np.dot(U, np.dot(np.diag(s), Vt))

def save_image(img_matrix, output_path):
    img_matrix = np.clip(img_matrix, 0, 1) * 255
    cv2.imwrite(output_path, img_matrix)


def visualize_images(original, compressed, denoised):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(original, cmap='gray')
    ax[0].set_title('Original Image')
    ax[1].imshow(compressed, cmap='gray')
    ax[1].set_title('Compressed Image')
    ax[2].imshow(denoised, cmap='gray')
    ax[2].set_title('Denoised Image')
    plt.show()


if __name__ == "__main__":

    image_path = "/Users/oleksijkonopada/Desktop/UCU/Linear_Algebra/Linear_project/672353.fig.007b.jpg"
    compressed_output_path = "/Users/oleksijkonopada/Desktop/UCU/Linear_Algebra/Linear_project/compressed.jpg"
    denoised_output_path = "/Users/oleksijkonopada/Desktop/UCU/Linear_Algebra/Linear_project/denoised.jpg"

    img = load_image(image_path)

    U, s, Vt = svd_decomposition(img)
    print(s)
    rank = round(len(s) * 0.02)
    print(rank)
    U_compressed, s_compressed, Vt_compressed = compress_image(U, s, Vt, rank)

    initial_threshold = 1.0
    threshold_step = 0.5
    max_iterations = 20
    target_mse = 0.01

    img_denoised, final_threshold = iterative_denoise(img, initial_threshold, threshold_step, max_iterations, target_mse)

    img_compressed = reconstruct_image(U_compressed, s_compressed, Vt_compressed)

    save_image(img_compressed, compressed_output_path)
    save_image(img_denoised, denoised_output_path)

    visualize_images(img, img_compressed, img_denoised)
