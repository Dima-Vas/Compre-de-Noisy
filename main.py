import numpy as np
import cv2
from scipy.linalg import svd
import matplotlib.pyplot as plt


def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255.0


def power_iteration(A, num_iterations=100, epsilon=1e-8):
    n, m = A.shape
    x = np.random.rand(m)
    for _ in range(num_iterations):
        x = np.dot(A, x)
        x_norm = np.linalg.norm(x) + epsilon
        x /= x_norm
    return x_norm - epsilon, x


def svd_decomposition(img_matrix, num_iterations=100, epsilon=1e-8):
    A = img_matrix
    ATA = np.dot(A.T, A)
    n, m = A.shape
    U = np.zeros((n, m))
    S = np.zeros(m)
    Vt = np.zeros((m, m))

    for i in range(m):
        s, v = power_iteration(ATA, num_iterations, epsilon)
        S[i] = np.sqrt(s)
        Vt[i] = v
        ATA = ATA - s * np.outer(v, v)
        u = np.dot(A, v) / (S[i] + epsilon)
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


def main(image_path, compressed_output_path, denoised_output_path, rank, denoising_threshold):
    # Load and preprocess the image
    img = load_image(image_path)

    # Perform SVD decomposition
    U, s, Vt = svd_decomposition(img)

    # Compress the image
    U_compressed, s_compressed, Vt_compressed = compress_image(U, s, Vt, rank)

    # Denoise the image
    U_denoised, s_denoised, Vt_denoised = denoise_image(U, s, Vt, denoising_threshold)

    # Reconstruct the images
    img_compressed = reconstruct_image(U_compressed, s_compressed, Vt_compressed)
    img_denoised = reconstruct_image(U_denoised, s_denoised, Vt_denoised)

    # Save the compressed and denoised images
    save_image(img_compressed, compressed_output_path)
    save_image(img_denoised, denoised_output_path)

    # Visualize the original, compressed, and denoised images
    visualize_images(img, img_compressed, img_denoised)


if __name__ == '__main__':
    image_path = '/Users/oleksijkonopada/Desktop/UCU/Linear_Algebra/Linear_project/golden-retriever-royalty-free-image-506756303-1560962726.jpeg'
    compressed_output_path = '/Users/oleksijkonopada/Desktop/UCU/Linear_Algebra/Linear_project/compressed.jpeg'
    denoised_output_path = '/Users/oleksijkonopada/Desktop/UCU/Linear_Algebra/Linear_project/denoised.jpeg'
    rank = 10
    denoising_threshold = 0.3
    main(image_path, compressed_output_path, denoised_output_path, rank, denoising_threshold)
