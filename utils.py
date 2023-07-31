import os
from collections import defaultdict
import pandas as pd
from PIL import Image as im
from PIL import ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine


CROP_BOXES = {
    "box01": (100, 20, 270, 230),
    "box02": (110, 20, 280, 230),
    "box03": (95, 25, 265, 235),
    "box04": (95, 15, 265, 225),
    "box05": (105, 15, 275, 225),
    "box06": (60, 15, 230, 225),
    "box07": (80, 20, 250, 230),
    "box08": (80, 20, 250, 230),
    "box09": (80, 20, 250, 230),
    "box10": (100, 20, 270, 230),
    "box11": (100, 20, 270, 230),
    "box12": (110, 20, 280, 230),
    "box13": (85, 30, 255, 240),
    "box14": (50, 20, 220, 230),
    "box15": (80, 20, 250, 230),
}

LEFT_PADDINGS = {
    "01": [1 / 4, 3 / 4],
    "02": [1 / 4, 7 / 8],
    "03": [1 / 8, 3 / 4],
    "04": [1 / 4, 3 / 4],
    "05": [1 / 4, 3 / 4],
    "06": [0, 5 / 8],
    "07": [1 / 4, 3 / 4],
    "08": [1 / 4, 3 / 4],
    "09": [1 / 8, 7 / 8],
    "10": [1 / 4, 3 / 4],
    "11": [1 / 4, 3 / 4],
    "12": [1 / 4, 3 / 4],
    "13": [1 / 4, 7 / 8],
    "14": [1 / 4, 3 / 4],
    "15": [1 / 4, 7 / 8],
}


def get_im_from_arr(arr: np.array, m=210, n=170) -> np.array:
    """Create image from array that can be viewed using `im.fromarray(arr)`"""
    return arr.astype(np.uint8).reshape(m, n)
    # Then run im.fromarray(arr)


def normalize_data(arr: np.array, max_=1) -> np.array:
    """Normalize values in array between 0 and max"""
    normal = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return normal * max_


def plot_top_6_eigenfaces(Z: np.array, subject: int, m=210, n=170) -> None:
    """Plots the top 6 eigenfaces assuming they are each row vectors"""
    _, axarr = plt.subplots(2, 3, figsize=(8, 8))
    i = 0
    j = 0
    for ix in range(6):
        pic = get_im_from_arr(Z[ix, :], m=m, n=n)
        axarr[i, j].imshow(pic, cmap="Greys_r")
        axarr[i, j].set_title(f"Eigenface {ix+1} | Subject {subject}")
        axarr[i, j].axis("off")
        if (j + 1) % 3 == 0:
            i += 1
            j = 0
        else:
            j += 1
    plt.show()


def turn_images_to_pngs(data_dir: str) -> None:
    """Add PNG extension to images if it doesn't already exist"""
    for f in os.listdir(data_dir):
        if "subject" in f and ".png" not in f:  # Don't add .png to PNG files
            full_path = os.path.join(data_dir, f)
            renamed = full_path + ".png"
            os.rename(full_path, renamed)


def crop_faces(data_dir: str) -> None:
    """Center-crops each face for consistency"""
    for f in os.listdir(data_dir):
        img_path = os.path.join(data_dir, f)
        img = im.open(img_path)
        # Get crop box coords
        subject_num = "".join([el for el in f if el.isnumeric()])
        key = "box" + subject_num
        box = CROP_BOXES[key]
        cropped = img.crop(box)
        # Save images
        data_dir_par, _ = os.path.split(data_dir)
        save_dir = os.path.join(data_dir_par, "cropped")
        save_path = os.path.join(save_dir, f)
        cropped.save(save_path)


def get_train_faces(data_dir: str, m=210, n=170) -> np.array:
    """Gather all images from the data dir EXCEPT the "centerlight" and "normal" ones. Those will be used for calculating a weight vector and testing"""
    train_imgs = [
        f for f in os.listdir(data_dir) if "centerlight" not in f and "normal" not in f
    ]
    tot_imgs = len(train_imgs)
    arr = np.zeros(
        (tot_imgs, m * n)
    )  # Treat each face as a data point - i.e. a single row
    for ix, f in enumerate(train_imgs):
        img_path = os.path.join(data_dir, f)
        img = im.open(img_path)
        img_arr = flatten_image(np.asarray(img))
        arr[ix, :] = img_arr.reshape(1, -1)
    return arr


def get_weight_faces(data_dir: str, n_subjects=15, m=210, n=170) -> np.array:
    """Generate array with vectorized "normal" faces for each subject"""
    normal_faces = np.zeros((n_subjects, m * n))
    i = 0
    for f in os.listdir(data_dir):
        if "normal" in f:
            img_path = os.path.join(data_dir, f)
            img = im.open(img_path)
            img_arr = flatten_image(np.asarray(img))
            normal_faces[i, :] = img_arr.reshape(1, -1)
            i += 1
    return normal_faces


def get_test_faces(data_dir: str, n_subjects=15, m=210, n=170) -> np.array:
    """Generate array with vectorized "centerlight" faces for each subject"""
    center_faces = np.zeros((n_subjects, m * n))
    i = 0
    for f in os.listdir(data_dir):
        if "centerlight" in f:
            img_path = os.path.join(data_dir, f)
            img = im.open(img_path)
            img_arr = flatten_image(np.asarray(img))
            center_faces[i, :] = img_arr.reshape(1, -1)
            i += 1
    return center_faces


def flatten_image(arr: np.array) -> np.array:
    """Flatten a 3D image into a 2D image. I'm not sure why, but we end up with a m x n x 3 array after drawing face masks. Each pixel from the original 2D image is represented by a 3x3 array in the 3D image, where all the numbers are the same in the 3x3 array"""
    if len(arr.shape) == 3:
        flattened = np.mean(
            arr, axis=2
        )  # NOTE: np.mean, np.max, np.min all work fine to flatten
        return flattened
    else:
        return arr


def get_top_n_eigenfaces(training_faces: np.array, n: int) -> np.array:
    """Get top n eigenfaces for a subject"""
    # Average of subject 1 faces
    avg_face = np.mean(training_faces, axis=0)
    # Subtract avg face from all training faces
    centered_faces = training_faces - avg_face
    # Find cov matrix
    C = centered_faces @ centered_faces.T
    # Get top n eigenfaces
    PCs = PCA(n_components=n).fit_transform(C)
    top_n_efaces = PCs.T @ centered_faces
    return top_n_efaces


def get_truth_weights(
    train_arr: np.array,
    weight_arr: np.array,
    top_efaces: np.array,
    faces_per_subject=9,
    n_subjects=15,
    n_efaces=6,
) -> np.array:
    """Use a training image from each subject to get the weights of each eigenface that create that training face"""
    truth_weights = np.zeros((n_subjects, n_efaces))
    for i in range(n_subjects):
        subject = i + 1
        training_faces = train_arr[i * faces_per_subject : (i + 1) * faces_per_subject]
        avg_face = np.mean(training_faces, axis=0)
        truth_weights[i] = normalize_data(
            top_efaces[subject] @ (weight_arr[i] - avg_face)
        )
    return truth_weights


def calc_dissimilarities(
    top_eigenfaces: dict, test_faces: np.array, eface: int = 1
) -> dict:
    """Calculate and store projection residual, L2 norm and cosine dissimilarity metrics for all test face/top eigenface combos. Optionally specify which "top" eigenface to use"""
    n_subjects = test_faces.shape[0]
    dissimilarities = {}
    for k, v in top_eigenfaces.items():
        top_eface = normalize_data(v[eface - 1].reshape(1, -1))
        sims = defaultdict(list)
        for i in range(n_subjects):
            to_compare = normalize_data(test_faces[i, :].reshape(1, -1))
            proj_res = np.linalg.norm(to_compare - top_eface @ top_eface.T @ to_compare)
            L2 = np.linalg.norm(to_compare - top_eface)
            cos_sim = cosine(to_compare.flatten(), top_eface.flatten())
            sims["proj_ress"].append(proj_res)
            sims["L2s"].append(L2)
            sims["coss"].append(cos_sim)
        dissimilarities[k] = sims
    return dissimilarities


def get_avg_faces(train_arr: np.array, faces_per_subject: int) -> np.array:
    m, n = train_arr.shape
    n_subjects = int(m / faces_per_subject)
    avg_faces = np.zeros((n_subjects, n))
    for i in range(n_subjects):
        subset = train_arr[i * faces_per_subject : (i + 1) * faces_per_subject]
        avg_face = np.mean(subset, axis=0)
        avg_faces[i] = avg_face
    return avg_faces


def calc_dissimilarities_v2(
    top_eigenfaces: dict,
    test_faces: np.array,
    truth_weights: np.array,
    avg_faces: np.array,
) -> dict:
    """Calculate and store L2 norm for weights of test face projected onto each eigenface space vs the 'truth' weight for each face class"""
    n_subjects = test_faces.shape[0]
    dissimilarities = {}
    for subject, _ in top_eigenfaces.items():
        w_truth = truth_weights[subject - 1]
        sims = []
        for i in range(n_subjects):
            to_compare = test_faces[i].reshape(-1, 1)
            avg_face = avg_faces[i].reshape(-1, 1)
            wt = normalize_data(top_eigenfaces[i + 1] @ (to_compare - avg_face))
            l2_dist = np.linalg.norm(w_truth - wt)
            sims.append(l2_dist)
        dissimilarities[subject] = sims
    return dissimilarities


def clf_accs_to_df(dissimilarities: dict) -> pd.DataFrame:
    """Create df containing classification accuracy using various facial recognition dissimilarity metrics"""
    cols = [
        "subject",
        "proj_residual_acc",
        "L2_acc",
        "cos_sim_acc",
        "max_acc",
        "avg_acc",
        "median_acc",
    ]
    df = pd.DataFrame(columns=cols)
    for ix, (k, v) in enumerate(dissimilarities.items()):
        proj_res = round(np.mean(v["proj_ress"][ix] <= v["proj_ress"]), 2)
        L2 = round(np.mean(v["L2s"][ix] <= v["L2s"]), 2)
        cos = round(np.mean(v["coss"][ix] >= v["coss"]), 2)
        max_acc = max(proj_res, L2, cos)
        avg_acc = round(np.mean([proj_res, L2, cos]), 2)
        median_acc = np.median([proj_res, L2, cos])
        row = pd.DataFrame(
            [[k, proj_res, L2, cos, max_acc, avg_acc, median_acc]], columns=cols
        )
        df = pd.concat([df, row])
    return df


def dissims_to_df(dissimilarities: dict) -> pd.DataFrame:
    """Create df containing classification accuracy using L2 distance for each face class"""
    cols = ["subject", "accuracy"]
    df = pd.DataFrame(columns=cols)
    for ix, (k, v) in enumerate(dissimilarities.items()):
        acc = round(np.mean(v[ix] <= v), 2)
        row = pd.DataFrame([[k, acc]], columns=cols)
        df = pd.concat([df, row])
    return df


def add_full_facemask(image_path: str) -> None:
    """Draw a black box over the faces to emulate a face mask. It should start at the bottom of the nose and end near the bottom of the chin."""
    # Load image
    img = im.open(image_path)
    img = img.convert("RGBA")

    # Specify face mask dimensions
    n, m = img.size  # img.size returns width x height
    w_frac = 4 / 5  # mask width as fraction of image width
    h_frac = 1 / 4  # mask height as fraction of image height
    w, h = w_frac * n, h_frac * m

    # Center the mask horizontally on faces
    left_pad = (n - w) / 2
    right_pad = (n - w) / 2
    # 5x as much space from top of mask to top of image as bottom of mask to bottom of image
    top_pad = (5 / 6) * (m - h)
    bottom_pad = (1 / 6) * (m - h)

    # Build face mask
    overlay = im.new("RGBA", img.size)
    shape = [(left_pad, top_pad), (n - right_pad, m - bottom_pad)]
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(shape, fill="black")
    img = im.alpha_composite(img, overlay)
    img = img.convert("RGB")

    # Save face-masked image
    cwd = os.getcwd()
    full_mask_dir = os.path.join(cwd, "full_mask_faces")
    image_fname = os.path.split(image_path)[-1]
    save_path = os.path.join(full_mask_dir, image_fname)
    img.save(save_path)


def add_partial_facemask(image_path: str) -> None:
    """Draw a black box over half of each face to emulate a partial face mask. It should start at the bottom of the nose and end near the bottom of the chin."""
    # Load image
    img = im.open(image_path)
    img = img.convert("RGBA")

    # Specify face mask dimensions
    n, m = img.size  # img.size returns width x height
    w_frac = 2 / 5  # mask width as fraction of image width
    h_frac = 1 / 4  # mask height as fraction of image height
    w, h = w_frac * n, h_frac * m

    # Horizontal-align masks to cover half of face. Also handle certain faces differently than others
    which_half = np.random.randint(2)  # Cover left or right side of face randomly
    _, fname = os.path.split(image_path)
    subject_num = "".join([el for el in fname if el.isnumeric()])
    left_pad_frac = LEFT_PADDINGS[subject_num][which_half]
    right_pad_frac = 1 - left_pad_frac
    left_pad = (n - w) * left_pad_frac
    right_pad = (n - w) * right_pad_frac

    # 5x as much space from top of mask to top of image as bottom of mask to bottom of image
    top_pad = (5 / 6) * (m - h)
    bottom_pad = (1 / 6) * (m - h)

    # Build face mask
    overlay = im.new("RGBA", img.size)
    shape = [(left_pad, top_pad), (n - right_pad, m - bottom_pad)]
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(shape, fill="black")
    img = im.alpha_composite(img, overlay)
    img = img.convert("RGB")

    # Save face-masked image
    cwd = os.getcwd()
    full_mask_dir = os.path.join(cwd, "half_mask_faces")
    image_fname = os.path.split(image_path)[-1]
    save_path = os.path.join(full_mask_dir, image_fname)
    img.save(save_path)


def add_sunglasses(image_path: str) -> None:
    """Draw a black box over the eyes to emulate sunglasses. It should start near the top of the forehead and end near he top of the nose."""
    # Load image
    img = im.open(image_path)
    img = img.convert("RGBA")

    # Specify face mask dimensions
    n, m = img.size  # img.size returns width x height
    w_frac = 4 / 5  # sunglasses width as fraction of image width
    h_frac = 1 / 4  # sunglasses height as fraction of image height
    w, h = w_frac * n, h_frac * m

    # Center the sunglasses horizontally on faces
    left_pad = (n - w) / 2
    right_pad = (n - w) / 2
    # Handle certain subjects differently
    _, image_fname = os.path.split(image_path)
    to_adjust = ["03", "06", "07", "08", "12", "13", "15"]
    subject_num = "".join([el for el in image_fname if el.isnumeric()])
    if subject_num in to_adjust:
        top_pad = (1 / 2) * (m - h)
        bottom_pad = (1 / 2) * (m - h)
    else:
        top_pad = (2 / 5) * (m - h)
        bottom_pad = (3 / 5) * (m - h)
    if subject_num == "06":
        left_pad = 0
        right_pad = n - w

    # Build face mask
    overlay = im.new("RGBA", img.size)
    shape = [(left_pad, top_pad), (n - right_pad, m - bottom_pad)]
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(shape, fill="black")
    img = im.alpha_composite(img, overlay)
    img = img.convert("RGB")

    # Save face-masked image
    cwd = os.getcwd()
    full_mask_dir = os.path.join(cwd, "sunglasses_faces")
    save_path = os.path.join(full_mask_dir, image_fname)
    img.save(save_path)


def cover_pct_of_face(image_path: str, frac: float) -> None:
    """Draw a black circle centered in the subjects' faces covering frac * 100 % of their face"""
    img = im.open(image_path)
    img = img.convert("RGBA")

    # Specify face covering dimensions
    n, m = img.size  # img.size returns width x height
    r_max = (
        2 / 5
    ) * n  # circle w diameter 4/5 of total image width will cover 100% of a face, on avg
    a_max = np.pi * (r_max**2)  # covers 100% of face area
    a_covered = a_max * frac  # covers (frac)% of face area
    r = np.sqrt(a_covered / np.pi)

    # Center the face covering horizontally on faces
    left_pad = (n - 2 * r) / 2
    right_pad = (n - 2 * r) - left_pad
    # Center the covering on the face. Handle certain subjects differently
    _, image_fname = os.path.split(image_path)
    subject_num = "".join([el for el in image_fname if el.isnumeric()])
    top_pad = (5 / 8) * (
        m - 2 * r
    )  # top_pad > bottom_pad to account for subjects' hair
    bottom_pad = (m - 2 * r) - top_pad
    if subject_num == "06":
        left_pad = 0
        right_pad = n - 2 * r

    # Build face covering
    overlay = im.new("RGBA", img.size)
    shape = [(left_pad, top_pad), (n - right_pad, m - bottom_pad)]
    draw = ImageDraw.Draw(overlay)
    draw.ellipse(shape, fill="black")
    img = im.alpha_composite(img, overlay)
    img = img.convert("RGB")

    # Save face-masked image
    cwd = os.getcwd()
    pct = int(frac * 100)
    pct_dir = f"cover_{pct}_pct_faces"
    full_pct_dir = os.path.join(cwd, pct_dir)
    save_path = os.path.join(full_pct_dir, image_fname)
    img.save(save_path)
