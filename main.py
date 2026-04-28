import cv2
import numpy as np


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def remove_black_border(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    return img[y:y+h, x:x+w]


def stitch_pair(img_left, img_right):
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)

    kp1, des1 = orb.detectAndCompute(gray_left, None)
    kp2, des2 = orb.detectAndCompute(gray_right, None)

    if des1 is None or des2 is None:
        raise ValueError("Not enough features detected.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    if len(good_matches) < 10:
        raise ValueError("Not enough good matches.")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    M, inliers = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0
    )

    if M is None:
        raise ValueError("Affine estimation failed.")

    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]

    corners_left = np.float32([
        [0, 0],
        [0, h1],
        [w1, h1],
        [w1, 0]
    ]).reshape(-1, 1, 2)

    warped_corners_left = cv2.transform(corners_left, M)

    corners_right = np.float32([
        [0, 0],
        [0, h2],
        [w2, h2],
        [w2, 0]
    ]).reshape(-1, 1, 2)

    all_corners = np.concatenate((warped_corners_left, corners_right), axis=0)

    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

    tx = -x_min
    ty = -y_min

    M_translate = M.copy()
    M_translate[0, 2] += tx
    M_translate[1, 2] += ty

    result_width = x_max - x_min
    result_height = y_max - y_min

    warped_left = cv2.warpAffine(
        img_left,
        M_translate,
        (result_width, result_height)
    )

    result = warped_left.copy()

    x_offset = tx
    y_offset = ty

    roi = result[y_offset:y_offset + h2, x_offset:x_offset + w2]

    mask_right = img_right > 0

    blended = roi.copy()
    blended[mask_right] = img_right[mask_right]

    result[y_offset:y_offset + h2, x_offset:x_offset + w2] = blended

    result = remove_black_border(result)

    return result


def main():
    left = load_image("left.jpg")
    center = load_image("center.jpg")
    right = load_image("right.jpg")

    print("Stitching left + center...")
    panorama = stitch_pair(left, center)

    print("Stitching panorama + right...")
    panorama = stitch_pair(panorama, right)

    cv2.imwrite("panorama_result.jpg", panorama)

    cv2.imshow("Panorama Result", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Done. Saved as panorama_result.jpg")


if __name__ == "__main__":
    main()